# SPDX-License-Identifier: Apache-2.0
"""SHM payload leases for atomic recurrent checkpoint generations.

The storage manager owns RAM and filesystem tiering. Workers own CUDA copies;
this service neither imports a CUDA context nor serializes tensor data over MQ.
Every lease remains pinned until its worker reports completion of the copy.
"""

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING
import hashlib
import json
import threading
import uuid

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchHandle,
    PrefetchRequestSpec,
    TrimPolicy,
)
from lmcache.v1.multiprocess.checkpoint_index import CheckpointIndex, CheckpointManifest
from lmcache.v1.multiprocess.transfer_context.shm import ShmSlotDescriptor

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.storage_manager import StorageManager
    from lmcache.v1.memory_management import MemoryObj


@dataclass(frozen=True)
class CheckpointPageGroup:
    """Persistent byte layout and logical positions for one cache storage.

    A logical position is relative to its KV group, never a GPU block ID.
    ``name`` identifies the storage's semantic role and layer independently of
    allocator addresses. Only the explicitly listed pages are transferred.
    """

    name: str
    page_bytes: int
    positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name or len(self.name) > 256:
            raise ValueError("checkpoint storage name must contain 1..256 characters")
        if not 0 < self.page_bytes <= 1024**3:
            raise ValueError("checkpoint page size must be in 1 byte..1 GiB")
        if (
            not self.positions
            or len(self.positions) > 1048576
            or any(type(i) is not int or i < 0 for i in self.positions)
            or tuple(sorted(set(self.positions))) != self.positions
        ):
            raise ValueError(
                "checkpoint page positions must be nonempty and increasing"
            )


def checkpoint_page_groups(
    manifest: CheckpointManifest,
) -> tuple[CheckpointPageGroup, ...]:
    """Validate byte groups in a version-1 checkpoint payload manifest.

    Args:
        manifest: Manifest whose JSON payload contains ``page_groups`` entries
            with ``name``, ``page_bytes`` and ``positions`` fields.

    Returns:
        Ordered storage groups, whose order defines LMCache object-group IDs.

    Raises:
        ValueError: If required groups are absent, malformed or duplicated.
    """
    payload = json.loads(manifest.payload)
    rows = payload.get("page_groups")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 4096:
        raise ValueError("checkpoint manifest requires 1..4096 storage groups")
    try:
        groups = tuple(
            CheckpointPageGroup(row["name"], row["page_bytes"], tuple(row["positions"]))
            for row in rows
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid checkpoint storage group") from exc
    if len({group.name for group in groups}) != len(groups):
        raise ValueError("checkpoint storage names must be unique")
    return groups


def checkpoint_object_keys(
    manifest: CheckpointManifest, rank: int
) -> tuple[tuple[ObjectKey, ...], ...]:
    """Derive immutable payload keys without embedding tokens or file paths.

    Args:
        manifest: Complete generation descriptor and isolation namespace.
        rank: Physical tensor-parallel worker rank in the manifest's world.

    Returns:
        Per-group keys ordered by logical page position. Each key includes the
        generation, rank, storage role and namespace; generations cannot mix.

    Raises:
        ValueError: If rank or the manifest's byte layout is invalid.
    """
    if not 0 <= rank < manifest.world_size:
        raise ValueError("checkpoint payload rank is out of range")
    namespace = hashlib.sha256(manifest.prefix.namespace.encode()).hexdigest()
    groups = checkpoint_page_groups(manifest)
    return tuple(
        tuple(
            ObjectKey(
                hashlib.sha256(
                    json.dumps([manifest.generation, group.name, position]).encode()
                ).digest(),
                f"recurrent-checkpoint-v1-{namespace}",
                rank,
                group_id,
            )
            for position in group.positions
        )
        for group_id, group in enumerate(groups)
    )


def _slot(memory: "MemoryObj", expected_bytes: int) -> ShmSlotDescriptor:
    tensor = memory.tensor
    if (
        tensor is None
        or tensor.dtype != torch.uint8
        or tuple(tensor.shape) != (expected_bytes,)
        or memory.shm_offset < 0
        or memory.shm_byte_length != expected_bytes
    ):
        raise ValueError("checkpoint payload does not match its SHM byte layout")
    return ShmSlotDescriptor(
        memory.shm_offset, memory.shm_byte_length, [expected_bytes], "uint8"
    )


@dataclass(frozen=True)
class CheckpointSlots:
    """Pinned SHM descriptors grouped in the manifest's storage/page order."""

    lease_id: str
    groups: tuple[tuple[ShmSlotDescriptor, ...], ...]


@dataclass
class _StoreLease:
    manifest: CheckpointManifest
    rank: int
    keys: list[ObjectKey]


@dataclass
class _RetrieveLease:
    manifest: CheckpointManifest
    rank: int
    keys: list[ObjectKey]
    handle: PrefetchHandle
    slots: CheckpointSlots | None = None
    cancelled: bool = False


class CheckpointPayloadStore:
    """Lease checkpoint byte pages through an existing SHM storage manager.

    Args:
        storage: SHM-backed storage manager; the service does not own its life.
        index: Directory used for all-rank publication and stale invalidation.
        max_leases: Shared bound for pending stores and retrieves. Exhaustion
            rejects admission without recycling a worker's live SHM buffers.

    The caller stages a manifest with ``index.begin`` before rank stores. Each
    successful store acknowledgement follows a drained worker D2H transfer.
    Retrieval completion similarly follows H2D completion, not MQ delivery.
    """

    def __init__(
        self,
        storage: "StorageManager",
        index: CheckpointIndex,
        *,
        max_leases: int = 1024,
    ) -> None:
        if max_leases < 1:
            raise ValueError("checkpoint lease capacity must be positive")
        self._storage = storage
        self._index = index
        self._max_leases = max_leases
        self._stores: dict[str, _StoreLease] = {}
        self._store_ranks: set[tuple[str, int]] = set()
        self._retrieves: dict[str, _RetrieveLease] = {}
        self._retrieve_admissions: set[str] = set()
        self._lock = threading.Lock()

    def prepare_store(
        self, manifest: CheckpointManifest, rank: int
    ) -> CheckpointSlots | None:
        """Reserve all pages for one rank or release the partial reservation.

        Args:
            manifest: Staged immutable generation descriptor.
            rank: Producer rank.

        Returns:
            A pinned writable lease, or None if any page cannot be reserved.
            Admission failure cancels publication of the whole generation.

        Raises:
            ValueError: For invalid layouts or a duplicate producer rank.
        """
        groups = checkpoint_page_groups(manifest)
        key_groups = checkpoint_object_keys(manifest, rank)
        if not self._index.is_pending(manifest):
            return None
        identity = (manifest.generation, rank)
        with self._lock:
            if identity in self._store_ranks:
                raise ValueError("checkpoint rank already has a store lease")
            if (
                len(self._store_ranks)
                + len(self._retrieves)
                + len(self._retrieve_admissions)
                >= self._max_leases
            ):
                self._index.abort(manifest.generation)
                return None
            self._store_ranks.add(identity)
        reserved: list[ObjectKey] = []
        admitted = False
        try:
            slots = []
            for group, keys in zip(groups, key_groups, strict=True):
                objects = self._storage.reserve_write(
                    list(keys),
                    MemoryLayoutDesc([torch.Size([group.page_bytes])], [torch.uint8]),
                    "new",
                )
                reserved.extend(objects)
                if len(objects) != len(keys):
                    return None
                slots.append(
                    tuple(_slot(objects[key], group.page_bytes) for key in keys)
                )
            lease_id = uuid.uuid4().hex
            with self._lock:
                self._stores[lease_id] = _StoreLease(manifest, rank, reserved)
            admitted = True
            return CheckpointSlots(lease_id, tuple(slots))
        finally:
            if not admitted:
                try:
                    self._storage.abort_write(reserved)
                finally:
                    try:
                        self._index.abort(manifest.generation)
                    finally:
                        with self._lock:
                            self._store_ranks.discard(identity)

    def finish_store(self, lease_id: str, success: bool) -> bool:
        """Commit or discard pages after the producer's CUDA event completes.

        Args:
            lease_id: Writable lease returned by ``prepare_store``.
            success: Whether all page copies completed successfully.

        Returns:
            True only if this acknowledgement publishes the complete generation.
            Unknown or previously completed leases return False.
        """
        with self._lock:
            lease = self._stores.pop(lease_id, None)
        if lease is None:
            return False
        try:
            if not success:
                self._storage.abort_write(lease.keys)
                self._index.abort(lease.manifest.generation)
                return False
            self._storage.finish_write(lease.keys)
            if len(self._storage.get_readable_keys(lease.keys)) != len(lease.keys):
                self._index.abort(lease.manifest.generation)
                return False
            return self._index.acknowledge(lease.manifest.generation, lease.rank)
        except Exception:
            # Commit can fail before releasing every exclusive write lock.
            # abort_write leaves already-readable objects intact; none become
            # discoverable through this generation after publication is aborted.
            try:
                self._storage.abort_write(lease.keys)
            finally:
                self._index.abort(lease.manifest.generation)
            raise
        finally:
            with self._lock:
                self._store_ranks.remove((lease.manifest.generation, lease.rank))

    def begin_retrieve(self, manifest: CheckpointManifest, rank: int) -> str | None:
        """Start asynchronous RAM/filesystem lookup for every page of one rank.

        Args:
            manifest: Candidate selected by the checkpoint directory.
            rank: Consumer rank, using exactly the producer's parallel geometry.

        Returns:
            Lease identifier to poll, or None if the lease budget is exhausted.
            Admission does not indicate a successful cache hit.

        Raises:
            ValueError: If the manifest layout or rank is invalid.
        """
        groups = checkpoint_page_groups(manifest)
        keys = [
            key for group in checkpoint_object_keys(manifest, rank) for key in group
        ]
        layouts = {
            group_id: MemoryLayoutDesc([torch.Size([group.page_bytes])], [torch.uint8])
            for group_id, group in enumerate(groups)
        }
        lease_id = uuid.uuid4().hex
        with self._lock:
            if (
                len(self._store_ranks)
                + len(self._retrieves)
                + len(self._retrieve_admissions)
                >= self._max_leases
            ):
                return None
            self._retrieve_admissions.add(lease_id)
        try:
            handle = self._storage.submit_prefetch_task(
                PrefetchRequestSpec(keys, layouts, policy=TrimPolicy.SPARSE),
                external_request_id=f"checkpoint-{lease_id}",
            )
            with self._lock:
                self._retrieves[lease_id] = _RetrieveLease(manifest, rank, keys, handle)
                self._retrieve_admissions.remove(lease_id)
        finally:
            with self._lock:
                self._retrieve_admissions.discard(lease_id)
        return lease_id

    def report_status(self) -> dict[str, int]:
        """Return live lease counts without releasing worker-owned copy buffers.

        Store counts include reservations being prepared. Retrieve counts
        include prefetch submissions that have not yet returned their handle.
        A shutdown coordinator must drain these leases before closing storage.
        """
        with self._lock:
            return {
                "store_leases": len(self._store_ranks),
                "retrieve_leases": len(self._retrieves)
                + len(self._retrieve_admissions),
                "max_leases": self._max_leases,
            }

    def poll_retrieve(self, lease_id: str) -> CheckpointSlots | bool | None:
        """Return slots only when every required page is readable and pinned.

        Args:
            lease_id: Identifier from ``begin_retrieve``.

        Returns:
            None while prefetch is pending, False on a miss/cancellation, or
            the pinned slots. A miss releases all acquired locks and invalidates
            only the failed generation. It must not advance computed tokens.

        Raises:
            KeyError: If the lease is unknown or already finished.
            ValueError: If stored payload byte layouts do not match the manifest.
        """
        with self._lock:
            lease = self._retrieves[lease_id]
            if lease.slots is not None:
                return lease.slots
            found = self._storage.query_prefetch_status(lease.handle)
            if found is None:
                return None
            readable_keys = [key for i, key in enumerate(lease.keys) if found.test(i)]
            if len(readable_keys) != len(lease.keys) or lease.cancelled:
                self._storage.finish_read_prefetched(readable_keys)
                del self._retrieves[lease_id]
                if not lease.cancelled:
                    self._index.invalidate(lease.manifest.generation)
                return False
            try:
                keys, objects = self._storage.unsafe_read(lease.keys)
                if keys != lease.keys or len(objects) != len(keys):
                    raise ValueError("checkpoint pages changed during pinned retrieval")
                slots = []
                offset = 0
                for group in checkpoint_page_groups(lease.manifest):
                    count = len(group.positions)
                    slots.append(
                        tuple(
                            _slot(obj, group.page_bytes)
                            for obj in objects[offset : offset + count]
                        )
                    )
                    offset += count
                lease.slots = CheckpointSlots(lease_id, tuple(slots))
                return lease.slots
            except Exception:
                self._storage.finish_read_prefetched(lease.keys)
                self._index.invalidate(lease.manifest.generation)
                del self._retrieves[lease_id]
                raise

    def finish_retrieve(self, lease_id: str) -> None:
        """Release a prepared read lease after the consumer's CUDA event completes.

        Args:
            lease_id: Pinned lease whose H2D work has drained.

        Raises:
            ValueError: If lookup is still pending; use ``cancel_retrieve``.
        """
        with self._lock:
            lease = self._retrieves.get(lease_id)
            if lease is None:
                return
            if lease.slots is None:
                raise ValueError("checkpoint retrieval has not produced a read lease")
            del self._retrieves[lease_id]
        self._storage.finish_read_prefetched(lease.keys)

    def cancel_retrieve(self, lease_id: str) -> None:
        """Mark a pending lookup for draining without exposing its SHM slots.

        Args:
            lease_id: Lookup whose consumer was cancelled before H2D submission.
                The caller must keep polling until False releases lookup locks.

        Raises:
            ValueError: If slots were already exposed; their GPU copy must first
                drain and use ``finish_retrieve`` instead.
        """
        with self._lock:
            lease = self._retrieves[lease_id]
            if lease.slots is not None:
                raise ValueError(
                    "prepared checkpoint retrieval requires copy completion"
                )
            lease.cancelled = True
