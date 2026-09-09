# SPDX-License-Identifier: Apache-2.0
"""CPU-only SHM service for atomic recurrent checkpoint generations."""

# Standard
from pathlib import Path

# First Party
from lmcache.v1.distributed.admission import AdmissionFailure
from lmcache.v1.multiprocess.checkpoint_index import (
    CheckpointIndex,
    CheckpointManifest,
    CheckpointPrefix,
)
from lmcache.v1.multiprocess.checkpoint_storage import (
    CheckpointPayloadStore,
    CheckpointSlots,
    checkpoint_page_groups,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import HandlerSpec, ThreadPoolType
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.multiprocess.protocols.checkpoint import (
    CheckpointCapabilities,
    CheckpointLeaseResponse,
)


def _ready(slots: CheckpointSlots) -> CheckpointLeaseResponse:
    return CheckpointLeaseResponse(
        "ready",
        slots.lease_id,
        tuple(
            tuple(
                (slot.offset, slot.length) if slot is not None else (-1, 0)
                for slot in group
            )
            for group in slots.groups
        ),
    )


class CheckpointModule:
    """Publish complete all-rank manifests and lease their payload pages.

    Args:
        ctx: Existing CPU SHM storage context; no CUDA registration is created.
        index_path: SQLite file in a trusted, existing directory, or None for
            a RAM-only directory. Filesystem payloads alone do not imply a
            durable manifest directory.
        max_leases: Shared limit for pending rank stores and retrieves.

    Shutdown requires workers to finish or abort all submitted copy leases.
    A timeout must not recycle SHM while a worker can still access its bytes.
    """

    def __init__(
        self,
        ctx: MPCacheServerContext,
        index_path: Path | None = None,
        *,
        max_leases: int = 1024,
    ) -> None:
        if not ctx.shm_pool_info["shm_name"] or ctx.shm_pool_info["pool_size"] <= 0:
            raise ValueError("Recurrent checkpoint transfers require an SHM pool")
        self._ctx = ctx
        self._index = CheckpointIndex(index_path)
        self._payloads = CheckpointPayloadStore(
            ctx.storage_manager, self._index, max_leases=max_leases
        )
        self._capabilities = CheckpointCapabilities(
            format_version=1,
            shm_name=ctx.shm_pool_info["shm_name"],
            pool_size=ctx.shm_pool_info["pool_size"],
            durable_index=index_path is not None,
        )

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared server context without transferring ownership."""
        return self._ctx

    def get_handlers(self) -> list[HandlerSpec]:
        """Return metadata-only RPC handlers on the ordinary CPU thread pool."""
        handlers = (
            (RequestType.CHECKPOINT_FIND, self.find),
            (RequestType.CHECKPOINT_BEGIN, self.begin),
            (RequestType.CHECKPOINT_ABORT, self.abort),
            (RequestType.CHECKPOINT_PREPARE_STORE, self.prepare_store),
            (RequestType.CHECKPOINT_FINISH_STORE, self.finish_store),
            (RequestType.CHECKPOINT_BEGIN_RETRIEVE, self.begin_retrieve),
            (RequestType.CHECKPOINT_POLL_RETRIEVE, self.poll_retrieve),
            (RequestType.CHECKPOINT_FINISH_RETRIEVE, self.finish_retrieve),
            (RequestType.CHECKPOINT_CANCEL_RETRIEVE, self.cancel_retrieve),
        )
        return [
            HandlerSpec(
                RequestType.CHECKPOINT_CAPABILITIES,
                self.capabilities,
                ThreadPoolType.SYNC,
            ),
            *(
                HandlerSpec(request, handler, ThreadPoolType.NORMAL)
                for request, handler in handlers
            ),
        ]

    def capabilities(self) -> CheckpointCapabilities:
        """Return the supported format and exact SHM pool identity."""
        return self._capabilities

    def begin(self, manifest: CheckpointManifest) -> bool:
        """Stage a validated manifest; False means directory admission failed.

        The manifest's generation must be shared by all producer ranks. A
        changed or already published generation raises ValueError.
        """
        checkpoint_page_groups(manifest)
        return self._index.begin(manifest)

    def find(self, prefixes: tuple[CheckpointPrefix, ...]) -> CheckpointManifest | None:
        """Return the longest complete candidate for authenticated prefix roots.

        A candidate is not a cache hit until all payload ranks restore it.
        """
        return self._index.find(prefixes)

    def abort(self, generation: str) -> bool:
        """Prevent publication; rank copy leases still require explicit finish."""
        self._index.abort(generation)
        return True

    def prepare_store(
        self, manifest: CheckpointManifest, rank: int
    ) -> CheckpointLeaseResponse:
        """Return all writable rank slots or miss without partial admission.

        Invalid layouts, ranks or duplicate active rank stores raise ValueError.
        """
        admission = self._payloads.prepare_store(manifest, rank)
        if admission is AdmissionFailure.BUSY:
            return CheckpointLeaseResponse("busy")
        return (
            _ready(admission)
            if isinstance(admission, CheckpointSlots)
            else CheckpointLeaseResponse("miss")
        )

    def finish_store(self, lease_id: str, success: bool) -> bool:
        """Finish drained D2H work; True means all ranks published the manifest."""
        return self._payloads.finish_store(lease_id, success)

    def begin_retrieve(
        self, manifest: CheckpointManifest, rank: int
    ) -> CheckpointLeaseResponse:
        """Start an asynchronous all-page rank lookup or reject lease admission."""
        lease_id = self._payloads.begin_retrieve(manifest, rank)
        return (
            CheckpointLeaseResponse("pending", lease_id)
            if lease_id is not None
            else CheckpointLeaseResponse("miss")
        )

    def poll_retrieve(self, lease_id: str) -> CheckpointLeaseResponse:
        """Return pending, complete pinned slots, or an atomic rank miss.

        Unknown or completed leases raise KeyError. Missing data never yields
        partial slots; malformed byte layouts raise ValueError after cleanup.
        """
        result = self._payloads.poll_retrieve(lease_id)
        if isinstance(result, CheckpointSlots):
            return _ready(result)
        return CheckpointLeaseResponse(
            "pending" if result is None else "miss", lease_id
        )

    def finish_retrieve(self, lease_id: str) -> bool:
        """Release a read lease after H2D drains; pending lookup raises ValueError."""
        self._payloads.finish_retrieve(lease_id)
        return True

    def cancel_retrieve(self, lease_id: str) -> bool:
        """Cancel unexposed lookup; callers must poll until its locks drain.

        Once slots are exposed, use finish_retrieve after H2D completion instead.
        """
        self._payloads.cancel_retrieve(lease_id)
        return True

    def report_status(self) -> dict[str, dict[str, int]]:
        """Report capability and live leases without advancing or releasing work."""
        return {
            "recurrent_checkpoints": {
                "format_version": self._capabilities.format_version,
                "durable_index": self._capabilities.durable_index,
                **self._index.report_status(),
                **self._payloads.report_status(),
            }
        }

    def close(self) -> None:
        """Close the directory after copy leases drain, otherwise raise RuntimeError.

        This module never frees buffers solely because a copy took too long.
        The owning process shutdown must coordinate GPU worker termination.
        """
        status = self._payloads.report_status()
        if status["store_leases"] or status["retrieve_leases"]:
            raise RuntimeError("Checkpoint worker copy leases must drain before close")
        self._index.close()
