# SPDX-License-Identifier: Apache-2.0
"""Background, metadata-only checkpoint RPC with explicit copy-lease ownership."""

# Standard
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal
import threading
import time

# First Party
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.multiprocess.checkpoint_index import CheckpointManifest
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.multiprocess.protocols.checkpoint import (
    CheckpointCapabilities,
    CheckpointLeaseResponse,
)


class UnsafeCheckpointCopyError(RuntimeError):
    """A copy or SHM lease could not drain; its resources must remain owned.

    This is a fatal worker error, not a cache miss. The serving engine must
    terminate affected workers before reclaiming either side of the transfer.
    """


class _DrainedCheckpointLeaseTimeout(LMCacheTimeoutError):
    """A late RPC reply was reclaimed without submitting a GPU copy."""


@dataclass(frozen=True)
class CheckpointTransferJob:
    """One rank's copy between immutable storage and caller-owned GPU pages.

    ``block_ids`` contain only the live physical IDs in manifest group order,
    including the auxiliary group. The caller pins every ID before submission
    and retains the pins until the completion future resolves successfully or
    with a recoverable failure. IDs never enter the persistent manifest.
    """

    manifest: CheckpointManifest
    rank: int
    direction: Literal["STORE", "RETRIEVE"]
    block_ids: tuple[tuple[int, ...], ...]
    producer_event: Any = None


class CheckpointTransferWorker:
    """Run checkpoint lookup, copy and lease completion off the model thread.

    Args:
        client: Shared thread-safe metadata queue client; ownership stays outside.
        copy_pages: Callback performing the entire copy and draining it before
            returning or raising. If draining fails, it must raise
            UnsafeCheckpointCopyError so no lease is released prematurely.
        workers: Maximum simultaneous copy tasks.
        max_pending: Admission limit including queued jobs; rejection starts no copy.
        rpc_timeout: Metadata reply deadline, in seconds. A timed-out lease
            acquisition or completion receives one additional deadline to reconcile
            ownership. Failure is fatal instead of abandoning its SHM reservation.

    Successful store completion acknowledges drained rank bytes, not all-rank
    manifest publication. The directory exclusively owns that publication.
    """

    def __init__(
        self,
        client: MessageQueueClient,
        copy_pages: Callable[[CheckpointTransferJob, CheckpointLeaseResponse], None],
        *,
        workers: int = 2,
        max_pending: int = 16,
        rpc_timeout: float = 30,
    ) -> None:
        if workers < 1 or max_pending < workers or rpc_timeout <= 0:
            raise ValueError(
                "Checkpoint worker capacities and timeout must be positive"
            )
        self._client = client
        self._copy_pages = copy_pages
        self._rpc_timeout = rpc_timeout
        self._executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="lmcache_checkpoint_copy"
        )
        self._capacity = threading.BoundedSemaphore(max_pending)
        self._lock = threading.Lock()
        self._closing = False
        self._unsafe = False

    def capabilities(self) -> CheckpointCapabilities:
        """Read the server's format and exact shared-memory identity.

        Returns:
            Server capabilities, without mapping or creating a GPU context.

        Raises:
            TimeoutError: If the server does not reply within rpc_timeout.
        """
        return self._call(RequestType.CHECKPOINT_CAPABILITIES)

    def submit(self, job: CheckpointTransferJob) -> Future[bool] | None:
        """Admit a rank transfer without blocking the model thread.

        Args:
            job: Immutable manifest and caller-pinned source/destination pages.

        Returns:
            Future resolving after the copy drains and the lease is finished,
            or None if capacity/shutdown rejects admission. False is a miss or
            cancelled store. Unsafe copy errors are fatal and retain ownership.

        Raises:
            ValueError: If direction or rank is invalid.
        """
        if job.direction not in ("STORE", "RETRIEVE") or not (
            0 <= job.rank < job.manifest.world_size
        ):
            raise ValueError("Invalid checkpoint transfer direction or rank")
        with self._lock:
            if (
                self._closing
                or self._unsafe
                or not self._capacity.acquire(blocking=False)
            ):
                return None
            try:
                future = self._executor.submit(self._run, job)
            except BaseException:
                self._capacity.release()
                raise
        future.add_done_callback(lambda _future: self._capacity.release())
        return future

    def close(self) -> None:
        """Drain admitted jobs without cancelling queued copies.

        Raises:
            UnsafeCheckpointCopyError: If any submitted copy could not drain.
                In that case shared mappings and GPU pages must remain alive
                until worker process termination.
        """
        with self._lock:
            self._closing = True
        self._executor.shutdown(wait=True, cancel_futures=False)
        if self._unsafe:
            raise UnsafeCheckpointCopyError("Checkpoint copies did not drain")

    def _call(self, request: RequestType, *payloads: object) -> Any:
        future: Any = self._client.submit_request(request, list(payloads))
        try:
            return future.result(timeout=self._rpc_timeout)
        except TimeoutError as timeout:
            if request in (
                RequestType.CHECKPOINT_FINISH_STORE,
                RequestType.CHECKPOINT_FINISH_RETRIEVE,
            ):
                try:
                    # Completion may already have released the lease. Reconcile
                    # its acknowledgement without submitting a second finish.
                    return future.result(timeout=self._rpc_timeout)
                except BaseException as drain_error:
                    with self._lock:
                        self._unsafe = True
                    raise UnsafeCheckpointCopyError(
                        "Checkpoint completion ownership could not drain"
                    ) from drain_error
            if request not in (
                RequestType.CHECKPOINT_PREPARE_STORE,
                RequestType.CHECKPOINT_BEGIN_RETRIEVE,
                RequestType.CHECKPOINT_POLL_RETRIEVE,
            ):
                raise
            try:
                # A local deadline does not cancel the server operation. Keep
                # the same future until its lease can be explicitly reclaimed.
                lease = future.result(timeout=self._rpc_timeout)
                self._discard_uncopied_lease(
                    lease, store=request == RequestType.CHECKPOINT_PREPARE_STORE
                )
            except BaseException as drain_error:
                with self._lock:
                    self._unsafe = True
                raise UnsafeCheckpointCopyError(
                    "Timed-out checkpoint lease ownership could not drain"
                ) from drain_error
            raise _DrainedCheckpointLeaseTimeout(str(timeout)) from timeout

    def _discard_uncopied_lease(
        self, lease: CheckpointLeaseResponse, *, store: bool
    ) -> None:
        """Release a lease whose slots never reached the GPU copy callback."""

        def call(request: RequestType, *payloads: object) -> Any:
            return self._client.submit_request(request, list(payloads)).result(
                timeout=self._rpc_timeout
            )

        if lease.status == "miss":
            return
        if store:
            call(RequestType.CHECKPOINT_FINISH_STORE, lease.lease_id, False)
            return
        if lease.status == "ready":
            call(RequestType.CHECKPOINT_FINISH_RETRIEVE, lease.lease_id)
            return
        call(RequestType.CHECKPOINT_CANCEL_RETRIEVE, lease.lease_id)
        deadline = time.monotonic() + self._rpc_timeout
        while lease.status == "pending":
            if time.monotonic() >= deadline:
                raise LMCacheTimeoutError(
                    "Cancelled checkpoint storage lookup did not drain"
                )
            time.sleep(0.001)
            lease = call(RequestType.CHECKPOINT_POLL_RETRIEVE, lease.lease_id)
        if lease.status == "ready":
            call(RequestType.CHECKPOINT_FINISH_RETRIEVE, lease.lease_id)

    def _run(self, job: CheckpointTransferJob) -> bool:
        store = job.direction == "STORE"
        request = (
            RequestType.CHECKPOINT_PREPARE_STORE
            if store
            else RequestType.CHECKPOINT_BEGIN_RETRIEVE
        )
        lease: CheckpointLeaseResponse = self._call(request, job.manifest, job.rank)
        if not store:
            # Poll in the background; the scheduler can continue serving other
            # requests. A pending lookup owns no worker-visible byte slots yet.
            deadline = time.monotonic() + self._rpc_timeout
            cancelled = False
            try:
                while lease.status == "pending":
                    if not cancelled and (
                        self._closing or time.monotonic() >= deadline
                    ):
                        self._call(
                            RequestType.CHECKPOINT_CANCEL_RETRIEVE, lease.lease_id
                        )
                        cancelled = True
                        deadline = time.monotonic() + self._rpc_timeout
                    if cancelled and time.monotonic() >= deadline:
                        raise LMCacheTimeoutError(
                            "Checkpoint lookup cancellation did not drain"
                        )
                    time.sleep(0.001)
                    lease = self._call(
                        RequestType.CHECKPOINT_POLL_RETRIEVE, lease.lease_id
                    )
            except (_DrainedCheckpointLeaseTimeout, UnsafeCheckpointCopyError):
                raise
            except BaseException:
                # No slots have reached the copy callback. The server can
                # safely cancel this lookup, but must drain its storage locks.
                try:
                    self._discard_uncopied_lease(lease, store=False)
                except BaseException as drain_error:
                    with self._lock:
                        self._unsafe = True
                    raise UnsafeCheckpointCopyError(
                        "Cancelled checkpoint lease ownership could not drain"
                    ) from drain_error
                raise
        if lease.status == "miss":
            return False
        if lease.status != "ready":
            raise ValueError("Checkpoint copy requires a complete ready lease")
        success = False
        safe_to_release = True
        try:
            self._copy_pages(job, lease)
            success = True
        except UnsafeCheckpointCopyError:
            safe_to_release = False
            with self._lock:
                self._unsafe = True
            raise
        finally:
            if safe_to_release:
                if store:
                    self._call(
                        RequestType.CHECKPOINT_FINISH_STORE, lease.lease_id, success
                    )
                else:
                    self._call(RequestType.CHECKPOINT_FINISH_RETRIEVE, lease.lease_id)
        return success
