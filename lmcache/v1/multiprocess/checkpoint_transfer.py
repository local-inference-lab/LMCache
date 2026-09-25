# SPDX-License-Identifier: Apache-2.0
"""Background, metadata-only checkpoint RPC with explicit copy-lease ownership."""

# Standard
import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.multiprocess.checkpoint_index import CheckpointManifest
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.multiprocess.protocols.checkpoint import (
    CheckpointCapabilities,
    CheckpointLeaseResponse,
)

logger = init_logger(__name__)


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
        timeout = float(
            os.getenv("LMCACHE_CHECKPOINT_CLOSE_TIMEOUT", "180.0")
        )
        drain = threading.Thread(
            target=self._executor.shutdown,
            kwargs={"wait": True, "cancel_futures": False},
            daemon=True,
        )
        drain.start()
        drain.join(timeout)
        if drain.is_alive():
            # H3: the drain deadline is crossed, but the interpreter can
            # still wedge past this raise — concurrent.futures' atexit
            # hook joins the executor's non-daemon threads in
            # threading._shutdown, and a stuck copy blocks that join
            # forever. Arm a daemon watchdog BEFORE the raise: if
            # teardown has not completed within the hard ceiling, log
            # the force-exit banner and call os._exit(1). Daemon
            # threads are killed only at the very end of interpreter
            # finalization, and finalization is itself blocked by the
            # non-daemon join — the wedge keeps the watchdog alive, so
            # the _exit wins the race. Teardown-time only: this branch
            # is unreachable while serving.
            ceiling = float(
                os.getenv("LMCACHE_CHECKPOINT_FORCE_EXIT_CEILING", "60.0")
            )
            watchdog_deadline = time.monotonic() + ceiling

            def _force_exit() -> None:
                delay = watchdog_deadline - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                try:
                    pending = self._executor._work_queue.qsize()
                except BaseException:
                    pending = -1
                # The banner must not be loseable: write it to stderr
                # directly first (no locks, no buffering), then attempt
                # the logger (best effort — a wedged logging lock must
                # not prevent os._exit).
                banner = (
                    f"[STORM] checkpoint drain force-exit: drain deadline "
                    f"{timeout:.0f}s crossed, hard ceiling {ceiling:.0f}s "
                    f"elapsed, pending={pending}; calling os._exit(1)"
                )
                try:
                    os.write(2, (banner + chr(10)).encode())
                except BaseException:
                    pass
                try:
                    logger.error("%s", banner)
                except BaseException:
                    pass
                os._exit(1)

            threading.Thread(
                target=_force_exit,
                name="storm-drain-force-exit",
                daemon=True,
            ).start()
            # The executor keeps draining in the background; the late
            # case still surfaces via _unsafe or the process teardown.
            raise UnsafeCheckpointCopyError(
                f"Checkpoint copies did not drain within {timeout:.0f}s"
            )
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

        if lease.status in ("miss", "busy"):
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
        if store:
            deadline = time.monotonic() + self._rpc_timeout
            retry_delay = 0.001
            while lease.status == "busy":
                remaining = deadline - time.monotonic()
                if self._closing or remaining <= 0:
                    return False
                time.sleep(min(retry_delay, remaining))
                lease = self._call(request, job.manifest, job.rank)
                retry_delay = min(retry_delay * 2, 0.01)
        if not store:
            # Poll in the background; the scheduler can continue serving other
            # requests. A pending lookup owns no worker-visible byte slots yet.
            started = time.monotonic()
            deadline = started + self._rpc_timeout
            cancelled = False
            try:
                while lease.status == "pending":
                    if not cancelled and (
                        self._closing or time.monotonic() >= deadline
                    ):
                        if not self._closing:
                            logger.warning(
                                "Checkpoint retrieve of %d tokens for rank %d is "
                                "still waiting for storage after %.0f s; cancelling "
                                "it, so the request recomputes its prompt",
                                job.manifest.prefix.num_tokens,
                                job.rank,
                                time.monotonic() - started,
                            )
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
