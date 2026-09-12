# SPDX-License-Identifier: Apache-2.0
"""Async engine-driven data transfer context for multiprocess worker adapters."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any
import os
import threading
import time

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class
from lmcache.v1.multiprocess.transfer_context.base import (
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
from lmcache.v1.multiprocess.transfer_context.pickle import (
    EngineDrivenContextPickle,
)
from lmcache.v1.multiprocess.transfer_context.shm import EngineDrivenContextShm
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
    IPCEvent,
    _collapse_chunks_for_single_destination,
    _drop_skipped_chunks,
    _single_group_block_ids,
)

_ASYNC_MULTIGROUP_STORE = os.environ.get("LMCACHE_ASYNC_MULTIGROUP_STORE", "1") == "1"

logger = init_logger(__name__)

# Number of background threads used to run commit (CPU->server) work for the
# async engine-driven store path. >1 so that a slow gather for one store does
# not block the commit of another store whose gather already finished.
DEFAULT_ENGINE_DRIVEN_COMMIT_WORKERS = 4


# Poll period of one WAIT_PREFETCH_STATUS round trip while a restore window
# loads; bounds how long a server thread blocks per call.
_RESTORE_WAIT_POLL_SECONDS = 0.25


@dataclass
class _RestoreWindow:
    """One window of a windowed retrieve: a chunk-aligned token range of the
    request's external prefix that is loaded, scattered and committed as a
    unit.

    Attributes:
        start: First token of the window (chunk-aligned).
        end: Exclusive last token (chunk-aligned).
        known: False when the server no longer tracks the request's lookup;
            the restore stops at this window.
        pinned_end_chunk: Chunks below this index were loaded and read-locked
            by the request's lookup and need no wait.
        submitted_chunks: Chunks the server submitted for loading from L2.
        job_id: Prefetch job to wait on when ``submitted_chunks`` is positive.
        loaded_chunks: Chunks of the submitted part that loaded, once known.
    """

    start: int
    end: int
    known: bool = True
    pinned_end_chunk: int = 0
    submitted_chunks: int = 0
    job_id: str = ""
    loaded_chunks: int = -1

class AsyncEngineDrivenTransferContext(EngineDrivenTransferContext):
    """Fully async engine-driven data transfer context (store-only async).

    "Store-only async" means ``submit_store`` returns an *unresolved* future
    that resolves only after the deferred gather (GPU->CPU copy) and commit
    (CPU->server) both complete off the forward thread, while
    ``submit_retrieve`` stays synchronous and returns an already-resolved
    future exactly as on the base context.

    Inherits :class:`EngineDrivenTransferContext` and reuses its
    ``register()`` (layout / SHM registration, no stream dependency) and
    ``submit_retrieve()`` (this path does not change retrieve). Only the store
    is made async.

    Store is three-phase, all executed entirely in a background thread:

    1. prepare: call prepare_store() to negotiate buffers with the server
       (the costliest step in pickle mode due to the synchronous RPC round-trip).
    2. gather: wait for the forward event on the copy stream, then enqueue
       GPU->CPU copies. When SHM buffers are available, gather writes directly
       into SHM views (matching the synchronous path). Otherwise, gather
       targets pinned staging buffers.
    3. commit: wait for gather completion (via a recorded CUDA event), then
       perform commit_store() and resolve the returned future.

    ``submit_store`` performs only O(1) work on the forward thread (registration
    check and block-id flattening) before submitting all three phases to the
    background ``commit_executor``, so the forward thread is never blocked by
    the RPC round-trip or gather kernel launch latency.

    This class is only instantiated by the factory when the device is
    async-capable, so the constructor creates async resources unconditionally;
    there is no ``self._async_capable`` flag.

    ``submit_windowed_retrieve`` is the one asynchronous retrieve: it restores
    an external prefix in chunk windows off the forward thread (server-side
    window loads from L2 overlap the scatter of the previous window), so a
    prefix larger than the L1 pin limit is restored in full instead of being
    truncated to what L1 holds, and the forward thread never waits for it.
    """

    def __init__(
        self,
        commit_workers: int = DEFAULT_ENGINE_DRIVEN_COMMIT_WORKERS,
    ) -> None:
        """Initialize the async context and create its async resources.

        Args:
            commit_workers: Number of background threads used to run commit
                (CPU->server) work. >1 so a slow gather for one store does not
                block the commit of another whose gather is already done.
        """
        super().__init__()
        self._commit_workers = max(1, int(commit_workers))
        self._copy_stream: Any = torch_dev.Stream()
        self._commit_executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=self._commit_workers,
            thread_name_prefix="lmcache_engine_driven_commit",
        )
        self._inflight_lock = threading.Lock()
        self._inflight_gather_events: set[Any] = set()
        self._inflight_overwrite_gather_events: set[Any] = set()
        # Tracks gather tasks that have been submitted to _commit_executor but
        # have not yet recorded their CUDA event. flush_inflight_stores waits
        # on all of these before synchronizing _inflight_gather_events, closing
        # the window where preemption could overwrite paged KV blocks before an
        # in-flight gather has had a chance to record its CUDA event.
        self._pending_stores: set[threading.Event] = set()
        # Recurrent and sliding-window groups can update a physical page in the
        # next forward. Publish their marker/event before stable paged-attention
        # groups finish copying so only the true source hazard blocks compute.
        self._pending_overwrite_stores: set[threading.Event] = set()
        # Serializes commit_store calls across worker threads, since the
        # underlying ZMQ socket is not thread-safe and commit_workers defaults
        # to >1.
        self._commit_lock = threading.Lock()
        self._staging_pool: dict[
            tuple[tuple[int, ...], torch.dtype], list[torch.Tensor]
        ] = {}
        self._is_closing = False
        # Windowed retrieves: one restore at a time per worker on its own
        # stream, so window scatters never queue behind store gathers.
        self._restore_stream: Any = torch_dev.Stream()
        self._restore_executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="lmcache_windowed_restore"
        )
        self._restore_lock = threading.Lock()
        self._cancelled_retrieves: set[str] = set()
        self._retrieve_progress: dict[str, int] = {}

    def _alloc_pinned_staging(
        self, shape: torch.Size, dtype: torch.dtype, count: int
    ) -> list[torch.Tensor]:
        """Allocate pinned (page-locked) staging tensors for GPU->CPU copies.

        Tensors are reused from the pool when available to avoid repeated
        allocations on the hot path.

        Args:
            shape: Tensor shape to allocate.
            dtype: Tensor dtype to allocate.
            count: Number of tensors needed.

        Returns:
            List of ``count`` pinned CPU tensors.
        """
        key = (tuple(shape), dtype)
        with self._inflight_lock:
            pooled = self._staging_pool.setdefault(key, [])
            staged = [pooled.pop() for _ in range(min(len(pooled), count))]
        if len(staged) == count:
            return staged

        missing = count - len(staged)
        for _ in range(missing):
            try:
                staged.append(
                    torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
                )
            except RuntimeError:
                # Graceful fallback for CPU-only / pin-memory-disabled setups.
                logger.warning(
                    "Falling back to non-pinned CPU staging buffer "
                    "(shape=%s, dtype=%s)",
                    tuple(shape),
                    dtype,
                )
                staged.append(torch.empty(shape, dtype=dtype, device="cpu"))
        return staged

    def _release_staging(self, chunks: list[torch.Tensor]) -> None:
        """Return staging tensors to the pool for reuse.

        Args:
            chunks: Tensors previously obtained from :meth:`_alloc_pinned_staging`.
        """
        if not chunks:
            return
        key = (tuple(chunks[0].shape), chunks[0].dtype)
        with self._inflight_lock:
            self._staging_pool.setdefault(key, []).extend(chunks)

    def submit_store(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        _event: IPCEvent,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Three-phase async store (prepare, gather and commit all in background).

        Performs only O(1) work on the forward thread (registration check and
        block-id flattening), then submits all three phases — prepare_store,
        gather (GPU->CPU), and commit — to the background ``commit_executor``.
        Returns an unresolved future that resolves only after all three phases
        complete.

        Args:
            _request_id: External request identifier (used for logging).
            key: LMCache key object for the store range.
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            block_ids: vLLM block IDs to store, indexed by LMCache KV group id.
            _event: Synchronization event; ``wait()`` is called in background.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.

        Returns:
            An unresolved :class:`MessagingFuture` that resolves to ``True``
            on success, ``False`` on failure.

        Raises:
            RuntimeError: If register() was not called first.
        """
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Engine-driven transfer context is not registered. "
                "Call register() before submit_store()."
            )
        if self._group_states:
            # Multi-group stores run the same three-phase background pipeline
            # as single-group stores unless disabled by env.
            if _ASYNC_MULTIGROUP_STORE and not isinstance(
                self._engine_driven_context, EngineDrivenContextPickle
            ):
                return self._submit_store_multigroup_async(
                    _request_id, key, instance_id, kv_caches, block_ids, _event
                )
            _event.wait()
            return self._submit_store_multigroup(key, instance_id, kv_caches, block_ids)
        completion: MessagingFuture[bool] = MessagingFuture()
        engine_driven_context = self._engine_driven_context
        commit_executor = self._commit_executor

        # Signals when this task has recorded its CUDA event (or exited early),
        # allowing flush_inflight_stores to safely proceed.
        gather_launched = threading.Event()
        try:
            with self._inflight_lock:
                if self._is_closing:
                    completion.set_result(False)
                    return completion
                self._pending_stores.add(gather_launched)

            full_block_ids = _single_group_block_ids(block_ids)

            def _prepare_gather_and_commit() -> None:
                gather_done: Any | None = None
                ok = False
                # Whether we gathered directly into SHM views (True) or into
                # pinned staging buffers that need to be released later (False).
                used_shm_direct = False
                staged_chunks: list[torch.Tensor] = []
                try:
                    # --- Phase 1: prepare_store ---
                    # In pickle mode this is the costliest step (sync RPC
                    # round-trip).  Running it here keeps the forward thread free.
                    result = engine_driven_context.prepare_store(key, instance_id)
                    out_buffers, chunk_indices = (
                        result if result is not None else (None, None)
                    )

                    if chunk_indices is not None and len(chunk_indices) == 0:
                        # All chunks are already in cache: no gather, no commit.
                        ok = True
                        return

                    num_chunks = (
                        len(chunk_indices)
                        if chunk_indices is not None
                        else len(full_block_ids) // blocks_in_chunk
                    )

                    # Determine gather target:
                    # - SHM path (out_buffers available): gather into SHM views
                    # - Pickle path (no out_buffers): gather into pinned staging
                    if out_buffers is not None:
                        gather_target = out_buffers
                        used_shm_direct = True
                    else:
                        layout_desc = engine_driven_context.layout_desc
                        if not layout_desc.shapes:
                            raise RuntimeError(
                                "engine-driven layout_desc.shapes is empty"
                            )
                        if not layout_desc.dtypes:
                            raise RuntimeError(
                                "engine-driven layout_desc.dtypes is empty"
                            )
                        staged_chunks = self._alloc_pinned_staging(
                            layout_desc.shapes[0],
                            layout_desc.dtypes[0],
                            num_chunks,
                        )
                        gather_target = staged_chunks

                    # --- Phase 2: gather (GPU->CPU copy on copy stream) ---
                    with torch.inference_mode(), torch_dev.stream(self._copy_stream):
                        _event.wait(stream=self._copy_stream)

                        gather_paged_kv_to_cpu(
                            kv_caches,
                            full_block_ids,
                            blocks_in_chunk,
                            layout_hints=self._layout_hints,
                            engine_kv_format=self._engine_kv_format,
                            out=gather_target,
                            chunk_indices=chunk_indices,
                        )

                        gather_done = torch_dev.Event()
                        gather_done.record(self._copy_stream)

                    with self._inflight_lock:
                        if gather_done is not None:
                            self._inflight_gather_events.add(gather_done)
                        self._pending_stores.discard(gather_launched)
                    gather_launched.set()

                    if gather_done is not None:
                        gather_done.synchronize()

                    # --- Phase 3: commit ---
                    with self._commit_lock:
                        ok = engine_driven_context.commit_store(
                            key, instance_id, gather_target
                        )

                    if not ok:
                        logger.error(
                            "Async engine-driven commit_store failed for request_id=%s",
                            _request_id,
                        )
                except Exception:
                    logger.exception(
                        "Async engine-driven store failed for request_id=%s",
                        _request_id,
                    )
                    ok = False
                finally:
                    if not used_shm_direct:
                        self._release_staging(staged_chunks)
                    with self._inflight_lock:
                        if gather_done is not None:
                            self._inflight_gather_events.discard(gather_done)
                        self._pending_stores.discard(gather_launched)
                    gather_launched.set()
                    completion.set_result(ok)

            # Submitting the task is the ownership-transfer point: once it
            # succeeds, the closure is solely responsible for releasing staging
            # buffers and resolving the future. The except below therefore only
            # handles failures that occur *before* this submit.
            commit_executor.submit(_prepare_gather_and_commit)
        except Exception:
            logger.exception("Failed to submit async engine-driven store")
            with self._inflight_lock:
                self._pending_stores.discard(gather_launched)
            gather_launched.set()
            completion.set_result(False)
            return completion

        return completion

    def _submit_store_multigroup_async(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        _event: IPCEvent,
    ) -> MessagingFuture:
        """Three-phase background store for multi-group (hybrid) KV layouts.

        Mirrors `_submit_store_multigroup` (same prepare/gather/commit calls
        and per-group slot mapping) but off the forward thread: the copy
        stream waits on the forward's CUDA event instead of a device
        synchronize, and the gather's completion event is tracked in
        `_inflight_gather_events` so `flush_inflight_stores` blocks before
        vLLM may overwrite the paged blocks. PREPARE_STORE and COMMIT hold
        `_commit_lock` so stores of one worker stay ordered on the socket.
        """
        ctx = self._engine_driven_context
        assert ctx is not None
        completion: MessagingFuture[bool] = MessagingFuture()
        if len(block_ids) != len(self._group_states):
            raise RuntimeError(
                f"got {len(block_ids)} block-id lists for "
                f"{len(self._group_states)} registered groups"
            )
        # Validate and project every group's inputs on the forward thread so a
        # malformed store fails before any server-side slot is reserved.
        transfer_inputs = [
            self._group_transfer_inputs(state, key, kv_caches, block_ids[gid])
            for gid, state in enumerate(self._group_states)
        ]
        group_states = list(self._group_states)
        overwrite_group_ids = [
            gid for gid, state in enumerate(group_states) if state.overwrites_in_place
        ]
        stable_group_ids = [
            gid
            for gid, state in enumerate(group_states)
            if not state.overwrites_in_place
        ]
        gather_launched = threading.Event()
        overwrite_gather_launched = threading.Event()
        try:
            with self._inflight_lock:
                if self._is_closing:
                    completion.set_result(False)
                    return completion
                self._pending_stores.add(gather_launched)
                if overwrite_group_ids:
                    self._pending_overwrite_stores.add(overwrite_gather_launched)

            def _prepare_gather_and_commit_grouped() -> None:
                gather_done: Any | None = None
                ok = False
                overwrite_gather_done: Any | None = None
                overwrite_gather_started = False
                try:
                    # Executor threads start on CUDA device 0; pin this thread to
                    # the worker's device so the stream context never creates
                    # a context on another rank's GPU (observed as CUDA OOM on
                    # ranks 1-7 when restoring the previous stream).
                    torch_dev.set_device(self._copy_stream.device)
                    with self._commit_lock:
                        result = ctx.prepare_store_grouped(key, instance_id)
                    if result is None:
                        return
                    tensors, chunk_indices, group_ids = result
                    if not tensors:
                        ok = True
                        return

                    def _gather_groups(group_ids_to_gather: list[int]) -> None:
                        for gid in group_ids_to_gather:
                            state = group_states[gid]
                            out_g, chunks_g = self._group_slots(
                                tensors, group_ids, gid, chunk_indices
                            )
                            if not out_g:
                                continue
                            transfer_kv_caches, transfer_block_ids = transfer_inputs[
                                gid
                            ]
                            gather_paged_kv_to_cpu(
                                transfer_kv_caches,
                                transfer_block_ids,
                                state.blocks_in_chunk,
                                layout_hints=self._layout_hints,
                                engine_kv_format=state.engine_kv_format,
                                out=out_g,
                                chunk_indices=chunks_g,
                                blocks_per_window=state.blocks_per_window,
                            )

                    with torch.inference_mode(), torch_dev.stream(self._copy_stream):
                        _event.wait(stream=self._copy_stream)
                        overwrite_gather_started = bool(overwrite_group_ids)
                        _gather_groups(overwrite_group_ids)
                        if overwrite_group_ids:
                            overwrite_gather_done = torch_dev.Event()
                            overwrite_gather_done.record(self._copy_stream)
                            with self._inflight_lock:
                                self._inflight_overwrite_gather_events.add(
                                    overwrite_gather_done
                                )
                                self._pending_overwrite_stores.discard(
                                    overwrite_gather_launched
                                )
                            overwrite_gather_launched.set()
                        _gather_groups(stable_group_ids)
                        gather_done = torch_dev.Event()
                        gather_done.record(self._copy_stream)
                    with self._inflight_lock:
                        self._inflight_gather_events.add(gather_done)
                        self._pending_stores.discard(gather_launched)
                    gather_launched.set()
                    gather_done.synchronize()
                    with self._commit_lock:
                        ok = ctx.commit_store(key, instance_id, [])
                    if not ok:
                        logger.error(
                            "Async grouped commit_store failed for request_id=%s",
                            _request_id,
                        )
                except Exception:
                    logger.exception(
                        "Async grouped engine-driven store failed for request_id=%s",
                        _request_id,
                    )
                    ok = False
                finally:
                    if overwrite_gather_started and overwrite_gather_done is None:
                        try:
                            torch_dev.synchronize()
                        except Exception:
                            logger.exception(
                                "Failed to drain an incomplete overwrite gather"
                            )
                    with self._inflight_lock:
                        if gather_done is not None:
                            self._inflight_gather_events.discard(gather_done)
                        if overwrite_gather_done is not None:
                            self._inflight_overwrite_gather_events.discard(
                                overwrite_gather_done
                            )
                        self._pending_stores.discard(gather_launched)
                        self._pending_overwrite_stores.discard(
                            overwrite_gather_launched
                        )
                    gather_launched.set()
                    overwrite_gather_launched.set()
                    completion.set_result(ok)

            self._commit_executor.submit(_prepare_gather_and_commit_grouped)
        except Exception:
            logger.exception("Failed to submit async grouped engine-driven store")
            with self._inflight_lock:
                self._pending_stores.discard(gather_launched)
                self._pending_overwrite_stores.discard(overwrite_gather_launched)
            gather_launched.set()
            overwrite_gather_launched.set()
            completion.set_result(False)
            return completion
        return completion

    # ------------------------------------------------------------------
    # Windowed retrieve
    # ------------------------------------------------------------------

    def cancel_retrieve(self, request_id: str) -> None:
        """Stop a windowed retrieve at its next window boundary.

        Called when the engine finished the request while its load was still
        in flight. The restore releases the locks of windows it will not
        retrieve and resolves its future as failed; the engine keeps the
        request's blocks until that future is reported.
        """
        with self._restore_lock:
            self._cancelled_retrieves.add(request_id)

    def pop_retrieve_progress(self, request_id: str) -> int | None:
        """Return and forget the exclusive token end a windowed retrieve
        wrote into the paged cache, or None when the request had no windowed
        retrieve. Blocks at or past this token were not written."""
        with self._restore_lock:
            return self._retrieve_progress.pop(request_id, None)

    def submit_windowed_retrieve(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
        *,
        window_chunks: int,
        tp_size: int,
        readers_per_object: int,
    ) -> MessagingFuture:
        """Retrieve ``[key.start, key.end)`` in chunk windows off the forward
        thread.

        Each window is loaded by the server (RESTORE_WINDOW + WAIT), scattered
        on the restore stream and committed; the next window's load is
        submitted before the current one is scattered so disk reads overlap
        the copies. The returned future resolves True when every chunk was
        written; otherwise False, with ``pop_retrieve_progress`` reporting the
        exclusive token end that was written so the adapter invalidates only
        the remainder. Falls back to the synchronous retrieve when the
        transport is not SHM or the chunk geometry is unknown.

        Args:
            request_id: External request identifier.
            key: Worker key of the full retrieve range.
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            block_ids: Block IDs per LMCache group covering the range.
            event: Forward-step event (unused: the range's blocks are not
                read by any forward until the load is reported).
            blocks_in_chunk: Paged blocks per LMCache chunk (single-group).
            skip_first_n_tokens: Leading tokens of the range not to write.
            window_chunks: Chunks per window (positive).
            tp_size: Tensor-parallel size forwarded to RESTORE_WINDOW.
            readers_per_object: Workers retrieving each object; carried in
                the window keys so the server takes one lock per reader.
        """
        del event
        ctx = self._engine_driven_context
        chunk = self._external_chunk_size
        if (
            not isinstance(ctx, EngineDrivenContextShm)
            or window_chunks <= 0
            or chunk <= 0
            or key.start % chunk
            or key.end % chunk
            or key.end <= key.start
        ):
            return self.submit_retrieve(
                request_id,
                key,
                instance_id,
                kv_caches,
                block_ids,
                None,  # type: ignore[arg-type]
                blocks_in_chunk,
                skip_first_n_tokens,
            )
        if self._group_states and len(block_ids) != len(self._group_states):
            raise RuntimeError(
                f"got {len(block_ids)} block-id lists for "
                f"{len(self._group_states)} registered groups"
            )
        windows = [
            _RestoreWindow(start=ws, end=min(ws + window_chunks * chunk, key.end))
            for ws in range(key.start, key.end, window_chunks * chunk)
        ]
        completion: MessagingFuture[bool] = MessagingFuture()
        with self._restore_lock:
            if self._is_closing:
                completion.set_result(False)
                return completion
            self._retrieve_progress[request_id] = key.start
            self._cancelled_retrieves.discard(request_id)
        self._restore_executor.submit(
            self._run_windowed_retrieve,
            request_id,
            key,
            instance_id,
            kv_caches,
            block_ids,
            blocks_in_chunk,
            skip_first_n_tokens,
            windows,
            tp_size,
            readers_per_object,
            completion,
        )
        return completion

    def _is_retrieve_cancelled(self, request_id: str) -> bool:
        with self._restore_lock:
            return self._is_closing or request_id in self._cancelled_retrieves

    def _mq_call(self, request_type: RequestType, payload: list[Any]) -> Any:
        """One request to the server, serialized with the store commits."""
        ctx = self._engine_driven_context
        assert isinstance(ctx, EngineDrivenContextShm)
        with self._commit_lock:
            future = ctx.mq_client.submit_request(
                request_type, payload, get_response_class(request_type)
            )
        return future.result(timeout=ctx.mq_timeout)

    def _submit_restore_window(
        self, key: Any, window: _RestoreWindow, tp_size: int, readers: int
    ) -> None:
        wkey = replace(
            key, start=window.start, end=window.end, readers_per_object=readers
        )
        response = self._mq_call(RequestType.RESTORE_WINDOW, [wkey, tp_size])
        window.known = bool(response.known)
        window.pinned_end_chunk = int(response.pinned_chunk_end)
        window.submitted_chunks = int(response.submitted_chunks)
        window.job_id = str(response.job_id)
        if window.submitted_chunks == 0:
            window.loaded_chunks = 0

    def _wait_restore_window(self, request_id: str, window: _RestoreWindow) -> int:
        """Block until the window's submitted chunks loaded (or the wait is
        cancelled); returns the loaded chunk count of the submitted part."""
        if window.loaded_chunks >= 0:
            return window.loaded_chunks
        ctx = self._engine_driven_context
        assert isinstance(ctx, EngineDrivenContextShm)
        deadline = time.monotonic() + max(ctx.mq_timeout, 1.0) * 4
        while True:
            result = self._mq_call(
                RequestType.WAIT_PREFETCH_STATUS,
                [window.job_id, _RESTORE_WAIT_POLL_SECONDS],
            )
            if result is not None:
                window.loaded_chunks = max(0, int(result))
                return window.loaded_chunks
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"restore window {window.job_id} did not load within "
                    f"{deadline} s"
                )
            # A cancelled restore still consumes the job so the server drops
            # it, but it stops as soon as the job resolves.
            if self._is_retrieve_cancelled(request_id):
                continue

    def _free_range_locks(self, key: Any, start: int, end: int, tp_size: int) -> None:
        """Release this rank's read locks on ``[start, end)`` (chunk-aligned)."""
        if end <= start:
            return
        try:
            self._mq_call(
                RequestType.FREE_LOOKUP_LOCKS,
                [replace(key, start=start, end=end), tp_size],
            )
        except Exception:
            logger.exception(
                "Failed to release restore locks [%d, %d) for request_id=%s",
                start,
                end,
                key.request_id,
            )

    def _scatter_window(
        self,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        blocks_in_chunk: int,
        start: int,
        end: int,
        skip_first_n_tokens: int,
    ) -> bool:
        """Retrieve ``[start, end)`` from L1 slots into the paged cache.

        Returns True when the server served every chunk of every group and
        the scatter completed; the read locks are released either way.
        """
        ctx = self._engine_driven_context
        assert isinstance(ctx, EngineDrivenContextShm)
        chunk = self._external_chunk_size
        first_chunk = (start - key.start) // chunk
        last_chunk = (end - key.start) // chunk
        rkey = replace(key, start=start, end=end)
        ok = False
        try:
            with torch.inference_mode(), torch_dev.stream(self._restore_stream):
                if self._group_states:
                    with self._commit_lock:
                        result = ctx.prepare_retrieve_grouped(rkey, instance_id)
                    if result is not None:
                        tensors, group_ids = result
                        for gid, state in enumerate(self._group_states):
                            bic = state.blocks_in_chunk
                            group_block_ids = block_ids[gid][
                                first_chunk * bic : last_chunk * bic
                            ]
                            src_g, _ = self._group_slots(tensors, group_ids, gid)
                            src_g, kept_ids, dropped = _drop_skipped_chunks(
                                src_g, group_block_ids, bic
                            )
                            if not src_g:
                                continue
                            group_start = rkey.start + dropped * chunk
                            group_skip = max(0, skip_first_n_tokens - dropped * chunk)
                            transfer_kv_caches, transfer_ids = (
                                self._group_transfer_inputs(
                                    state,
                                    rkey,
                                    kv_caches,
                                    kept_ids,
                                    start_token_idx=group_start,
                                )
                            )
                            physical_skip = self._physical_skip_tokens(
                                state, group_skip
                            )
                            if physical_skip == 0:
                                src_g, transfer_ids = (
                                    _collapse_chunks_for_single_destination(
                                        src_g,
                                        transfer_ids,
                                        state.blocks_in_chunk,
                                        state.blocks_per_window,
                                    )
                                )
                            scatter_cpu_to_paged_kv(
                                transfer_kv_caches,
                                transfer_ids,
                                src_g,
                                state.blocks_in_chunk,
                                skip_first_n_tokens=physical_skip,
                                layout_hints=self._layout_hints,
                                engine_kv_format=state.engine_kv_format,
                                blocks_per_window=state.blocks_per_window,
                            )
                        ok = True
                else:
                    with self._commit_lock:
                        src_buffers = ctx.prepare_retrieve(rkey, instance_id)
                    if src_buffers is not None:
                        single_ids = _single_group_block_ids(block_ids)[
                            first_chunk * blocks_in_chunk : last_chunk * blocks_in_chunk
                        ]
                        src_buffers, single_ids, dropped = _drop_skipped_chunks(
                            src_buffers, single_ids, blocks_in_chunk
                        )
                        single_skip = 0 if dropped else skip_first_n_tokens
                        if src_buffers:
                            scatter_cpu_to_paged_kv(
                                kv_caches,
                                single_ids,
                                src_buffers,
                                blocks_in_chunk,
                                skip_first_n_tokens=single_skip,
                                layout_hints=self._layout_hints,
                                engine_kv_format=self._engine_kv_format,
                            )
                        ok = True
                done = torch_dev.Event()
                done.record(self._restore_stream)
            # The server may reuse the slots right after commit: finish the
            # device writes first (this thread only; no device-wide sync).
            done.synchronize()
        except (RuntimeError, ValueError, TypeError, IndexError):
            logger.exception(
                "Failed to scatter restore window [%d, %d) for request_id=%s",
                start,
                end,
                key.request_id,
            )
            ok = False
        finally:
            with self._commit_lock:
                ctx.commit_retrieve(rkey, instance_id)
        return ok

    def _run_windowed_retrieve(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        blocks_in_chunk: int,
        skip_first_n_tokens: int,
        windows: list[_RestoreWindow],
        tp_size: int,
        readers: int,
        completion: MessagingFuture,
    ) -> None:
        chunk = self._external_chunk_size
        loaded_end = key.start
        submitted = 0
        complete = False
        pinned_end_chunk = 0
        try:
            torch_dev.set_device(self._restore_stream.device)
            # One window of lookahead keeps the disk busy while a window is
            # scattered; L1 holds at most two windows of this restore.
            lookahead = 1
            for index, window in enumerate(windows):
                while submitted < len(windows) and submitted <= index + lookahead:
                    self._submit_restore_window(
                        key, windows[submitted], tp_size, readers
                    )
                    submitted += 1
                    if not windows[submitted - 1].known:
                        break
                if not window.known or self._is_retrieve_cancelled(request_id):
                    break
                pinned_end_chunk = window.pinned_end_chunk
                total_chunks = (window.end - window.start) // chunk
                pinned_chunks = min(
                    total_chunks,
                    max(0, window.pinned_end_chunk - window.start // chunk),
                )
                loaded = self._wait_restore_window(request_id, window)
                usable = min(total_chunks, pinned_chunks + loaded)
                if self._is_retrieve_cancelled(request_id):
                    # Locks of the loaded part are released with the tail.
                    self._free_range_locks(
                        key, window.start, window.start + usable * chunk, tp_size
                    )
                    break
                if usable > 0:
                    usable_end = window.start + usable * chunk
                    scattered = self._scatter_window(
                        key,
                        instance_id,
                        kv_caches,
                        block_ids,
                        blocks_in_chunk,
                        window.start,
                        usable_end,
                        skip_first_n_tokens if index == 0 else 0,
                    )
                    if not scattered:
                        break
                    loaded_end = usable_end
                    with self._restore_lock:
                        self._retrieve_progress[request_id] = loaded_end
                if usable < total_chunks:
                    break
            else:
                complete = loaded_end == key.end
        except Exception:
            logger.exception(
                "Windowed retrieve failed for request_id=%s at token %d",
                request_id,
                loaded_end,
            )
        finally:
            if not complete:
                # Release what will not be retrieved: the lookup's pinned
                # prefix past the written range and every window load that
                # was submitted (its loaded prefix holds this rank's locks).
                # Loads still in flight are awaited first so no lock is taken
                # after its release.
                self._free_range_locks(
                    key,
                    loaded_end,
                    min(key.end, pinned_end_chunk * chunk),
                    tp_size,
                )
                for window in windows[:submitted]:
                    if not window.known or window.submitted_chunks == 0:
                        continue
                    try:
                        loaded = self._wait_restore_window(request_id, window)
                    except Exception:
                        logger.exception(
                            "Could not settle restore window %s", window.job_id
                        )
                        continue
                    load_start = max(window.start, window.pinned_end_chunk * chunk)
                    load_end = min(window.end, load_start + loaded * chunk)
                    if load_end > loaded_end:
                        self._free_range_locks(
                            key, max(load_start, loaded_end), load_end, tp_size
                        )
            with self._restore_lock:
                self._cancelled_retrieves.discard(request_id)
                self._retrieve_progress[request_id] = loaded_end
            logger.info(
                "Windowed retrieve %s for request_id=%s: wrote tokens [%d, %d) "
                "of [%d, %d) in %d window(s)",
                "completed" if complete else "stopped",
                request_id,
                key.start,
                loaded_end,
                key.start,
                key.end,
                len(windows),
            )
            completion.set_result(complete)

    def flush_inflight_stores(self) -> None:
        """Synchronize all in-flight gather (GPU->CPU) events.

        Called at preemption/eviction time so that vLLM cannot overwrite
        paged KV blocks before a deferred gather has finished reading them.

        Waits for all submitted-but-not-yet-launched stores to record their
        CUDA events before synchronizing those events, preventing a race where
        ``flush_inflight_stores`` returns before a background gather has
        started.
        """
        with self._inflight_lock:
            pending = list(self._pending_stores)
        for ev in pending:
            ev.wait()
        self._sync_gather_events(suppress_errors=False)

    def flush_inflight_overwrite_gathers(self) -> None:
        """Wait for snapshots whose source pages can be overwritten in place.

        Grouped stores publish this fence after recurrent/window groups reach
        host memory but before stable attention groups or the server commit
        necessarily finish. A next-forward boundary can call this method to
        protect in-place state without disabling asynchronous stores.
        """
        with self._inflight_lock:
            pending = list(self._pending_overwrite_stores)
        for marker in pending:
            marker.wait()
        if not self._group_states:
            self.flush_inflight_stores()
            return
        with self._inflight_lock:
            events = list(self._inflight_overwrite_gather_events)
        for event in events:
            event.synchronize()

    def close(self) -> None:
        """Drain in-flight gather/commit work before closing the base context."""
        with self._inflight_lock:
            self._is_closing = True
            pending = list(self._pending_stores)
        for ev in pending:
            ev.wait()
        self._sync_gather_events(suppress_errors=True)
        self._commit_executor.shutdown(wait=True, cancel_futures=False)
        with self._restore_lock:
            self._cancelled_retrieves.update(self._retrieve_progress.keys())
        self._restore_executor.shutdown(wait=True, cancel_futures=True)
        super().close()

    def _sync_gather_events(self, suppress_errors: bool = False) -> None:
        """Synchronize all in-flight gather (GPU->CPU) events.

        Args:
            suppress_errors: If True, log exceptions instead of propagating.
        """
        with self._inflight_lock:
            gather_events = list(self._inflight_gather_events)
        for event in gather_events:
            try:
                event.synchronize()
            except Exception:
                if not suppress_errors:
                    raise
                logger.exception("Failed while draining gather events")
