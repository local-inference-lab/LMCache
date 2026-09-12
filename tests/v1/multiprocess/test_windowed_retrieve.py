# SPDX-License-Identifier: Apache-2.0
"""Windowed retrieve on the asynchronous engine-driven transfer context.

``submit_windowed_retrieve`` restores ``[key.start, key.end)`` in chunk
windows off the forward thread: per window it asks the server to load the
window (RESTORE_WINDOW), waits for the load (WAIT_PREFETCH_STATUS), retrieves
the loaded prefix (PREPARE/COMMIT_RETRIEVE) and scatters it. These tests drive
the loop with a scripted server: ordering and lookahead of the window loads,
the partial stop with lock release, cancellation, and progress reporting.
"""

# Standard
from contextlib import nullcontext
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.protocols.engine import RestoreWindowResponse
from lmcache.v1.multiprocess.transfer_context import async_engine_driven
from lmcache.v1.multiprocess.transfer_context.async_engine_driven import (
    AsyncEngineDrivenTransferContext,
)
from lmcache.v1.multiprocess.transfer_context.shm import EngineDrivenContextShm

CHUNK = 8
BLOCKS_IN_CHUNK = 2


class _FakeEvent:
    def record(self, stream: object | None = None) -> None:
        return None

    def synchronize(self) -> None:
        return None

    def wait(self, stream: object | None = None) -> None:
        return None


class _FakeTorchDev:
    def __init__(self) -> None:
        self._stream = SimpleNamespace(device="cpu")

    def Stream(self) -> object:
        return self._stream

    def stream(self, stream: object) -> object:
        return nullcontext(stream)

    def Event(self, **_kwargs: Any) -> _FakeEvent:
        return _FakeEvent()

    def set_device(self, _device: object) -> None:
        return None

    def synchronize(self) -> None:
        return None


@dataclass
class _ServerScript:
    """Scripted server answers keyed by window start token."""

    pinned_chunk_end: int = 0
    known: bool = True
    # window start -> chunks that load (default: all submitted)
    loaded: dict[int, int] | None = None
    # window start -> number of WAIT polls that return None first
    slow_polls: dict[int, int] | None = None
    # window starts whose PREPARE_RETRIEVE misses
    miss: set[int] | None = None


class _FakeMQClient:
    """Answers the restore protocol from a script and records every call."""

    def __init__(self, script: _ServerScript) -> None:
        self.script = script
        self.calls: list[tuple[RequestType, list[Any]]] = []
        self._polls: dict[str, int] = {}
        self.lock = threading.Lock()

    def submit_request(self, request_type, payload, response_cls=None):
        with self.lock:
            self.calls.append((request_type, payload))
        future: MessagingFuture[Any] = MessagingFuture()
        future.set_result(self._answer(request_type, payload))
        return future

    def _answer(self, request_type, payload):
        script = self.script
        if request_type is RequestType.RESTORE_WINDOW:
            key = payload[0]
            if not script.known:
                return RestoreWindowResponse(known=False)
            start_chunk = max(key.start // CHUNK, script.pinned_chunk_end)
            end_chunk = key.end // CHUNK
            submitted = max(0, end_chunk - start_chunk)
            return RestoreWindowResponse(
                known=True,
                pinned_chunk_end=script.pinned_chunk_end,
                submitted_chunks=submitted,
                job_id=f"job-{key.start}" if submitted else "",
            )
        if request_type is RequestType.WAIT_PREFETCH_STATUS:
            job_id = payload[0]
            start = int(job_id.split("-")[1])
            slow = (script.slow_polls or {}).get(start, 0)
            polls = self._polls.get(job_id, 0)
            self._polls[job_id] = polls + 1
            if polls < slow:
                return None
            key_end_chunks = None
            for rt, pl in self.calls:
                if rt is RequestType.RESTORE_WINDOW and pl[0].start == start:
                    key_end_chunks = pl[0].end // CHUNK
            assert key_end_chunks is not None
            submitted = key_end_chunks - max(start // CHUNK, script.pinned_chunk_end)
            return (script.loaded or {}).get(start, submitted)
        if request_type is RequestType.FREE_LOOKUP_LOCKS:
            return None
        raise AssertionError(f"unexpected request {request_type}")


class _FakeShmContext(EngineDrivenContextShm):
    """SHM context whose server calls are answered by the fake MQ client."""

    def __init__(self, mq: _FakeMQClient, miss: set[int]) -> None:  # noqa: D107
        # Skip EngineDrivenContextShm.__init__ (needs a real SHM segment).
        self.mq_client = mq
        self.mq_timeout = 5.0
        self.metadata = SimpleNamespace(group_layouts=None)
        self.prepared: list[tuple[int, int]] = []
        self.committed: list[tuple[int, int]] = []
        self._miss = miss

    def prepare_retrieve(self, key, instance_id):
        self.prepared.append((key.start, key.end))
        if key.start in self._miss:
            return None
        chunks = (key.end - key.start) // CHUNK
        return [torch.zeros(1) for _ in range(chunks)]

    def commit_retrieve(self, key, instance_id):
        self.committed.append((key.start, key.end))
        return True

    def close(self) -> None:
        return None


def _key(start: int, end: int) -> IPCCacheServerKey:
    return IPCCacheServerKey(
        model_name="model",
        world_size=1,
        worker_id=0,
        token_ids=tuple(range(end)),
        start=start,
        end=end,
        request_id="req-1",
    )


@pytest.fixture
def scatter_calls(monkeypatch):
    calls: list[dict[str, Any]] = []

    def fake_scatter(kv_caches, block_ids, chunks, blocks_per_chunk, **kwargs):
        calls.append(
            {
                "block_ids": list(block_ids),
                "chunks": len(chunks),
                "skip": kwargs.get("skip_first_n_tokens", 0),
            }
        )

    monkeypatch.setattr(async_engine_driven, "scatter_cpu_to_paged_kv", fake_scatter)
    monkeypatch.setattr(async_engine_driven, "torch_dev", _FakeTorchDev())
    return calls


def _context(script: _ServerScript) -> tuple[AsyncEngineDrivenTransferContext, Any]:
    ctx = AsyncEngineDrivenTransferContext(commit_workers=1)
    mq = _FakeMQClient(script)
    ctx._engine_driven_context = _FakeShmContext(mq, script.miss or set())
    ctx._external_chunk_size = CHUNK
    ctx._group_states = []  # single-group path
    return ctx, mq


def _block_ids(start: int, end: int) -> list[list[int]]:
    chunks = (end - start) // CHUNK
    return [list(range(100, 100 + chunks * BLOCKS_IN_CHUNK))]


def _windows_submitted(mq: _FakeMQClient) -> list[int]:
    return [pl[0].start for rt, pl in mq.calls if rt is RequestType.RESTORE_WINDOW]


def _freed(mq: _FakeMQClient) -> list[tuple[int, int]]:
    return [
        (pl[0].start, pl[0].end)
        for rt, pl in mq.calls
        if rt is RequestType.FREE_LOOKUP_LOCKS
    ]


def test_restores_every_window_in_order_with_one_window_of_lookahead(
    scatter_calls,
):
    script = _ServerScript(pinned_chunk_end=2)  # chunks 0-1 pinned by lookup
    ctx, mq = _context(script)
    key = _key(0, 6 * CHUNK)  # 6 chunks, windows of 2 -> 3 windows

    future = ctx.submit_windowed_retrieve(
        "req-1",
        key,
        7,
        {},
        _block_ids(0, 6 * CHUNK),
        _FakeEvent(),
        BLOCKS_IN_CHUNK,
        skip_first_n_tokens=3,
        window_chunks=2,
        tp_size=1,
        readers_per_object=1,
    )
    assert future.result(timeout=5.0) is True
    shm = ctx._engine_driven_context
    assert shm.prepared == [(0, 16), (16, 32), (32, 48)]
    assert shm.committed == shm.prepared
    # Window 0 is fully pinned (no load); windows 1 and 2 were loaded, and
    # window 1's load was submitted before window 0 was scattered.
    assert _windows_submitted(mq) == [0, 16, 32]
    first_wait = next(
        i for i, (rt, _) in enumerate(mq.calls) if rt is RequestType.WAIT_PREFETCH_STATUS
    )
    assert _windows_submitted(mq)[:2] == [0, 16]
    assert mq.calls[first_wait][1][0] == "job-16"
    # Only the first window skips the already-computed prefix.
    assert [c["skip"] for c in scatter_calls] == [3, 0, 0]
    assert [c["block_ids"] for c in scatter_calls] == [
        [100, 101, 102, 103],
        [104, 105, 106, 107],
        [108, 109, 110, 111],
    ]
    assert _freed(mq) == []
    assert ctx.pop_retrieve_progress("req-1") == 48
    ctx.close()


def test_partial_window_stops_the_restore_and_releases_unretrieved_locks(
    scatter_calls,
):
    # Window [16, 32) loads only 1 of its 2 chunks; window [32, 48) was
    # already submitted (lookahead) and its 2 loaded chunks must be released.
    script = _ServerScript(pinned_chunk_end=0, loaded={16: 1})
    ctx, mq = _context(script)
    key = _key(0, 6 * CHUNK)

    future = ctx.submit_windowed_retrieve(
        "req-1", key, 7, {}, _block_ids(0, 6 * CHUNK), _FakeEvent(),
        BLOCKS_IN_CHUNK, window_chunks=2, tp_size=1, readers_per_object=1,
    )
    assert future.result(timeout=5.0) is False
    shm = ctx._engine_driven_context
    assert shm.prepared == [(0, 16), (16, 24)]
    assert ctx.pop_retrieve_progress("req-1") == 24
    # The lookahead window's loaded chunks [32, 48) hold locks -> released;
    # nothing of the pinned prefix (none) or the failed tail is touched.
    assert _freed(mq) == [(32, 48)]
    assert [c["chunks"] for c in scatter_calls] == [2, 1]
    ctx.close()


def test_prepare_miss_counts_as_partial(scatter_calls):
    script = _ServerScript(pinned_chunk_end=0, miss={16})
    ctx, mq = _context(script)
    key = _key(0, 4 * CHUNK)

    future = ctx.submit_windowed_retrieve(
        "req-1", key, 7, {}, _block_ids(0, 4 * CHUNK), _FakeEvent(),
        BLOCKS_IN_CHUNK, window_chunks=2, tp_size=1, readers_per_object=1,
    )
    assert future.result(timeout=5.0) is False
    assert ctx.pop_retrieve_progress("req-1") == 16
    shm = ctx._engine_driven_context
    # The miss is still committed so the server drops its pending read.
    assert shm.committed == [(0, 16), (16, 32)]
    ctx.close()


def test_unknown_request_stops_before_loading(scatter_calls):
    script = _ServerScript(known=False)
    ctx, mq = _context(script)
    key = _key(0, 4 * CHUNK)

    future = ctx.submit_windowed_retrieve(
        "req-1", key, 7, {}, _block_ids(0, 4 * CHUNK), _FakeEvent(),
        BLOCKS_IN_CHUNK, window_chunks=2, tp_size=1, readers_per_object=1,
    )
    assert future.result(timeout=5.0) is False
    assert ctx.pop_retrieve_progress("req-1") == 0
    assert ctx._engine_driven_context.prepared == []
    assert scatter_calls == []
    ctx.close()


def test_cancel_releases_the_pinned_remainder(scatter_calls):
    # Every WAIT for window [16, 32) first returns None (still loading);
    # cancel during that wait; the loaded chunks of the window and the
    # lookup-pinned chunks past the written range are released.
    script = _ServerScript(pinned_chunk_end=1, slow_polls={16: 2})
    ctx, mq = _context(script)
    key = _key(0, 4 * CHUNK)
    gate = threading.Event()
    original_wait = ctx._wait_restore_window

    def cancelling_wait(request_id, window):
        if window.start == 16 and not gate.is_set():
            gate.set()
            ctx.cancel_retrieve(request_id)
        return original_wait(request_id, window)

    ctx._wait_restore_window = cancelling_wait  # type: ignore[method-assign]

    future = ctx.submit_windowed_retrieve(
        "req-1", key, 7, {}, _block_ids(0, 4 * CHUNK), _FakeEvent(),
        BLOCKS_IN_CHUNK, window_chunks=2, tp_size=1, readers_per_object=1,
    )
    assert future.result(timeout=5.0) is False
    assert ctx.pop_retrieve_progress("req-1") == 16
    # Window 0 (chunks 0-1: chunk 0 pinned, chunk 1 loaded) was written;
    # window 1's loaded chunks [16, 32) are released after the cancel.
    assert (16, 32) in _freed(mq)
    ctx.close()


def test_falls_back_to_the_synchronous_retrieve_without_shm(monkeypatch):
    ctx = AsyncEngineDrivenTransferContext(commit_workers=1)
    ctx._engine_driven_context = SimpleNamespace(mq_client=None, close=lambda: None)  # not SHM
    ctx._external_chunk_size = CHUNK
    calls = []
    monkeypatch.setattr(
        ctx, "submit_retrieve", lambda *a, **k: calls.append((a, k)) or MessagingFuture()
    )
    ctx.submit_windowed_retrieve(
        "req-1", _key(0, 16), 7, {}, [[1, 2]], _FakeEvent(), BLOCKS_IN_CHUNK,
        window_chunks=2, tp_size=1, readers_per_object=1,
    )
    assert len(calls) == 1
    ctx.close()


def test_window_key_carries_the_reader_count(scatter_calls):
    script = _ServerScript(pinned_chunk_end=0)
    ctx, mq = _context(script)
    key = _key(0, 2 * CHUNK)
    future = ctx.submit_windowed_retrieve(
        "req-1", key, 7, {}, _block_ids(0, 2 * CHUNK), _FakeEvent(),
        BLOCKS_IN_CHUNK, window_chunks=2, tp_size=4, readers_per_object=2,
    )
    assert future.result(timeout=5.0) is True
    restore_calls = [pl for rt, pl in mq.calls if rt is RequestType.RESTORE_WINDOW]
    assert restore_calls[0][0] == replace(key, readers_per_object=2)
    assert restore_calls[0][1] == 4
    ctx.close()
