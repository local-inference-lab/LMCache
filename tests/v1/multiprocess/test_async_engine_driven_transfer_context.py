# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable
from unittest.mock import MagicMock
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.transfer_context import (
    async_engine_driven,
    worker_transfer,
)
from lmcache.v1.multiprocess.transfer_context.async_engine_driven import (
    AsyncEngineDrivenTransferContext,
)
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
)


@dataclass
class _FakeStoreContext:
    """Minimal engine-driven context for async store tests."""

    commit_impl: Callable[[list[torch.Tensor]], bool]
    prepare_result: tuple[list[torch.Tensor], list[int]] | None = None
    prepare_impl: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        self.layout_desc = SimpleNamespace(
            shapes=[torch.Size([2, 1, 1, 1])], dtypes=[torch.float32]
        )

    def prepare_store(
        self, _key: object, _instance_id: int
    ) -> tuple[list[torch.Tensor], list[int]] | None:
        if self.prepare_impl is not None:
            self.prepare_impl()
        return self.prepare_result

    def commit_store(
        self, _key: object, _instance_id: int, chunks: list[torch.Tensor]
    ) -> bool:
        return bool(self.commit_impl(chunks))

    def abort_store(self, _key: object, _instance_id: int) -> bool:
        return True

    def close(self) -> None:
        return None


@dataclass
class _FakeGroupedStoreContext:
    """Minimal shared-memory context for async hybrid-KV store tests."""

    commit_impl: Callable[[list[list[torch.Tensor]]], bool]
    prepare_result: tuple[list[list[torch.Tensor]], list[list[int]]] | None
    prepare_impl: Callable[[], None] | None = None

    def prepare_store_grouped(
        self, _key: object, _instance_id: int
    ) -> tuple[list[list[torch.Tensor]], list[list[int]]] | None:
        if self.prepare_impl is not None:
            self.prepare_impl()
        return self.prepare_result

    def commit_store_grouped(
        self, _key: object, _instance_id: int, chunks: list[list[torch.Tensor]]
    ) -> bool:
        return bool(self.commit_impl(chunks))

    def abort_store(self, _key: object, _instance_id: int) -> bool:
        return True

    def close(self) -> None:
        return None


class _FakeEvent:
    def __init__(self, gate: threading.Event):
        self._gate = gate

    def record(self, stream: object | None = None) -> None:
        return None

    def wait(self, stream: object | None = None) -> None:
        return None

    def synchronize(self) -> None:
        self._gate.wait(timeout=2)

    def query(self) -> bool:
        return self._gate.is_set()

    def ipc_handle(self) -> object:
        return object()


class _FakeTorchDev:
    def __init__(self, gather_gate: threading.Event):
        self._stream = object()
        self._gather_gate = gather_gate

    def Stream(self) -> object:
        return object()

    def stream(self, stream: object) -> object:
        return nullcontext(stream)

    def current_stream(self) -> object:
        return self._stream

    def Event(self, interprocess: bool = False) -> _FakeEvent:
        return _FakeEvent(self._gather_gate)

    def synchronize(self) -> None:
        pass


def _install_fake_gather(monkeypatch: pytest.MonkeyPatch) -> None:
    def _gather(
        _kv_caches: dict[str, torch.Tensor],
        _block_ids: list[int],
        _blocks_in_chunk: int,
        **kwargs: object,
    ) -> list[torch.Tensor]:
        out = kwargs.get("out")
        if out is None:
            return [torch.ones(1)]
        assert isinstance(out, list)
        for tensor in out:
            tensor.fill_(1.0)
        return out

    # Patch gather in both modules so either path is exercised.
    monkeypatch.setattr(async_engine_driven, "gather_paged_kv_to_cpu", _gather)
    monkeypatch.setattr(worker_transfer, "gather_paged_kv_to_cpu", _gather)


def _new_context(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gather_gate: threading.Event,
    commit_impl: Callable[[list[torch.Tensor]], bool],
    max_inflight: int = 8,
) -> AsyncEngineDrivenTransferContext:
    monkeypatch.setattr(async_engine_driven, "torch_dev", _FakeTorchDev(gather_gate))
    _install_fake_gather(monkeypatch)
    ctx = AsyncEngineDrivenTransferContext(commit_workers=max_inflight)
    ctx._engine_driven_context = (
        _FakeStoreContext(commit_impl=commit_impl)  # type: ignore[assignment]
    )
    return ctx


def _install_two_group_state(ctx: AsyncEngineDrivenTransferContext) -> None:
    """Install two minimal hybrid-KV group descriptors on a test context."""
    ctx._worker_groups = (
        worker_transfer._EngineDrivenWorkerGroup(
            layout=SimpleNamespace(
                blocks_per_chunk=1,
                blocks_per_window=1,
                object_group_id=0,
            ),
            layer_names=("layer_0",),
            engine_kv_format=object(),
        ),
        worker_transfer._EngineDrivenWorkerGroup(
            layout=SimpleNamespace(
                blocks_per_chunk=2,
                blocks_per_window=1,
                object_group_id=1,
            ),
            layer_names=("layer_1",),
            engine_kv_format=object(),
        ),
    )  # type: ignore[assignment]


def test_submit_store_returns_pending_future_until_gather_and_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gather_gate = threading.Event()
    ctx = _new_context(
        monkeypatch, gather_gate=gather_gate, commit_impl=lambda _c: True
    )
    future = ctx.submit_store(
        "r1", object(), 1, {"k": torch.zeros(1)}, [[0]], _FakeEvent(gather_gate), 1
    )
    assert not future.query()
    gather_gate.set()
    assert future.result(timeout=1) is True
    ctx.close()


def test_submit_store_commit_waits_for_gather_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gather_gate = threading.Event()
    commit_called = threading.Event()

    def _commit(_chunks: list[torch.Tensor]) -> bool:
        commit_called.set()
        return True

    ctx = _new_context(monkeypatch, gather_gate=gather_gate, commit_impl=_commit)
    future = ctx.submit_store(
        "r1", object(), 1, {"k": torch.zeros(1)}, [[0]], _FakeEvent(gather_gate), 1
    )
    assert not commit_called.wait(timeout=0.05)
    gather_gate.set()
    assert future.result(timeout=1) is True
    assert commit_called.is_set()
    ctx.close()


def test_close_drains_inflight_async_store(monkeypatch: pytest.MonkeyPatch) -> None:
    gather_gate = threading.Event()
    commit_gate = threading.Event()
    gather_gate.set()

    def _commit(_chunks: list[torch.Tensor]) -> bool:
        commit_gate.wait(timeout=2)
        return True

    ctx = _new_context(monkeypatch, gather_gate=gather_gate, commit_impl=_commit)
    future = ctx.submit_store(
        "r1", object(), 1, {"k": torch.zeros(1)}, [[0]], _FakeEvent(gather_gate), 1
    )
    closed = threading.Event()

    def _close() -> None:
        ctx.close()
        closed.set()

    t = threading.Thread(target=_close, daemon=True)
    t.start()
    assert not closed.wait(timeout=0.05)
    commit_gate.set()
    t.join(timeout=1)
    assert closed.is_set()
    assert future.result(timeout=1) is True


def test_commit_failure_sets_false_and_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gather_gate = threading.Event()

    def _commit(_chunks: list[torch.Tensor]) -> bool:
        raise RuntimeError("commit failed")

    log_exception = MagicMock()
    monkeypatch.setattr(async_engine_driven.logger, "exception", log_exception)
    ctx = _new_context(monkeypatch, gather_gate=gather_gate, commit_impl=_commit)
    future = ctx.submit_store(
        "r1", object(), 1, {"k": torch.zeros(1)}, [[0]], _FakeEvent(gather_gate), 1
    )
    gather_gate.set()
    assert future.result(timeout=1) is False
    log_exception.assert_called_once()
    ctx.close()


def test_flush_inflight_stores_no_inflight_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gather_gate = threading.Event()
    gather_gate.set()
    ctx = _new_context(
        monkeypatch, gather_gate=gather_gate, commit_impl=lambda _c: True
    )
    # No in-flight events: flush is a cheap no-op and must not raise.
    ctx.flush_inflight_stores()
    ctx.close()


def test_sync_engine_driven_context_returns_resolved_future(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RecordingTorchDev:
        def __init__(self) -> None:
            self.synchronize_calls = 0

        def synchronize(self) -> None:
            self.synchronize_calls += 1

    fake = _RecordingTorchDev()
    monkeypatch.setattr(worker_transfer, "torch_dev", fake)
    _install_fake_gather(monkeypatch)
    ctx = EngineDrivenTransferContext()
    ctx._engine_driven_context = (
        _FakeStoreContext(commit_impl=lambda _c: True)  # type: ignore[assignment]
    )

    future = ctx.submit_store(
        "r1",
        object(),
        1,
        {"k": torch.zeros(1)},
        [[0]],
        _FakeEvent(threading.Event()),
        1,
    )

    # Sync path resolves inline.
    assert future.query()
    assert future.result(timeout=1) is True
    assert fake.synchronize_calls >= 1
    # flush_inflight_stores is the inherited base no-op; must not raise.
    ctx.flush_inflight_stores()
    ctx.close()


def test_sync_engine_driven_context_has_no_async_resources() -> None:
    ctx = EngineDrivenTransferContext()
    assert not hasattr(ctx, "_copy_stream")
    assert not hasattr(ctx, "_commit_executor")
    assert not hasattr(ctx, "_inflight_semaphore")
    # close() on an unregistered sync context must not raise.
    ctx.close()


def test_build_engine_driven_context_dispatches_on_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker_transfer, "_supports_async_primitives", lambda: True)
    # Avoid touching real stream/event primitives when instantiating the async
    # context returned by the capable branch.
    monkeypatch.setattr(
        async_engine_driven, "torch_dev", _FakeTorchDev(threading.Event())
    )
    capable = worker_transfer._build_engine_driven_context()
    assert isinstance(capable, AsyncEngineDrivenTransferContext)
    capable.close()

    monkeypatch.setattr(worker_transfer, "_supports_async_primitives", lambda: False)
    fallback = worker_transfer._build_engine_driven_context()
    assert isinstance(fallback, EngineDrivenTransferContext)
    assert not isinstance(fallback, AsyncEngineDrivenTransferContext)
    fallback.close()


def test_supports_async_primitives_false_without_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A torch_dev without Stream/Event is not async-capable.
    monkeypatch.setattr(worker_transfer, "torch_dev", object())
    assert worker_transfer._supports_async_primitives() is False


def test_flush_inflight_stores_waits_for_pending_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """flush_inflight_stores must not return before a submitted-but-not-yet-
    launched gather has recorded its CUDA event (preemption race condition)."""
    gather_gate = threading.Event()
    flush_returned = threading.Event()
    gather_started = threading.Event()

    # Gate that blocks gather from completing until we are ready.
    class _BlockingFakeTorchDev(_FakeTorchDev):
        def stream(self, stream: object) -> object:
            gather_started.set()
            # Block here until the test releases the gate — simulating the
            # background thread not yet recording the CUDA event.
            gather_gate.wait(timeout=2)
            return super().stream(stream)

    # Pass gather_gate so _FakeEvent.synchronize() also uses the same gate.
    monkeypatch.setattr(
        async_engine_driven, "torch_dev", _BlockingFakeTorchDev(gather_gate)
    )
    _install_fake_gather(monkeypatch)

    ctx = AsyncEngineDrivenTransferContext(commit_workers=1)
    ctx._engine_driven_context = (
        _FakeStoreContext(commit_impl=lambda _c: True)  # type: ignore[assignment]
    )

    ctx.submit_store(
        "r1", object(), 1, {"k": torch.zeros(1)}, [[0]], _FakeEvent(gather_gate), 1
    )

    # Wait until the background thread has started (entered stream()), proving
    # it has not yet recorded its CUDA event.
    assert gather_started.wait(timeout=1), "background thread never started"

    def _flush() -> None:
        ctx.flush_inflight_stores()
        flush_returned.set()

    t = threading.Thread(target=_flush, daemon=True)
    t.start()

    # flush_inflight_stores must NOT return while the gather is still pending.
    assert not flush_returned.wait(timeout=0.05), (
        "flush_inflight_stores returned before gather launched — race condition!"
    )

    # Now let the background gather proceed; flush should complete shortly.
    gather_gate.set()
    t.join(timeout=2)
    assert flush_returned.is_set(), "flush_inflight_stores did not complete"
    ctx.close()


def test_commit_store_serialized_by_commit_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent commit_store calls must be serialized by _commit_lock to
    protect ZMQ socket access."""
    gather_gate = threading.Event()
    gather_gate.set()
    concurrent_commits = threading.Event()
    active_commits = [0]

    def _commit(_chunks: list[torch.Tensor]) -> bool:
        active_commits[0] += 1
        if active_commits[0] > 1:
            concurrent_commits.set()
        # Briefly yield so a second commit could slip in if unlocked.
        time.sleep(0.02)
        active_commits[0] -= 1
        return True

    ctx = _new_context(
        monkeypatch, gather_gate=gather_gate, commit_impl=_commit, max_inflight=4
    )

    futures = [
        ctx.submit_store(
            f"r{i}",
            object(),
            1,
            {"k": torch.zeros(1)},
            [[0]],
            _FakeEvent(gather_gate),
            1,
        )
        for i in range(4)
    ]

    for f in futures:
        assert f.result(timeout=2) is True

    # With _commit_lock, no two commits should have been active simultaneously.
    assert not concurrent_commits.is_set(), (
        "commit_store called concurrently — _commit_lock not working"
    )
    ctx.close()


def test_prepare_store_runs_on_background_thread_not_forward_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """submit_store must return immediately even when prepare_store is slow.

    The forward thread must not be blocked by the prepare_store RPC round-trip.
    Both prepare_store and buffer allocation now run entirely inside the
    background ``_prepare_gather_and_commit`` closure.
    """
    gather_gate = threading.Event()
    gather_gate.set()
    prepare_started = threading.Event()
    prepare_gate = threading.Event()

    def _slow_prepare() -> None:
        prepare_started.set()
        # Block until the test explicitly releases the gate.
        prepare_gate.wait(timeout=2)

    monkeypatch.setattr(async_engine_driven, "torch_dev", _FakeTorchDev(gather_gate))
    _install_fake_gather(monkeypatch)

    ctx = AsyncEngineDrivenTransferContext(commit_workers=1)
    ctx._engine_driven_context = _FakeStoreContext(  # type: ignore[assignment]
        commit_impl=lambda _c: True,
        prepare_impl=_slow_prepare,
    )

    submit_returned = threading.Event()

    def _submit() -> None:
        ctx.submit_store(
            "r1", object(), 1, {"k": torch.zeros(1)}, [[0]], _FakeEvent(gather_gate), 1
        )
        submit_returned.set()

    t = threading.Thread(target=_submit, daemon=True)
    t.start()

    # submit_store must return before prepare_store finishes.
    assert submit_returned.wait(timeout=1), (
        "submit_store blocked the forward thread — prepare_store is not async"
    )

    # Confirm prepare_store was actually reached by the background thread.
    assert prepare_started.wait(timeout=1), (
        "background thread never reached prepare_store"
    )

    # Now release prepare_store and let the background work complete.
    prepare_gate.set()
    t.join(timeout=1)
    ctx.close()


def test_multigroup_shm_store_runs_prepare_and_gather_in_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid-KV SHM stores must not block the model forward thread."""
    gather_gate = threading.Event()
    prepare_started = threading.Event()
    prepare_gate = threading.Event()
    gather_calls: list[tuple[list[str], list[int], int, list[int] | None]] = []
    committed = threading.Event()

    def _prepare() -> None:
        prepare_started.set()
        prepare_gate.wait(timeout=2)

    def _gather(
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[int],
        blocks_in_chunk: int,
        **kwargs: object,
    ) -> list[torch.Tensor]:
        out = kwargs["out"]
        assert isinstance(out, list)
        chunk_indices = kwargs.get("chunk_indices")
        assert chunk_indices is None or isinstance(chunk_indices, list)
        gather_calls.append(
            (list(kv_caches), block_ids, blocks_in_chunk, chunk_indices)
        )
        for tensor in out:
            tensor.fill_(float(len(gather_calls)))
        return out

    monkeypatch.setattr(async_engine_driven, "torch_dev", _FakeTorchDev(gather_gate))
    monkeypatch.setattr(async_engine_driven, "gather_paged_kv_to_cpu", _gather)
    monkeypatch.setattr(worker_transfer, "gather_paged_kv_to_cpu", _gather)
    ctx = AsyncEngineDrivenTransferContext(commit_workers=1)

    def _commit(chunks: list[list[torch.Tensor]]) -> bool:
        committed.set()
        return chunks == []

    ctx._engine_driven_context = _FakeGroupedStoreContext(  # type: ignore[assignment]
        commit_impl=_commit,
        prepare_result=(
            [[torch.zeros(1)], [torch.zeros(1)]],
            [[0], [0]],
        ),
        prepare_impl=_prepare,
    )
    _install_two_group_state(ctx)

    future = ctx.submit_store(
        "hybrid-request",
        object(),
        1,
        {"layer_0": torch.zeros(1), "layer_1": torch.zeros(1)},
        [[10], [20, 21]],
        _FakeEvent(gather_gate),
        1,
    )

    assert prepare_started.wait(timeout=1)
    assert not future.query()
    prepare_gate.set()
    assert not committed.wait(timeout=0.05)
    gather_gate.set()
    assert future.result(timeout=1) is True
    assert gather_calls == [
        (["layer_0"], [10], 1, [0]),
        (["layer_1"], [20, 21], 2, [0]),
    ]
    assert committed.is_set()
    ctx.close()


def test_multigroup_pickle_store_commits_group_major_payload_in_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid-KV pickle stores preserve group boundaries off the forward thread."""
    gather_gate = threading.Event()
    prepare_started = threading.Event()
    prepare_gate = threading.Event()
    committed_payload: list[list[list[torch.Tensor]]] = []

    def _prepare() -> None:
        prepare_started.set()
        prepare_gate.wait(timeout=2)

    def _gather(
        kv_caches: dict[str, torch.Tensor],
        _block_ids: list[int],
        _blocks_in_chunk: int,
        **_kwargs: object,
    ) -> list[torch.Tensor]:
        value = 1.0 if "layer_0" in kv_caches else 2.0
        return [torch.full((1,), value)]

    monkeypatch.setattr(async_engine_driven, "torch_dev", _FakeTorchDev(gather_gate))
    monkeypatch.setattr(async_engine_driven, "gather_paged_kv_to_cpu", _gather)
    monkeypatch.setattr(worker_transfer, "gather_paged_kv_to_cpu", _gather)
    ctx = AsyncEngineDrivenTransferContext(commit_workers=1)

    def _commit(chunks: list[list[torch.Tensor]]) -> bool:
        committed_payload.append(chunks)
        return True

    ctx._engine_driven_context = _FakeGroupedStoreContext(  # type: ignore[assignment]
        commit_impl=_commit,
        prepare_result=None,
        prepare_impl=_prepare,
    )
    _install_two_group_state(ctx)

    future = ctx.submit_store(
        "hybrid-pickle-request",
        object(),
        1,
        {"layer_0": torch.zeros(1), "layer_1": torch.zeros(1)},
        [[10], [20, 21]],
        _FakeEvent(gather_gate),
        1,
    )

    assert prepare_started.wait(timeout=1)
    assert not future.query()
    prepare_gate.set()
    assert not future.query()
    gather_gate.set()
    assert future.result(timeout=1) is True
    assert len(committed_payload) == 1
    assert [group[0].item() for group in committed_payload[0]] == [1.0, 2.0]
    ctx.close()


def test_multigroup_store_rejects_mismatched_block_id_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every registered hybrid-KV group requires one block-ID sequence."""
    gather_gate = threading.Event()
    gather_gate.set()
    monkeypatch.setattr(async_engine_driven, "torch_dev", _FakeTorchDev(gather_gate))
    ctx = AsyncEngineDrivenTransferContext(commit_workers=1)
    ctx._engine_driven_context = _FakeGroupedStoreContext(  # type: ignore[assignment]
        commit_impl=lambda _chunks: True,
        prepare_result=None,
    )
    _install_two_group_state(ctx)

    with pytest.raises(ValueError, match="1 block-id lists.*2 registered groups"):
        ctx.submit_store(
            "hybrid-request",
            object(),
            1,
            {"layer_0": torch.zeros(1), "layer_1": torch.zeros(1)},
            [[10]],
            _FakeEvent(gather_gate),
            1,
        )
    ctx.close()


def test_multigroup_shm_store_skips_gather_and_commit_on_full_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server-reported full hit requires no worker data transfer."""
    gather_gate = threading.Event()
    gather_gate.set()
    gather = MagicMock()
    commit = MagicMock(return_value=True)
    monkeypatch.setattr(async_engine_driven, "torch_dev", _FakeTorchDev(gather_gate))
    monkeypatch.setattr(async_engine_driven, "gather_paged_kv_to_cpu", gather)
    monkeypatch.setattr(worker_transfer, "gather_paged_kv_to_cpu", gather)
    ctx = AsyncEngineDrivenTransferContext(commit_workers=1)
    ctx._engine_driven_context = _FakeGroupedStoreContext(  # type: ignore[assignment]
        commit_impl=commit,
        prepare_result=([[], []], [[], []]),
    )
    _install_two_group_state(ctx)

    future = ctx.submit_store(
        "hybrid-hit",
        object(),
        1,
        {"layer_0": torch.zeros(1), "layer_1": torch.zeros(1)},
        [[10], [20, 21]],
        _FakeEvent(gather_gate),
        1,
    )

    assert future.result(timeout=1) is True
    gather.assert_not_called()
    commit.assert_not_called()
    ctx.close()


def test_multigroup_flush_waits_until_gather_event_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preemption cannot reuse hybrid-KV blocks before gather is protected."""
    gather_gate = threading.Event()
    gather_started = threading.Event()
    flush_returned = threading.Event()

    class _BlockingFakeTorchDev(_FakeTorchDev):
        def stream(self, stream: object) -> object:
            gather_started.set()
            gather_gate.wait(timeout=2)
            return super().stream(stream)

    monkeypatch.setattr(
        async_engine_driven, "torch_dev", _BlockingFakeTorchDev(gather_gate)
    )
    _install_fake_gather(monkeypatch)
    ctx = AsyncEngineDrivenTransferContext(commit_workers=1)
    ctx._engine_driven_context = _FakeGroupedStoreContext(  # type: ignore[assignment]
        commit_impl=lambda chunks: chunks == [],
        prepare_result=(
            [[torch.zeros(1)], [torch.zeros(1)]],
            [[0], [0]],
        ),
    )
    _install_two_group_state(ctx)

    future = ctx.submit_store(
        "hybrid-preemption",
        object(),
        1,
        {"layer_0": torch.zeros(1), "layer_1": torch.zeros(1)},
        [[10], [20, 21]],
        _FakeEvent(gather_gate),
        1,
    )
    assert gather_started.wait(timeout=1)

    def _flush() -> None:
        ctx.flush_inflight_stores()
        flush_returned.set()

    thread = threading.Thread(target=_flush, daemon=True)
    thread.start()
    assert not flush_returned.wait(timeout=0.05)

    gather_gate.set()
    thread.join(timeout=2)
    assert flush_returned.is_set()
    assert future.result(timeout=1) is True
    ctx.close()
