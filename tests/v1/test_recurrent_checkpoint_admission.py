# SPDX-License-Identifier: Apache-2.0
"""Scheduler-admitted copies must fit even while completion callbacks lag."""

# Standard
from collections.abc import Callable
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any, cast
import inspect
import threading

# Third Party
import pytest

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm import (  # noqa: E402
    recurrent_checkpoint_connector as connector,
)
from lmcache.v1.multiprocess.checkpoint_transfer import (  # noqa: E402
    CheckpointTransferWorker,
)
from lmcache.v1.multiprocess.protocols.base import RequestType  # noqa: E402


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("completed_copies", [0, 1, 2])
@pytest.mark.no_shared_allocator
def test_scheduler_admitted_copies_fit_while_callbacks_lag(
    monkeypatch: pytest.MonkeyPatch, rank: int, completed_copies: int
) -> None:
    """Scheduler-admitted copies must not be rejected by a rank's local queue."""
    release_initial = threading.Event()
    release_pending = threading.Event()
    release_callbacks = threading.Event()
    callbacks_entered = threading.Event()
    callback_returned = threading.Event()
    callback_lock = threading.Lock()
    delayed_count = 0
    registrations = 0
    original_add_callback = Future.add_done_callback

    def add_callback(
        future: Future[bool], callback: Callable[[Future[bool]], Any]
    ) -> None:
        nonlocal registrations
        registrations += 1
        if registrations > completed_copies:
            original_add_callback(future, callback)
            return

        def delayed_callback(result: Future[bool]) -> None:
            nonlocal delayed_count
            with callback_lock:
                delayed_count += 1
                if delayed_count == completed_copies:
                    callbacks_entered.set()
            try:
                assert release_callbacks.wait(timeout=10)
            finally:
                callback(result)
                callback_returned.set()

        original_add_callback(future, delayed_callback)

    monkeypatch.setattr(Future, "add_done_callback", add_callback)

    def copy_pages(job: Any, _lease: Any) -> None:
        gate = (
            release_initial
            if job.manifest.generation.startswith("done")
            else release_pending
        )
        assert gate.wait(timeout=10)

    class Client:
        def submit_request(self, request: RequestType, _payloads: Any) -> Future[Any]:
            result: Future[Any] = Future()
            if request in (
                RequestType.CHECKPOINT_PREPARE_STORE,
                RequestType.CHECKPOINT_BEGIN_RETRIEVE,
            ):
                result.set_result(SimpleNamespace(status="ready", lease_id="mock"))
            else:
                result.set_result(True)
            return result

    # Exercise the connector's real factories and worker protocol, replacing
    # allocator/DMA dependencies so the regression needs neither GPUs nor SHM.
    limits: list[int] = []
    default_limit = (
        inspect.signature(connector.CheckpointSchedulerBridge)
        .parameters["max_tasks"]
        .default
    )

    def scheduler_factory(*_args: Any, **options: Any) -> SimpleNamespace:
        limit = options.get("max_tasks", default_limit)
        limits.append(limit)
        return SimpleNamespace()

    workers: list[CheckpointTransferWorker] = []
    copy_threads: list[int] = []
    default_threads = (
        inspect.signature(CheckpointTransferWorker).parameters["workers"].default
    )

    def worker_factory(*args: Any, **options: Any) -> CheckpointTransferWorker:
        copy_threads.append(options.get("workers", default_threads))
        worker = CheckpointTransferWorker(*args, **options)
        workers.append(worker)
        return worker

    monkeypatch.setattr(connector, "CheckpointSchedulerBridge", scheduler_factory)
    monkeypatch.setattr(connector, "CheckpointTransferWorker", worker_factory)
    monkeypatch.setattr(
        connector, "ShmPoolMapping", lambda *_: SimpleNamespace(close=lambda: None)
    )
    monkeypatch.setattr(connector, "CheckpointPageCopier", lambda *_: copy_pages)
    monkeypatch.setattr(
        connector,
        "build_parallel_strategy_from_vllm_config",
        lambda *_: SimpleNamespace(vllm_worker_id=rank),
    )
    monkeypatch.setattr(
        connector.torch_dev, "Event", lambda: SimpleNamespace(record=lambda: None)
    )
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            get_from_extra_config=lambda _key, default: default
        ),
        compute_hash=lambda: "runtime",
        model_config=SimpleNamespace(compute_hash=lambda: "model"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            decode_context_parallel_size=2,
            cp_kv_cache_interleave_size=1,
            world_size=2,
        ),
    )
    # This test owns the binding context; no connector constructor or RPC server
    # is needed to exercise the public bind/submit/completion methods.
    context = SimpleNamespace(
        _vllm_config=config,
        _client=Client(),
        _mq_timeout=60,
        _role=connector.KVConnectorRole.WORKER,
        _mapping=None,
        _capabilities=SimpleNamespace(shm_name="unused", pool_size=1),
        _pending={},
        _rejected=set(),
        _layout_sent=False,
    )
    bound = cast(connector.LMCacheRecurrentCheckpointConnector, context)
    connector.LMCacheRecurrentCheckpointConnector.bind_boundary_checkpoint_cache(
        bound, cast(Any, object())
    )
    connector.LMCacheRecurrentCheckpointConnector.bind_boundary_checkpoint_state(
        bound,
        cast(
            Any,
            SimpleNamespace(
                get_external_checkpoint_layout=lambda: {"schema_version": 1},
                get_external_checkpoint_page_pool=lambda: None,
                initialize_external_checkpoint_layout=lambda _: None,
            ),
        ),
    )
    assert limits == [32]
    assert copy_threads == [2]
    worker = workers[0]

    def dispatch(names: list[str]) -> None:
        tasks = [
            connector.CheckpointEngineTask(
                name,
                cast(Any, SimpleNamespace(world_size=2, generation=name)),
                "RETRIEVE" if i % 2 else "STORE",
                (),
            )
            for i, name in enumerate(names)
        ]
        context._connector_metadata = connector.RecurrentCheckpointMetadata(tasks)
        connector.LMCacheRecurrentCheckpointConnector.start_load_kv(bound, None)

    def collect() -> dict[str, dict[int, bool]]:
        return (
            connector.LMCacheRecurrentCheckpointConnector.build_connector_worker_meta(
                bound
            ).results
        )

    try:
        initial = [f"done-{i}" for i in range(completed_copies)]
        if initial:
            dispatch(initial)
            release_initial.set()
            assert callbacks_entered.wait(timeout=5)
            assert collect() == {name: {rank: True} for name in initial}

        names = [f"pending-{i}" for i in range(limits[0])]
        dispatch(names)
        assert not callback_returned.is_set(), "Completion callbacks stopped lagging"
        assert collect() == {}, "Scheduler-admitted copies were reported as misses"
    finally:
        release_initial.set()
        release_pending.set()
        release_callbacks.set()
        worker.close()

    assert collect() == {name: {rank: True} for name in names}
