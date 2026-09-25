# SPDX-License-Identifier: Apache-2.0
"""Reaper deadline tests for the checkpoint scheduler bridge.

``take_tasks`` runs on every scheduler step, including connector-only
steps, so it is the one place a wall-clock deadline holds even when
``poll_prefix`` is never re-entered for a parked request. A sent copy
that never completes must release the request to recompute without
deleting the task or freeing its pages: a late-draining copy still
writes into pinned destinations and finishes through the normal
all-rank completion path.
"""

# Standard
import sys
import time
import types

# Third Party
import pytest


def _import_bridge_module():
    """Import the bridge module, tolerating CPU-only runners.

    ``vllm.v1.worker.gpu.boundary_checkpoint`` evaluates triton constants
    at import time and fails when vllm disabled triton for lack of an
    active driver (CPU-only runners). The bridge uses that module's name
    only as a type annotation, so a placeholder class keeps the bridge
    testable; on GPU runners the real module loads.
    """
    try:
        # Third Party
        import vllm.v1.worker.gpu.boundary_checkpoint  # noqa: F401
    except Exception:
        stub = types.ModuleType("vllm.v1.worker.gpu.boundary_checkpoint")
        stub.BoundaryCheckpointState = type("BoundaryCheckpointState", (), {})
        sys.modules.setdefault("vllm.v1.worker.gpu.boundary_checkpoint", stub)
    # noqa: E402 - the semantic-transfer helpers must load after the stub
    # First Party
    from tests.v1.test_vllm_semantic_checkpoint_transfer import (  # noqa: E402
        make_bridge,
        make_manager,
        make_request,
        open_checkpoint_rpc,
    )

    return make_bridge, make_manager, make_request, open_checkpoint_rpc


make_bridge, make_manager, make_request, open_checkpoint_rpc = _import_bridge_module()


def test_take_tasks_releases_a_parked_request_when_its_copy_wedges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sent retrieve copy that never completes must not park its request."""
    monkeypatch.setenv("LMCACHE_CHECKPOINT_TASK_TIMEOUT", "0.2")
    with open_checkpoint_rpc() as (client, module, mapping, _name):
        manager = make_manager()
        bridge = make_bridge(client, manager, lookup_timeout=5.0)
        producer = make_request("producer")
        checkpoint = manager.reserve_external_boundary_checkpoint(
            producer,
            11,
            manager.boundary_checkpoint_page_positions(11),
            draft_prefix_len=11,
            kind="prompt",
            num_ranks=4,
        )
        assert checkpoint is not None
        for rank in range(4):
            manager.acknowledge_external_boundary_checkpoint(
                checkpoint.checkpoint_id, rank
            )
        bridge.store(producer, checkpoint)
        bridge.finish_request(producer.request_id)
        store_tasks = bridge.take_tasks()
        deadline = time.monotonic() + 5
        while not store_tasks and time.monotonic() < deadline:
            store_tasks = bridge.take_tasks()
            time.sleep(0.001)
        assert len(store_tasks) == 1
        store_task = store_tasks[0]

        def copy_pages(job, lease) -> None:
            for group_id, group in enumerate(lease.slots):
                for page_id, (offset, size) in enumerate(group):
                    pattern = bytes([job.rank * 16 + group_id * 4 + page_id]) * size
                    mapping[offset : offset + size] = pattern

        # First Party
        from lmcache.v1.multiprocess.checkpoint_transfer import (  # noqa: E402
            CheckpointTransferJob,
            CheckpointTransferWorker,
        )

        worker = CheckpointTransferWorker(client, copy_pages)
        try:
            for rank in range(4):
                future = worker.submit(
                    CheckpointTransferJob(
                        store_task.manifest,
                        rank,
                        "STORE",
                        store_task.block_ids,
                    )
                )
                assert future is not None and future.result(timeout=10)
                bridge.complete({store_task.task_id: {rank: True}})
                manager.reset_prefix_cache()
            assert not bridge.has_pending

            consumer = make_request("consumer")
            retrieve = []
            deadline = time.monotonic() + 5
            while not retrieve and time.monotonic() < deadline:
                bridge.poll_prefix(consumer)
                retrieve = [
                    task for task in bridge.take_tasks() if task.direction == "RETRIEVE"
                ]
                time.sleep(0.001)
            assert len(retrieve) == 1
            # The retrieve copy wedges: it is never executed. Only take_tasks
            # runs from here on, so the deadline must hold without a
            # poll_prefix re-entry.
            time.sleep(0.4)
            bridge.take_tasks()
            assert bridge.poll_prefix(consumer), (
                "the parked request was not released to recompute after its "
                "retrieve copy exceeded the task deadline"
            )
        finally:
            worker.close()


def test_reaper_retains_the_task_for_a_late_draining_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A copy that drains after the deadline still finishes through the normal path.

    The reaper releases the parked request but must never delete the task or
    free its pages: ``complete()`` raises ValueError on an unknown task id,
    so a late-draining copy completing through the all-rank path after the
    reaper fired is the public proof of retention.
    """
    monkeypatch.setenv("LMCACHE_CHECKPOINT_TASK_TIMEOUT", "0.2")
    with open_checkpoint_rpc() as (client, module, mapping, _name):
        manager = make_manager()
        bridge = make_bridge(client, manager, lookup_timeout=5.0)
        producer = make_request("producer")
        checkpoint = manager.reserve_external_boundary_checkpoint(
            producer,
            11,
            manager.boundary_checkpoint_page_positions(11),
            draft_prefix_len=11,
            kind="prompt",
            num_ranks=4,
        )
        assert checkpoint is not None
        for rank in range(4):
            manager.acknowledge_external_boundary_checkpoint(
                checkpoint.checkpoint_id, rank
            )
        bridge.store(producer, checkpoint)
        bridge.finish_request(producer.request_id)
        store_tasks = bridge.take_tasks()
        deadline = time.monotonic() + 5
        while not store_tasks and time.monotonic() < deadline:
            store_tasks = bridge.take_tasks()
            time.sleep(0.001)
        assert len(store_tasks) == 1
        store_task = store_tasks[0]

        def copy_pages(job, lease) -> None:
            for group_id, group in enumerate(lease.slots):
                for page_id, (offset, size) in enumerate(group):
                    pattern = bytes([job.rank * 16 + group_id * 4 + page_id]) * size
                    mapping[offset : offset + size] = pattern

        # First Party
        from lmcache.v1.multiprocess.checkpoint_transfer import (  # noqa: E402
            CheckpointTransferJob,
            CheckpointTransferWorker,
        )

        worker = CheckpointTransferWorker(client, copy_pages)
        try:
            for rank in range(4):
                future = worker.submit(
                    CheckpointTransferJob(
                        store_task.manifest,
                        rank,
                        "STORE",
                        store_task.block_ids,
                    )
                )
                assert future is not None and future.result(timeout=10)
                bridge.complete({store_task.task_id: {rank: True}})
                manager.reset_prefix_cache()
            assert not bridge.has_pending

            consumer = make_request("consumer")
            retrieve = []
            deadline = time.monotonic() + 5
            while not retrieve and time.monotonic() < deadline:
                bridge.poll_prefix(consumer)
                retrieve = [
                    task for task in bridge.take_tasks() if task.direction == "RETRIEVE"
                ]
                time.sleep(0.001)
            assert len(retrieve) == 1
            retrieve_task = retrieve[0]
            # The copy wedges past the task deadline; only take_tasks runs.
            time.sleep(0.4)
            bridge.take_tasks()
            assert bridge.poll_prefix(consumer), (
                "the parked request was not released to recompute"
            )
            # The reaper must not delete the task: a second take_tasks
            # delivers nothing new, and the late drain still completes.
            assert bridge.take_tasks() == []
            for rank in range(4):
                future = worker.submit(
                    CheckpointTransferJob(
                        retrieve_task.manifest,
                        rank,
                        "RETRIEVE",
                        retrieve_task.block_ids,
                    )
                )
                assert future is not None and future.result(timeout=10)
                bridge.complete({retrieve_task.task_id: {rank: True}})
            assert not bridge.has_pending
        finally:
            worker.close()


def test_reaper_unregisters_a_never_draining_store_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wedged store must not leave its staged registration standing forever.

    The store's CHECKPOINT_BEGIN stages the manifest before any page is
    materialized. A store copy that never drains leaves that staged
    generation in the server directory: it occupies the pending admission
    budget and silently suppresses every identical re-store (begin returns
    False for an already-pending generation, and the bridge aborts such a
    task before delivering it). On the task deadline the reaper submits
    CHECKPOINT_ABORT so the registration is unregistered and a re-store is
    admitted again. The bridge task itself is still never deleted: a
    late-draining copy keeps finishing through the normal path.
    """
    monkeypatch.setenv("LMCACHE_CHECKPOINT_TASK_TIMEOUT", "0.2")
    with open_checkpoint_rpc() as (client, module, mapping, _name):
        manager = make_manager()
        bridge = make_bridge(client, manager, lookup_timeout=5.0)
        producer = make_request("producer")
        checkpoint = manager.reserve_external_boundary_checkpoint(
            producer,
            11,
            manager.boundary_checkpoint_page_positions(11),
            draft_prefix_len=11,
            kind="prompt",
            num_ranks=4,
        )
        assert checkpoint is not None
        for rank in range(4):
            manager.acknowledge_external_boundary_checkpoint(
                checkpoint.checkpoint_id, rank
            )
        bridge.store(producer, checkpoint)
        bridge.finish_request(producer.request_id)
        store_tasks = bridge.take_tasks()
        deadline = time.monotonic() + 5
        while not store_tasks and time.monotonic() < deadline:
            store_tasks = bridge.take_tasks()
            time.sleep(0.001)
        assert len(store_tasks) == 1

        # The store copy wedges: the worker never executes it. The task
        # deadline passes with only take_tasks running, so the reaper must
        # unregister the staged registration.
        time.sleep(0.4)
        bridge.take_tasks()

        # An identical re-store must be admitted again: same tokens, same
        # content-derived generation, so without the abort the staged ghost
        # suppresses it (begin returns False and the task is never delivered).
        successor = make_request("producer-successor")
        successor_checkpoint = manager.reserve_external_boundary_checkpoint(
            successor,
            11,
            manager.boundary_checkpoint_page_positions(11),
            draft_prefix_len=11,
            kind="prompt",
            num_ranks=4,
        )
        assert successor_checkpoint is not None
        for rank in range(4):
            manager.acknowledge_external_boundary_checkpoint(
                successor_checkpoint.checkpoint_id, rank
            )
        bridge.store(successor, successor_checkpoint)
        bridge.finish_request(successor.request_id)
        restocked = bridge.take_tasks()
        deadline = time.monotonic() + 5
        while not restocked and time.monotonic() < deadline:
            restocked = bridge.take_tasks()
            time.sleep(0.001)
        assert len(restocked) == 1, (
            "the reaped store's staged registration still suppresses an "
            "identical re-store"
        )
