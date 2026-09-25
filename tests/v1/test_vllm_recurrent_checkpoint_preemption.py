# SPDX-License-Identifier: Apache-2.0
"""Preemption flush tests for the recurrent checkpoint connector.

vLLM calls ``handle_preemptions`` before the worker zeroes or re-allocates
a preempted request's pages. The recurrent checkpoint connector submits
collective copy commands whose in-flight copies read those pages, so a
preemption must drain them first; a copy that cannot drain is fatal rather
than silently corrupted.
"""

# Standard
from concurrent.futures import Future
from threading import Timer
from types import SimpleNamespace
from typing import cast
import sys
import types

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.v1.core.sched.output import SchedulerOutput
import pytest

# First Party
from lmcache.v1.multiprocess.checkpoint_transfer import UnsafeCheckpointCopyError


def _import_connector():
    """Import the recurrent checkpoint connector, tolerating CPU-only runners.

    ``vllm.v1.worker.gpu.boundary_checkpoint`` evaluates triton constants at
    import time and fails when vllm disabled triton for lack of an active
    driver (CPU-only runners). The connector uses the module's single name
    only as a type annotation, so a placeholder class keeps the
    scheduler-side logic testable; on GPU runners the real module loads.
    """
    try:
        # Third Party
        import vllm.v1.worker.gpu.boundary_checkpoint  # noqa: F401
    except Exception:
        stub = types.ModuleType("vllm.v1.worker.gpu.boundary_checkpoint")
        stub.BoundaryCheckpointState = type("BoundaryCheckpointState", (), {})
        sys.modules.setdefault("vllm.v1.worker.gpu.boundary_checkpoint", stub)
    # First Party
    from lmcache.integration.vllm.recurrent_checkpoint_connector import (  # noqa: E402
        LMCacheRecurrentCheckpointConnector,
    )

    return LMCacheRecurrentCheckpointConnector


LMCacheRecurrentCheckpointConnector = _import_connector()


def _connector(pending: dict[str, Future[bool]] | None = None):
    stub = SimpleNamespace(_pending=pending if pending is not None else {})
    return cast(LMCacheRecurrentCheckpointConnector, stub)


def _meta_connector(tasks: list):
    stub = SimpleNamespace(
        _scheduler=SimpleNamespace(take_tasks=lambda: tasks),
    )
    return cast(LMCacheRecurrentCheckpointConnector, stub)


def test_build_connector_meta_flags_preempted_requests() -> None:
    """Scheduler output with preempted requests must mark the metadata for a
    preemption flush; a step without preemptions must not."""
    empty = LMCacheRecurrentCheckpointConnector.build_connector_meta(
        _meta_connector([]),
        cast(SchedulerOutput, SimpleNamespace(preempted_req_ids=set())),
    )
    assert empty.need_flush is False

    flagged = LMCacheRecurrentCheckpointConnector.build_connector_meta(
        _meta_connector([]),
        cast(SchedulerOutput, SimpleNamespace(preempted_req_ids={"req-1"})),
    )
    assert flagged.need_flush is True


def test_handle_preemptions_drains_pending_before_overwrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flagged preemption waits for every in-flight copy.

    A copy that does not drain by the deadline is fatal: the preempted
    request's pages are about to be overwritten, so failing loudly is the
    ownership-retaining contract.
    """
    monkeypatch.setenv("LMCACHE_CHECKPOINT_PREEMPT_FLUSH_TIMEOUT", "0.2")
    slow = Future()
    done = Future()
    done.set_result(True)
    connector = _connector({"task-done": done, "task-slow": slow})
    with pytest.raises(UnsafeCheckpointCopyError):
        LMCacheRecurrentCheckpointConnector.handle_preemptions(
            connector,
            cast(KVConnectorMetadata, SimpleNamespace(need_flush=True)),
        )
    # The delayed copy resolves afterwards without cancelling anything.
    slow.set_result(True)
    assert done.done()


def test_handle_preemptions_skips_unflagged_metadata() -> None:
    """Without the flush flag the hook returns without touching copies."""
    slow = Future()
    connector = _connector({"task-slow": slow})
    LMCacheRecurrentCheckpointConnector.handle_preemptions(
        connector,
        cast(KVConnectorMetadata, SimpleNamespace(need_flush=False)),
    )
    assert not slow.done()


def test_handle_preemptions_accepts_copy_that_drains_in_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A copy completing before the deadline drains without an error."""
    monkeypatch.setenv("LMCACHE_CHECKPOINT_PREEMPT_FLUSH_TIMEOUT", "5")
    draining = Future()
    Timer(0.05, lambda: draining.set_result(True)).start()
    connector = _connector({"task-draining": draining})
    LMCacheRecurrentCheckpointConnector.handle_preemptions(
        connector,
        cast(KVConnectorMetadata, SimpleNamespace(need_flush=True)),
    )
    assert draining.result(timeout=1) is True
