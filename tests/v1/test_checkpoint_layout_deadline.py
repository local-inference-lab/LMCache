# SPDX-License-Identifier: Apache-2.0
"""Byte-layout negotiation deadline and pre-layout store skip observability.

Worker byte layouts are negotiated at startup. When they never arrive the
bridge parked every restore-capable request forever and spun connector-only
steps for a layout that would never come; stores arriving in that window
were skipped silently.
"""

# Standard
from unittest.mock import MagicMock, patch
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
    # noqa: E402 - imports below must load after the stub
    # First Party
    from lmcache.integration.vllm import checkpoint_scheduler  # noqa: E402
    from tests.v1.test_vllm_semantic_checkpoint_transfer import (  # noqa: E402
        make_manager,
        make_request,
        open_checkpoint_rpc,
    )

    return checkpoint_scheduler, make_manager, make_request, open_checkpoint_rpc


(
    checkpoint_scheduler,
    make_manager,
    make_request,
    open_checkpoint_rpc,
) = _import_bridge_module()


def _bridge(client, manager):
    return checkpoint_scheduler.CheckpointSchedulerBridge(
        manager,
        client,
        {
            "target_revision": "a" * 40,
            "draft_revision": "",
            "source_revision": "b" * 40,
            "parallel": {"tp": 4, "dcp": 1},
        },
        4,
        lookup_timeout=5.0,
    )


def test_layout_negotiation_deadline_releases_parked_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requests park only while negotiation is still possible.

    A layout that never arrives must not park restore-capable requests
    forever: past the deadline the bridge serves without external
    checkpoints, warns once, and stops reporting connector-only steps.
    """
    monkeypatch.setenv("LMCACHE_CHECKPOINT_LAYOUT_TIMEOUT", "0.2")
    with open_checkpoint_rpc() as (client, _module, _mapping, _name):
        manager = make_manager()
        bridge = _bridge(client, manager)
        consumer = make_request("consumer")
        assert not bridge.poll_prefix(consumer)
        assert bridge.has_pending
        time.sleep(0.3)
        with patch.object(checkpoint_scheduler.logger, "warning") as warning:
            assert bridge.poll_prefix(consumer), (
                "the request stayed parked after the layout deadline"
            )
            assert not bridge.has_pending
            deadline_warnings = [
                call for call in warning.call_args_list if "not negotiated" in str(call)
            ]
            assert len(deadline_warnings) == 1, (
                "the layout deadline must warn exactly once"
            )


def test_store_before_layout_negotiation_is_logged_until_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pre-layout store skip is visible, then yields to the deadline warning.

    While negotiation is still pending, every silently skipped store logs
    one line. Once the layout deadline has failed the negotiation, the
    one-time deadline warning already covers the state and per-store
    logging would only spam.
    """
    monkeypatch.setenv("LMCACHE_CHECKPOINT_LAYOUT_TIMEOUT", "0.2")
    with open_checkpoint_rpc() as (client, _module, _mapping, _name):
        manager = make_manager()
        bridge = _bridge(client, manager)
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
        with patch.object(checkpoint_scheduler.logger, "info") as info:
            bridge.store(producer, checkpoint)
            skips = [
                call
                for call in info.call_args_list
                if "not negotiated yet" in str(call)
            ]
            assert len(skips) == 1, (
                "a store skipped before byte layout negotiation must log one line"
            )
        time.sleep(0.3)
        bridge.poll_prefix(producer)
        with patch.object(checkpoint_scheduler.logger, "info") as info:
            bridge.store(producer, MagicMock(name="checkpoint"))
            assert not [
                call
                for call in info.call_args_list
                if "not negotiated yet" in str(call)
            ], "per-store skip logging must stop once the deadline has failed"
