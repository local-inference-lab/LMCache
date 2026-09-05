# SPDX-License-Identifier: Apache-2.0
"""Tests for deterministic engine-driven recurrent-state restores."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.protocols.engine import RegisterEngineDrivenContextResponse
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
    _collapse_chunks_for_single_destination,
)


def _chunks() -> list[torch.Tensor]:
    return [torch.tensor([0]), torch.tensor([1]), torch.tensor([2])]


def test_complete_full_chunk_alias_keeps_newest_snapshot() -> None:
    chunks = _chunks()

    selected_chunks, selected_ids = _collapse_chunks_for_single_destination(
        chunks,
        [4, 5, 4, 5, 4, 5],
        blocks_in_chunk=2,
        blocks_per_window=2,
    )

    assert selected_chunks == chunks[-1:]
    assert selected_ids == [4, 5]


def test_complete_window_alias_keeps_newest_full_mapping() -> None:
    chunks = _chunks()
    block_ids = [10, 11, 12, 7, 20, 21, 22, 7, 30, 31, 32, 7]

    selected_chunks, selected_ids = _collapse_chunks_for_single_destination(
        chunks,
        block_ids,
        blocks_in_chunk=4,
        blocks_per_window=1,
    )

    assert selected_chunks == chunks[-1:]
    assert selected_ids == [30, 31, 32, 7]


def test_distinct_window_destinations_remain_unchanged() -> None:
    chunks = _chunks()
    block_ids = [10, 11, 12, 7, 20, 21, 22, 8, 30, 31, 32, 7]

    selected_chunks, selected_ids = _collapse_chunks_for_single_destination(
        chunks,
        block_ids,
        blocks_in_chunk=4,
        blocks_per_window=1,
    )

    assert selected_chunks is chunks
    assert selected_ids is block_ids


def test_incomplete_or_invalid_geometry_remains_unchanged() -> None:
    chunks = _chunks()
    cases = (
        ([0, 0], 1, 1),
        ([0, 0, 0], 2, 1),
        ([0, 0, 0], 1, 2),
    )

    for block_ids, blocks_in_chunk, blocks_per_window in cases:
        selected_chunks, selected_ids = _collapse_chunks_for_single_destination(
            chunks,
            block_ids,
            blocks_in_chunk=blocks_in_chunk,
            blocks_per_window=blocks_per_window,
        )
        assert selected_chunks is chunks
        assert selected_ids is block_ids


@pytest.mark.parametrize("logical_skip", [0, 1536])
@pytest.mark.parametrize("transport", ["shm", "pickle"])
def test_recurrent_registration_and_partial_prefix_restore_keep_whole_snapshot(
    logical_skip: int, transport: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial logical prefix skips no physical state and must not race aliases."""
    # First Party
    from lmcache import torch_dev
    from lmcache.v1.multiprocess.transfer_context import worker_transfer
    from lmcache.v1.multiprocess.transfer_context.pickle import (
        EngineDrivenContextPickle,
    )

    monkeypatch.setattr(torch_dev, "synchronize", lambda: None)
    monkeypatch.setattr(
        "lmcache.v1.gpu_connector.kv_format.detectors.vllm.torch_device_type", "cuda"
    )
    caches = {
        "recurrent": torch.empty(4, 1536, 1, 656, dtype=torch.uint8),
        "mla": torch.empty(4, 512, 656, dtype=torch.uint8),
    }
    groups = [
        EngineGroupInfo(0, (0,), tokens_per_block=4608, sw_size_tokens=4608),
        EngineGroupInfo(1, (1,), tokens_per_block=4608),
    ]
    snapshots = [
        torch.full((1, 1536, 656), value, dtype=torch.uint8) for value in (3, 5, 7)
    ]
    attention = [torch.empty(1, 512, 656, dtype=torch.uint8) for _ in range(3)]
    if transport == "pickle":
        storage = MagicMock(spec=EngineDrivenContextPickle)
        storage.prepare_retrieve_multigroup.return_value = [snapshots, attention]
    else:
        storage = MagicMock()
        storage.prepare_retrieve_grouped.return_value = (
            snapshots + attention,
            [0, 0, 0, 1, 1, 1],
        )
    monkeypatch.setattr(
        worker_transfer, "create_engine_driven_context", lambda *a, **k: storage
    )
    scatter = MagicMock()
    monkeypatch.setattr(worker_transfer, "scatter_cpu_to_paged_kv", scatter)
    response = MagicMock()
    response.result.return_value = RegisterEngineDrivenContextResponse()
    send = MagicMock(return_value=response)
    context = EngineDrivenTransferContext()
    context.register(
        instance_id=1,
        kv_caches=caches,
        model_name="recurrent-slot-test",
        world_size=9,
        blocks_in_chunk=1,
        mq_client=MagicMock(),
        mq_timeout=1,
        send_request=send,
        layout_hints={"kv_layout": "NHD"},
        engine_group_infos=groups,
        tokens_per_chunk=4608,
    )
    payload = send.call_args.args[2][0]
    assert payload.group_layouts[0].tokens_per_block == 4608
    assert payload.group_layouts[0].window_tokens == 1536
    assert payload.group_layouts[0].hidden_dim_size == 656
    assert payload.group_layouts[1].window_tokens == 512
    try:
        result = context.submit_retrieve(
            "request",
            SimpleNamespace(start=0, end=13824),
            1,
            caches,
            [[1, 1, 1], [0, 1, 2]],
            MagicMock(),
            1,
            skip_first_n_tokens=logical_skip,
        )
        assert result.result(timeout=1)
        recurrent_call = scatter.call_args_list[0]
        assert recurrent_call.args[1] == [1]
        assert recurrent_call.args[2] == snapshots[-1:]
        assert recurrent_call.kwargs["skip_first_n_tokens"] == 0
    finally:
        context.close()
