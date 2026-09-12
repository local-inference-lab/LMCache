# SPDX-License-Identifier: Apache-2.0
"""Worker adapter integration of the windowed retrieve.

With ``lmcache.mp.restore_window_chunks`` set, a retrieve is submitted through
``submit_windowed_retrieve`` when the transfer context offers it; a failed
windowed retrieve invalidates only the blocks past the written token end; a
request the engine finished while its load was in flight is cancelled.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.vllm_multi_process_adapter import LoadStoreOp
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from tests.v1.test_vllm_mp_adapter import (  # noqa: F401
    FakeHeartbeatThread,
    _make_worker_adapter,
    fake_adapter,
)


def _windowed_adapter(fake_adapter, monkeypatch, window_chunks=2):
    adapter, _send_mock, _future = fake_adapter
    monkeypatch.setattr(adapter, "_ensure_heartbeat_started", lambda: None)
    adapter._restore_window_chunks = window_chunks
    fake_tensor = MagicMock()
    fake_tensor.device.type = "cuda"
    adapter.kv_caches = {"layer.0": fake_tensor}
    transfer_ctx = MagicMock()
    adapter.transfer_ctx = transfer_ctx
    return adapter, transfer_ctx


def test_extra_config_sets_the_window_and_env_overrides_it(monkeypatch, fake_adapter):
    del fake_adapter
    monkeypatch.delenv("LMCACHE_RESTORE_WINDOW_CHUNKS", raising=False)
    adapter = _make_worker_adapter({"lmcache.mp.restore_window_chunks": 3})
    assert adapter._restore_window_chunks == 3
    monkeypatch.setenv("LMCACHE_RESTORE_WINDOW_CHUNKS", "5")
    adapter = _make_worker_adapter({"lmcache.mp.restore_window_chunks": 3})
    assert adapter._restore_window_chunks == 5
    monkeypatch.delenv("LMCACHE_RESTORE_WINDOW_CHUNKS", raising=False)
    adapter = _make_worker_adapter()
    assert adapter._restore_window_chunks == 0


def test_retrieve_uses_the_windowed_path_when_configured(fake_adapter, monkeypatch):
    adapter, transfer_ctx = _windowed_adapter(fake_adapter, monkeypatch)
    fake_future = MagicMock()
    transfer_ctx.submit_windowed_retrieve.return_value = fake_future
    op = LoadStoreOp(token_ids=[1, 2, 3, 4], block_ids=[[0]], start=0, end=4)

    adapter.submit_retrieve_request("req-1", op, event=MagicMock())

    transfer_ctx.submit_retrieve.assert_not_called()
    kwargs = transfer_ctx.submit_windowed_retrieve.call_args.kwargs
    assert kwargs["window_chunks"] == 2
    assert kwargs["tp_size"] == 1 and kwargs["readers_per_object"] == 1
    assert adapter.retrieve_futures["req-1"] == (fake_future, [0])
    assert adapter._retrieve_ops["req-1"] is op


def test_retrieve_keeps_the_single_shot_path_when_disabled(fake_adapter, monkeypatch):
    adapter, transfer_ctx = _windowed_adapter(fake_adapter, monkeypatch, window_chunks=0)
    transfer_ctx.submit_retrieve.return_value = MagicMock()
    op = LoadStoreOp(token_ids=[1, 2, 3, 4], block_ids=[[0]], start=0, end=4)
    adapter.submit_retrieve_request("req-1", op, event=MagicMock())
    transfer_ctx.submit_windowed_retrieve.assert_not_called()
    transfer_ctx.submit_retrieve.assert_called_once()


def test_partial_failure_invalidates_only_unwritten_blocks(fake_adapter, monkeypatch):
    adapter, transfer_ctx = _windowed_adapter(fake_adapter, monkeypatch)
    adapter.engine_group_infos = [
        EngineGroupInfo(engine_group_id=0, layer_indices=(0,), tokens_per_block=16),
        EngineGroupInfo(engine_group_id=1, layer_indices=(1,), tokens_per_block=64),
    ]
    future = MagicMock()
    future.query.return_value = True
    future.result.return_value = False
    transfer_ctx.submit_windowed_retrieve.return_value = future
    # Range [0, 256): group 0 has 16 blocks of 16 tokens, group 1 four of 64.
    op = LoadStoreOp(
        token_ids=list(range(256)),
        block_ids=[list(range(16)), list(range(100, 104))],
        start=0,
        end=256,
    )
    adapter.submit_retrieve_request("req-1", op, event=MagicMock())
    transfer_ctx.pop_retrieve_progress.return_value = 96  # 6 blocks / 1.5 blocks

    _stores, finished = adapter.get_finished(set())

    assert finished == {"req-1"}
    invalid = adapter.get_block_ids_with_load_errors()
    assert invalid == set(range(6, 16)) | {101, 102, 103}
    assert "req-1" not in adapter._retrieve_ops


def test_failure_without_progress_invalidates_the_whole_span(fake_adapter, monkeypatch):
    adapter, transfer_ctx = _windowed_adapter(fake_adapter, monkeypatch)
    future = MagicMock()
    future.query.return_value = True
    future.result.return_value = False
    transfer_ctx.submit_windowed_retrieve.return_value = future
    transfer_ctx.pop_retrieve_progress.return_value = None
    op = LoadStoreOp(token_ids=list(range(32)), block_ids=[[3, 4]], start=0, end=32)
    adapter.submit_retrieve_request("req-1", op, event=MagicMock())

    _stores, finished = adapter.get_finished(set())

    assert finished == {"req-1"}
    assert adapter.get_block_ids_with_load_errors() == {3, 4}


def test_engine_finished_request_cancels_its_inflight_retrieve(fake_adapter, monkeypatch):
    adapter, transfer_ctx = _windowed_adapter(fake_adapter, monkeypatch)
    future = MagicMock()
    future.query.return_value = False
    transfer_ctx.submit_windowed_retrieve.return_value = future
    op = LoadStoreOp(token_ids=[1, 2, 3, 4], block_ids=[[0]], start=0, end=4)
    adapter.submit_retrieve_request("req-1", op, event=MagicMock())

    _stores, finished = adapter.get_finished({"req-1"})

    assert finished == set()
    transfer_ctx.cancel_retrieve.assert_called_once_with("req-1")
    # Still pending: reported once the future resolves.
    assert "req-1" in adapter.retrieve_futures
