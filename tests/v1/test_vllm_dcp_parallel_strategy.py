# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.integration.vllm import lmcache_mp_connector
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPSchedulerAdapter,
    ParallelStrategy,
)
from lmcache.v1.multiprocess.modules.lookup import compute_extra_count


@pytest.mark.parametrize(
    ("tp_size", "dcp_size"),
    [(4, 2), (4, 4), (6, 2), (6, 3), (6, 6), (8, 2), (8, 4), (8, 8)],
)
def test_mla_dcp_maps_replicated_tp_ranks_to_dcp_shards(
    tp_size: int, dcp_size: int
) -> None:
    strategies = [
        ParallelStrategy(
            use_mla=True,
            vllm_world_size=tp_size,
            vllm_worker_id=rank,
            tp_size=tp_size,
            pp_size=1,
            n_servers=1,
            dcp_size=dcp_size,
        )
        for rank in range(tp_size)
    ]

    assert {strategy.kv_world_size for strategy in strategies} == {dcp_size}
    assert [strategy.kv_worker_id for strategy in strategies] == [
        rank % dcp_size for rank in range(tp_size)
    ]
    assert {strategy.kv_tp_size for strategy in strategies} == {tp_size // dcp_size}
    assert {strategy.kv_readers_per_object for strategy in strategies} == {
        tp_size // dcp_size
    }
    assert [strategy.is_kv_writer for strategy in strategies] == [
        rank < dcp_size for rank in range(tp_size)
    ]


def test_build_parallel_strategy_reads_dcp_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lmcache_mp_connector, "mla_only", lambda _: True)
    parallel_config = SimpleNamespace(
        world_size=8,
        rank=5,
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        decode_context_parallel_size=4,
    )
    config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=object(),
    )

    strategy = lmcache_mp_connector.build_parallel_strategy_from_vllm_config(
        config, n_servers=1
    )

    assert strategy.dcp_size == 4
    assert strategy.kv_world_size == 4
    assert strategy.kv_worker_id == 1
    assert strategy.kv_tp_size == 2
    assert not strategy.is_kv_writer


@pytest.mark.parametrize(
    "overrides",
    [
        {"tp_size": 6, "dcp_size": 4},
        {"pp_size": 2, "dcp_size": 2},
        {"n_servers": 2, "dcp_size": 2},
    ],
)
def test_mla_dcp_rejects_unsupported_geometry(overrides: dict[str, int]) -> None:
    kwargs: dict[str, Any] = {
        "use_mla": True,
        "vllm_world_size": 8,
        "vllm_worker_id": 0,
        "tp_size": 8,
        "pp_size": 1,
        "n_servers": 1,
        "dcp_size": 4,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError):
        ParallelStrategy(**kwargs)


def test_dcp_lookup_key_carries_explicit_reader_count() -> None:
    adapter = LMCacheMPSchedulerAdapter.__new__(LMCacheMPSchedulerAdapter)
    adapter.model_name = "glm"
    adapter.parallel_strategy = ParallelStrategy(
        use_mla=True,
        vllm_world_size=8,
        vllm_worker_id=0,
        tp_size=8,
        pp_size=1,
        n_servers=1,
        dcp_size=4,
    )

    key = adapter._create_key([1, 2, 3], 0, 3, "req")

    assert key.world_size == 4
    assert key.readers_per_object == 2
    assert key.no_worker_id_version().readers_per_object == 2
    assert compute_extra_count(2, 4, key.readers_per_object) == 1


def test_explicit_reader_count_preserves_legacy_fallback() -> None:
    assert compute_extra_count(tp_size=8, world_size=1) == 7
    assert compute_extra_count(tp_size=8, world_size=8) == 0
    assert compute_extra_count(tp_size=2, world_size=4, readers_per_object=2) == 1
