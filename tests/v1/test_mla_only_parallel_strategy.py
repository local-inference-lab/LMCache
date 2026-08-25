# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace

# First Party
from lmcache.integration.vllm import lmcache_mp_connector
from lmcache.integration.vllm import lmcache_mp_connector_0180
from lmcache.integration.vllm import lmcache_mp_connector_0201
from lmcache.integration.vllm.utils import mla_enabled, mla_only


def test_hybrid_mla_keeps_rank_sharded_recurrent_state() -> None:
    model_config = SimpleNamespace(use_mla=True, is_hybrid=True)
    parallel_config = SimpleNamespace(
        world_size=8,
        rank=5,
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        decode_context_parallel_size=8,
    )
    config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=model_config,
    )

    assert mla_enabled(model_config)
    assert not mla_only(model_config)
    strategy = lmcache_mp_connector.build_parallel_strategy_from_vllm_config(
        config, n_servers=1
    )

    # Kimi-K3's recurrent/KDA state is rank-sharded even though its full
    # attention layers use MLA. It must not take the pure-MLA TP sharing path.
    assert not strategy.use_mla
    assert strategy.kv_world_size == 8
    assert strategy.kv_worker_id == 5
    assert strategy.kv_tp_size == 8
    assert strategy.kv_readers_per_object == 1
    assert strategy.is_kv_writer

    assert lmcache_mp_connector_0180.extract_world_size_and_kv_rank(8, 5, config) == (
        8,
        5,
    )
    legacy_strategy = lmcache_mp_connector_0201.build_parallel_strategy(config)
    assert not legacy_strategy.use_mla
    assert legacy_strategy.kv_world_size == 8
    assert legacy_strategy.kv_worker_id == 5
    assert legacy_strategy.kv_tp_size == 8
