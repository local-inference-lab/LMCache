# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the checkpoint_on_reuse store policy.

Tests are written against the StorePolicy contract: checkpoint pages are held
back when written and offered when reused; ordinary keys are written through.
"""

# First Party
from lmcache.v1.distributed.api import (
    RECURRENT_CHECKPOINT_MODEL_PREFIX,
    ObjectKey,
    is_recurrent_checkpoint_key,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.storage_controllers.checkpoint_reuse_store_policy import (
    CheckpointReuseStorePolicy,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
    DefaultStorePolicy,
    create_store_policy,
    get_registered_store_policies,
)


def make_key(chunk_id: int, model_name: str = "test_model") -> ObjectKey:
    """Create a test ObjectKey with the given chunk ID and model name."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
    )


def make_checkpoint_key(chunk_id: int) -> ObjectKey:
    """Create a key named like a recurrent checkpoint payload page."""
    return make_key(chunk_id, f"{RECURRENT_CHECKPOINT_MODEL_PREFIX}v3-{'a' * 64}")


def make_descriptor(index: int) -> AdapterDescriptor:
    """Create an AdapterDescriptor for testing."""
    config = MockL2AdapterConfig(max_size_gb=1.0, mock_bandwidth_gb=10.0)
    return AdapterDescriptor(index=index, config=config)


def test_policy_is_registered_by_name():
    assert "checkpoint_on_reuse" in get_registered_store_policies()
    assert isinstance(
        create_store_policy("checkpoint_on_reuse"), CheckpointReuseStorePolicy
    )


def test_checkpoint_keys_are_recognized_by_model_prefix():
    assert is_recurrent_checkpoint_key(make_checkpoint_key(0))
    assert not is_recurrent_checkpoint_key(make_key(0))


def test_written_checkpoint_pages_are_held_back():
    policy = CheckpointReuseStorePolicy()
    ordinary = [make_key(i) for i in range(2)]
    pages = [make_checkpoint_key(i) for i in range(3)]
    adapters = [make_descriptor(0), make_descriptor(1)]

    result = policy.select_store_targets(ordinary + pages, adapters)

    assert result == {0: ordinary, 1: ordinary}


def test_reused_checkpoint_pages_go_to_every_adapter():
    policy = CheckpointReuseStorePolicy()
    ordinary = [make_key(i) for i in range(2)]
    pages = [make_checkpoint_key(i) for i in range(3)]
    adapters = [make_descriptor(0), make_descriptor(1)]

    result = policy.select_reuse_targets(ordinary + pages, adapters)

    assert result == {0: pages, 1: pages}


def test_policy_keeps_keys_in_l1_and_declares_reuse_admission():
    policy = CheckpointReuseStorePolicy()
    assert policy.uses_reuse_admission()
    assert policy.select_l1_deletions([make_checkpoint_key(0)]) == []


def test_write_through_policy_ignores_reuse():
    policy = DefaultStorePolicy()
    assert not policy.uses_reuse_admission()
    assert (
        policy.select_reuse_targets([make_checkpoint_key(0)], [make_descriptor(0)])
        == {}
    )
