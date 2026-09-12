# SPDX-License-Identifier: Apache-2.0
"""``PrefetchMode.LOOKUP`` with a pin limit: load and read-lock a bounded head,
report the rest of the prefix from the L2 index.

With ``pin_limit_keys`` the storage manager loads at most that many leading
keys (L1 prefix hits included) and answers the remaining keys as a
presence-only continuation. The combined result still counts the whole prefix,
while L1 holds only the pinned head; the unpinned keys carry no lock.
"""

# Standard
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchMode,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    L2AdaptersConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.storage_manager import StorageManager

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)


def _wait_for(predicate, timeout: float = 10.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _found(sm: StorageManager, handle, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = sm.query_prefetch_status(handle)
        if found is not None:
            return found
        time.sleep(0.02)
    raise TimeoutError("prefetch status never became available")


def _key(chunk: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk),
        model_name="test_model",
        kv_rank=0,
    )


@pytest.fixture
def layout() -> MemoryLayoutDesc:
    return MemoryLayoutDesc(
        shapes=[torch.Size([100, 2, 512])],
        dtypes=[torch.bfloat16],
    )


@pytest.fixture
def storage_manager():
    config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=128 * 1024 * 1024,
                use_lazy=True,
                init_size_in_bytes=64 * 1024 * 1024,
                align_bytes=0x1000,
            ),
            write_ttl_seconds=600,
            read_ttl_seconds=300,
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(
            adapters=[MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)],
        ),
    )
    sm = StorageManager(config)
    yield sm
    sm.close()


def _write_and_persist(sm: StorageManager, keys: list[ObjectKey], layout) -> None:
    reserved = sm.reserve_write(keys, layout, mode="new")
    assert len(reserved) == len(keys)
    sm.finish_write(list(reserved.keys()))
    adapter = sm._l2_adapters[0]
    assert _wait_for(lambda: all(adapter.debug_has_key(k) for k in keys)), (
        "keys were not persisted to the mock L2 adapter"
    )
    time.sleep(0.05)


def _l1_state(sm: StorageManager, key: ObjectKey):
    return sm._l1_manager.get_object_state(key)


def test_pin_limit_loads_a_head_and_reports_the_tail_from_the_index(
    storage_manager, layout
):
    sm = storage_manager
    keys = [_key(i) for i in range(8)]
    _write_and_persist(sm, keys, layout)
    sm.clear()

    handle = sm.submit_prefetch_task(
        keys, layout, mode=PrefetchMode.LOOKUP, pin_limit_keys=3
    )
    assert handle.pinned_key_count == 3
    assert handle.tail_prefetch_request_id != -1
    assert handle.l2_orig_indices == (0, 1, 2)
    assert handle.tail_orig_indices == (3, 4, 5, 6, 7)
    assert sm.wait_prefetch_status(handle, timeout=10.0)
    found = _found(sm, handle)
    assert found.count_leading_ones() == 8

    # The head is loaded and read-locked; the tail is not in L1.
    for key in keys[:3]:
        state = _l1_state(sm, key)
        assert state is not None and state.read_lock.is_locked()
    for key in keys[3:]:
        assert _l1_state(sm, key) is None
    sm.finish_read_prefetched(keys[:3])


def test_pin_limit_counts_l1_prefix_hits(storage_manager, layout):
    sm = storage_manager
    keys = [_key(i) for i in range(6)]
    _write_and_persist(sm, keys, layout)
    # Keep keys 0-1 resident, evict the rest.
    sm.delete_l1_keys(keys[2:])
    assert all(_l1_state(sm, k) is not None for k in keys[:2])

    handle = sm.submit_prefetch_task(
        keys, layout, mode=PrefetchMode.LOOKUP, pin_limit_keys=3
    )
    assert handle.l1_found_indices == (0, 1)
    # Two L1 hits leave room for one loaded key under the pin limit.
    assert handle.l2_orig_indices == (2,)
    assert handle.tail_orig_indices == (3, 4, 5)
    assert handle.pinned_key_count == 3
    assert sm.wait_prefetch_status(handle, timeout=10.0)
    assert _found(sm, handle).count_leading_ones() == 6
    assert _l1_state(sm, keys[2]) is not None
    assert _l1_state(sm, keys[3]) is None
    sm.finish_read_prefetched(keys[:3])


def test_pin_limit_prefix_stops_at_a_missing_tail_key(storage_manager, layout):
    sm = storage_manager
    keys = [_key(i) for i in range(8)]
    _write_and_persist(sm, keys[:5] + keys[6:], layout)  # key 5 never stored
    sm.clear()

    handle = sm.submit_prefetch_task(
        keys, layout, mode=PrefetchMode.LOOKUP, pin_limit_keys=2
    )
    assert sm.wait_prefetch_status(handle, timeout=10.0)
    assert _found(sm, handle).count_leading_ones() == 5
    sm.finish_read_prefetched(keys[:2])


def test_early_hit_query_includes_the_presence_tail(storage_manager, layout):
    sm = storage_manager
    keys = [_key(i) for i in range(6)]
    _write_and_persist(sm, keys, layout)
    sm.clear()

    handle = sm.submit_prefetch_task(
        keys, layout, mode=PrefetchMode.LOOKUP, pin_limit_keys=2
    )
    assert sm.wait_prefetch_status(handle, timeout=10.0)
    assert _wait_for(lambda: sm.query_prefetch_lookup_hits(handle) == 6)
    assert _found(sm, handle).count_leading_ones() == 6
    sm.finish_read_prefetched(keys[:2])


def test_zero_pin_limit_keeps_the_full_load(storage_manager, layout):
    sm = storage_manager
    keys = [_key(i) for i in range(4)]
    _write_and_persist(sm, keys, layout)
    sm.clear()

    handle = sm.submit_prefetch_task(keys, layout, mode=PrefetchMode.LOOKUP)
    assert handle.pinned_key_count == -1
    assert handle.tail_prefetch_request_id == -1
    assert sm.wait_prefetch_status(handle, timeout=10.0)
    assert _found(sm, handle).count_leading_ones() == 4
    assert all(_l1_state(sm, k) is not None for k in keys)
    sm.finish_read_prefetched(keys)
