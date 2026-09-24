# SPDX-License-Identifier: Apache-2.0
"""A checkpoint lookup refreshes the found pages in L1 and L2 eviction order."""

# Standard
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from mmap import mmap
from pathlib import Path
from types import SimpleNamespace
from typing import cast
import time
import uuid

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.eviction_policy import LRUEvictionPolicy
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
    FSNativeL2AdapterConfig,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.multiprocess.checkpoint_index import CheckpointManifest
from lmcache.v1.multiprocess.checkpoint_storage import (
    CheckpointSlots,
    checkpoint_object_keys,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules.checkpoint import CheckpointModule
from lmcache.v1.multiprocess.posix_shm import shm_open_pool_as_mmap
from tests.v1.multiprocess.test_checkpoint_storage import (
    drain_l2_stores,
    make_manifest,
    open_store,
    poll,
    publish_all,
)

POOL_BYTES = 4 * 1024 * 1024


@contextmanager
def open_module_with_l2_lru(
    path: Path,
) -> Iterator[tuple[CheckpointModule, StorageManager, mmap]]:
    """A checkpoint module over fs_native L2 with an LRU eviction order."""
    name = f"lmcache_l1_pool_checkpoint_refresh_{uuid.uuid4().hex}"
    l2 = FSNativeL2AdapterConfig(str(path / "payloads"), max_capacity_gb=1.0)
    # Far below the watermark: the order is tracked but nothing is evicted.
    l2.eviction_config = EvictionConfig(eviction_policy="LRU")
    storage = StorageManager(
        StorageManagerConfig(
            L1ManagerConfig(L1MemoryManagerConfig(POOL_BYTES, False, shm_name=name)),
            EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=L2AdaptersConfig([l2]),
        )
    )
    with ExitStack() as cleanup:
        cleanup.callback(storage.close)
        mapping = cleanup.enter_context(shm_open_pool_as_mmap(name, POOL_BYTES))
        module = CheckpointModule(
            cast(
                MPCacheServerContext,
                SimpleNamespace(
                    storage_manager=storage,
                    shm_pool_info={"shm_name": name, "pool_size": POOL_BYTES},
                ),
            )
        )
        cleanup.callback(module.close)
        yield module, storage, mapping


def publish_through_module(
    module: CheckpointModule, mapping: mmap, entry: CheckpointManifest
) -> None:
    assert module.begin(entry)
    for rank in range(entry.world_size):
        lease = module.prepare_store(entry, rank)
        assert lease.status == "ready"
        for group in lease.slots:
            for offset, size in group:
                mapping[offset : offset + size] = bytes([rank + 1]) * size
        assert module.finish_store(lease.lease_id, True) == (
            rank == entry.world_size - 1
        )


def all_rank_keys(entry: CheckpointManifest) -> set[ObjectKey]:
    return {
        key
        for rank in range(entry.world_size)
        for group in checkpoint_object_keys(entry, rank)
        for key in group
    }


def l2_eviction_victims(storage: StorageManager, count: int) -> set[ObjectKey]:
    """Return the ``count`` keys the L2 LRU order would evict first."""
    (state,) = storage._l2_eviction_controller._adapter_states
    policy = cast(LRUEvictionPolicy, state.eviction_policy)
    actions = policy.get_eviction_actions((count + 0.5) / len(policy._order))
    return {key for action in actions for key in action.keys}


@pytest.mark.parametrize("found_again", [False, True])
def test_lookup_keeps_found_checkpoint_pages_out_of_l2_eviction(
    tmp_path: Path, found_again: bool
) -> None:
    """A conversation resumed from the engine's cache reads no payload.

    Without a refresh, its oldest pages stay first in L2 eviction order even
    though the lookup just proved the checkpoint is wanted.
    """
    early = make_manifest()
    later = replace(
        make_manifest(), prefix=replace(early.prefix, tail_tokens=(1, 2, 4))
    )
    with open_module_with_l2_lru(tmp_path) as (module, storage, mapping):
        publish_through_module(module, mapping, early)
        drain_l2_stores(storage)
        publish_through_module(module, mapping, later)
        drain_l2_stores(storage)
        if found_again:
            assert module.find((early.prefix,)) == early
        victims = l2_eviction_victims(storage, len(all_rank_keys(later)))
        assert victims == all_rank_keys(later if found_again else early)


def test_refresh_does_not_count_as_l2_residency_for_reuse_admission(
    tmp_path: Path,
) -> None:
    """A refresh changes eviction order only; it never claims L2 residency."""
    entry = make_manifest()
    with open_store(tmp_path, native=True, store_policy="checkpoint_on_reuse") as (
        service,
        index,
        storage,
        mapping,
    ):
        publish_all(service, index, mapping, entry)
        storage.touch_keys(sorted(all_rank_keys(entry), key=repr))
        for rank in range(entry.world_size):
            lease_id = service.begin_retrieve(entry, rank)
            assert lease_id is not None
            lease = poll(service, lease_id)
            assert isinstance(lease, CheckpointSlots)
            service.finish_retrieve(lease.lease_id)
        drain_l2_stores(storage)
        time.sleep(0.2)
        drain_l2_stores(storage)
    with open_store(tmp_path, native=True, store_policy="checkpoint_on_reuse") as (
        service,
        _index,
        _storage,
        _mapping,
    ):
        for rank in range(entry.world_size):
            lease_id = service.begin_retrieve(entry, rank)
            assert lease_id is not None
            lease = poll(service, lease_id)
            assert isinstance(lease, CheckpointSlots)
            service.finish_retrieve(lease.lease_id)
