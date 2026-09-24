# SPDX-License-Identifier: Apache-2.0
"""A checkpoint store abandoned by its worker must never be deduplicated."""

# Standard
import hashlib
import time
import uuid

# First Party
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.multiprocess.checkpoint_index import CheckpointIndex
from lmcache.v1.multiprocess.checkpoint_storage import (
    CheckpointPayloadStore,
    CheckpointSlots,
)
from lmcache.v1.multiprocess.posix_shm import shm_open_pool_as_mmap
from tests.v1.multiprocess.test_checkpoint_storage import make_content_manifest


def test_abandoned_store_pages_are_rewritten_not_reused() -> None:
    """A worker died between PREPARE_STORE and FINISH_STORE.

    Once its write locks expire, a later generation that shares those
    content-keyed pages must copy them again. Treating the unfinished pages
    as readable would publish a manifest whose restore returns whatever
    bytes the dead worker left behind.
    """
    name = f"lmcache_l1_pool_checkpoint_abandoned_{uuid.uuid4().hex}"
    size = 4 * 1024 * 1024
    storage = StorageManager(
        StorageManagerConfig(
            L1ManagerConfig(
                L1MemoryManagerConfig(size, False, shm_name=name),
                write_ttl_seconds=1,
            ),
            EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=L2AdaptersConfig([]),
        )
    )
    index = CheckpointIndex()
    shared = tuple(
        hashlib.sha256(f"system-prompt-{page}".encode()).hexdigest()
        for page in range(3)
    )
    abandoned = make_content_manifest(
        "abandoned", prefix_token=10, attention_keys=shared
    )
    later = make_content_manifest("later", prefix_token=11, attention_keys=shared)
    try:
        with shm_open_pool_as_mmap(name, size):
            service = CheckpointPayloadStore(storage, index)
            assert index.begin(abandoned)
            lease = service.prepare_store(abandoned, 0)
            assert isinstance(lease, CheckpointSlots)
            # The worker never finishes; its write locks expire.
            time.sleep(1.1)

            assert index.begin(later)
            deadline = time.monotonic() + 10
            while True:
                result = service.prepare_store(later, 0)
                if isinstance(result, CheckpointSlots):
                    break
                assert time.monotonic() < deadline, result
                time.sleep(0.1)
            assert all(slot is not None for group in result.groups for slot in group)
            assert service.finish_store(result.lease_id, True)
    finally:
        index.close()
        storage.close()
