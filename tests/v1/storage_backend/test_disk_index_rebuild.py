# SPDX-License-Identifier: Apache-2.0
"""Tests for the LocalDiskBackend index rebuild (CPU-only, no GPU/torch.distributed).

These exercise the real ``LocalDiskBackend.insert_key``, the real
``CacheEngineKey`` parser and the real LRU cache policy against a stub backend
that carries only the attributes those functions touch, so the filename<->key
round trip and the cap accounting are validated against production code rather
than a reimplementation.
"""

# Standard
from dataclasses import dataclass
import inspect
import os
import re
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.storage_backend import disk_index_rebuild
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.disk_index_rebuild import (
    purge_rebuilt,
    read_layout,
    rebuild_disk_index,
    write_layout,
)
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend

CHUNK_SIZE = 256
PACKED_BYTES = 1024
CHUNK_FILE_SIZE = CHUNK_SIZE * PACKED_BYTES
FMT = (
    MemoryFormat.KV_2LTD if hasattr(MemoryFormat, "KV_2LTD") else list(MemoryFormat)[0]
)
_OWN_ATTRS = {"_rebuilt_keys"}


@dataclass
class _Config:
    chunk_size: int = CHUNK_SIZE


@dataclass
class _Metadata:
    model_name: str = "/model"
    world_size: int = 4
    worker_id: int = 0
    kv_dtype: torch.dtype = torch.bfloat16


CONFIG = _Config()


class _StubBackend:
    """Stand-in limited to attributes ``LocalDiskBackend.__init__`` really assigns.

    Nothing may be added here that the real class does not have: an invented
    ``config`` attribute once made this suite pass while the running server threw
    ``AttributeError`` on the first cache init. See
    ``test_rebuild_only_reads_attributes_the_real_backend_defines``.
    """

    def __init__(self, path, max_cache_size=1 << 40):
        self.path = str(path)
        self.cache_policy = get_cache_policy("LRU")
        self.dict = self.cache_policy.init_mutable_mapping()
        self.disk_lock = threading.Lock()
        self.batched_msg_sender = None
        self.max_cache_size = max_cache_size
        self.current_cache_size = 0.0
        self.usage = 0
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

    _key_to_path = LocalDiskBackend._key_to_path
    insert_key = LocalDiskBackend.insert_key
    remove = LocalDiskBackend.remove


def _key(metadata, chunk_hash):
    return CacheEngineKey(
        model_name=metadata.model_name,
        world_size=metadata.world_size,
        worker_id=metadata.worker_id,
        chunk_hash=chunk_hash,
        dtype=metadata.kv_dtype,
    )


def _write_chunk(backend, key, size=CHUNK_FILE_SIZE, mtime=None):
    path = backend._key_to_path(key)
    with open(path, "wb") as handle:
        handle.write(b"\0" * size)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _populate(tmp_path, metadata, count, max_cache_size=1 << 40, size=CHUNK_FILE_SIZE):
    backend = _StubBackend(tmp_path, max_cache_size=max_cache_size)
    keys = [_key(metadata, 0x1000 + i) for i in range(count)]
    for i, key in enumerate(keys):
        _write_chunk(backend, key, size=size, mtime=1_000_000 + i)
    write_layout(backend.path, CHUNK_SIZE, PACKED_BYTES, torch.uint8, FMT)
    return backend, keys


def test_adopts_existing_chunks_and_accounts_bytes(tmp_path):
    metadata = _Metadata()
    backend, keys = _populate(tmp_path, metadata, 5)

    adopted = rebuild_disk_index(backend, CONFIG, metadata)

    assert adopted == 5 * CHUNK_FILE_SIZE
    assert backend.current_cache_size == 5 * CHUNK_FILE_SIZE
    assert backend.usage == 5 * CHUNK_FILE_SIZE
    assert set(backend.dict) == set(keys)
    meta = backend.dict[keys[0]]
    assert meta.size == CHUNK_FILE_SIZE
    assert tuple(meta.shape) == (1, 1, CHUNK_SIZE, PACKED_BYTES)
    assert meta.dtype == torch.uint8
    assert meta.fmt == FMT


def test_lru_order_follows_mtime(tmp_path):
    metadata = _Metadata()
    backend, keys = _populate(tmp_path, metadata, 4)

    rebuild_disk_index(backend, CONFIG, metadata)

    assert list(backend.dict) == keys


def test_model_name_with_slash_round_trips(tmp_path):
    metadata = _Metadata(model_name="/models/GLM-5.2-EXL3-TR3-3.0bpw")
    backend, keys = _populate(tmp_path, metadata, 3)

    assert rebuild_disk_index(backend, CONFIG, metadata) == 3 * CHUNK_FILE_SIZE
    assert set(backend.dict) == set(keys)


def test_rejects_chunks_belonging_to_another_model(tmp_path):
    backend, _ = _populate(tmp_path, _Metadata(), 3)

    assert (
        rebuild_disk_index(backend, CONFIG, _Metadata(model_name="/other-model")) == 0
    )
    assert len(backend.dict) == 0
    assert backend.current_cache_size == 0


@pytest.mark.parametrize("metadata", [_Metadata(world_size=8), _Metadata(worker_id=1)])
def test_adopts_tp_agnostic_dcp_keys_whatever_this_run_topology_is(tmp_path, metadata):
    """The writer's keys are ``<model>@1@0@<hash>@uint8`` under save_only_first_rank.

    Validating those fields against the run's metadata rejected 249 of 249 files
    the server itself had just written, so the topology must not be part of the
    predicate.
    """
    backend = _StubBackend(tmp_path)
    keys = [
        CacheEngineKey(
            model_name=metadata.model_name,
            world_size=1,
            worker_id=0,
            chunk_hash=0x3D77C89A43B0C8C4 + i,
            dtype=torch.uint8,
        )
        for i in range(3)
    ]
    for i, key in enumerate(keys):
        _write_chunk(backend, key, mtime=1_000_000 + i)
    write_layout(backend.path, CHUNK_SIZE, PACKED_BYTES, torch.uint8, FMT)

    assert rebuild_disk_index(backend, CONFIG, metadata) == 3 * CHUNK_FILE_SIZE
    assert set(backend.dict) == set(keys)


def test_rejects_wrong_sized_files(tmp_path):
    metadata = _Metadata()
    backend, _ = _populate(tmp_path, metadata, 2, size=CHUNK_FILE_SIZE + 1)

    assert rebuild_disk_index(backend, CONFIG, metadata) == 0
    assert len(backend.dict) == 0


def test_no_sidecar_means_no_adoption(tmp_path):
    metadata = _Metadata()
    backend, _ = _populate(tmp_path, metadata, 3)
    os.remove(os.path.join(backend.path, ".lmcache_layout.json"))

    assert rebuild_disk_index(backend, CONFIG, metadata) == 0
    assert len(backend.dict) == 0


def test_sidecar_chunk_size_mismatch_means_no_adoption(tmp_path):
    metadata = _Metadata()
    backend, _ = _populate(tmp_path, metadata, 3)
    write_layout(backend.path, CHUNK_SIZE * 2, PACKED_BYTES, torch.uint8, FMT)

    assert rebuild_disk_index(backend, CONFIG, metadata) == 0
    assert len(backend.dict) == 0


def test_enforces_cap_keeping_newest_and_deleting_the_rest(tmp_path):
    metadata = _Metadata()
    cap = 3 * CHUNK_FILE_SIZE
    backend, keys = _populate(tmp_path, metadata, 5, max_cache_size=cap)

    adopted = rebuild_disk_index(backend, CONFIG, metadata)

    assert adopted == cap
    assert backend.current_cache_size == cap
    assert set(backend.dict) == set(keys[2:])
    assert not os.path.exists(backend._key_to_path(keys[0]))
    assert not os.path.exists(backend._key_to_path(keys[1]))
    assert os.path.exists(backend._key_to_path(keys[4]))


def test_purge_removes_adopted_chunks_from_index_and_disk(tmp_path):
    metadata = _Metadata()
    backend, keys = _populate(tmp_path, metadata, 3)
    rebuild_disk_index(backend, CONFIG, metadata)

    freed = purge_rebuilt(backend, "layout mismatch")

    assert freed == 3 * CHUNK_FILE_SIZE
    assert len(backend.dict) == 0
    assert backend.current_cache_size == 0
    assert backend.usage == 0
    assert all(not os.path.exists(backend._key_to_path(key)) for key in keys)


def test_layout_sidecar_round_trips(tmp_path):
    write_layout(str(tmp_path), CHUNK_SIZE, PACKED_BYTES, torch.uint8, FMT)
    layout = read_layout(str(tmp_path))

    assert layout is not None
    assert layout["chunk_size"] == CHUNK_SIZE
    assert layout["packed_bytes"] == PACKED_BYTES
    assert layout["dtype"] == "uint8"
    assert layout["fmt"] == FMT.value


def test_unreadable_sidecar_is_ignored(tmp_path):
    metadata = _Metadata()
    backend, _ = _populate(tmp_path, metadata, 2)
    with open(os.path.join(backend.path, ".lmcache_layout.json"), "w") as handle:
        handle.write("{not json")

    assert rebuild_disk_index(backend, CONFIG, metadata) == 0


def test_rebuild_only_reads_attributes_the_real_backend_defines():
    source = inspect.getsource(LocalDiskBackend.__init__)
    assigned = set(re.findall(r"self\.(\w+)\s*(?::[^=\n]+)?=", source))
    available = assigned | set(dir(LocalDiskBackend)) | _OWN_ATTRS
    used = set(re.findall(r"backend\.(\w+)", inspect.getsource(disk_index_rebuild)))

    assert used <= available, (
        f"disk_index_rebuild reads {sorted(used - available)} off the backend, but "
        "LocalDiskBackend never defines them; a stub with those attributes would "
        "pass while the server raises AttributeError during cache init"
    )


def test_missing_chunk_file_does_not_break_removal(tmp_path):
    metadata = _Metadata()
    backend, keys = _populate(tmp_path, metadata, 2)
    rebuild_disk_index(backend, CONFIG, metadata)
    os.remove(backend._key_to_path(keys[0]))

    assert backend.remove(keys[0]) is True
    assert keys[0] not in backend.dict
    assert backend.usage == CHUNK_FILE_SIZE
    assert backend.remove(keys[1]) is True
    assert backend.usage == 0


def test_reput_of_an_indexed_key_claims_no_additional_space():
    source = inspect.getsource(LocalDiskBackend.submit_put_task)

    assert "overwrite = key in self.dict" in source
    assert "while not overwrite and (" in source
    assert "if evict_success and not overwrite:" in source, (
        "re-storing an already-indexed key overwrites one deterministic path and "
        "consumes no new bytes; counting it again inflates current_cache_size until "
        "a phantom eviction storm deletes the whole cache"
    )
    assert "if key not in self.dict:" in inspect.getsource(
        LocalDiskBackend.async_save_bytes_to_disk
    ), "reported disk usage must not grow on overwrite either"
