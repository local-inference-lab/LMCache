# SPDX-License-Identifier: Apache-2.0
"""A worker's checkpoint SHM mapping notices a server-recreated pool."""

# Standard
from multiprocessing import shared_memory
import os
import uuid

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.protocols.checkpoint import (
    CheckpointCapabilities,
    CheckpointLeaseResponse,
)
from lmcache.v1.multiprocess.transfer_context.shm import ShmPoolMapping

SIZE = 64 * 1024


def first_bytes(mapping: ShmPoolMapping, name: str) -> bytes:
    """Read the pool's first four bytes through a checkpoint lease view."""
    (view,) = mapping.checkpoint_slot_views(
        CheckpointCapabilities(format_version=1, shm_name=name, pool_size=SIZE),
        CheckpointLeaseResponse("ready", "lease", (((0, 4),),)),
        ((4,),),
    )[0]
    assert view is not None
    return bytes(view.cpu().numpy())


@pytest.mark.skipif(not os.path.isdir("/dev/shm"), reason="requires POSIX /dev/shm")
def test_mapping_detects_a_recreated_pool_and_remaps() -> None:
    name = f"lmcache_checkpoint_pool_test_{uuid.uuid4().hex}"
    server = shared_memory.SharedMemory(name=name, create=True, size=SIZE)
    try:
        server.buf[:4] = b"old!"
        mapping = ShmPoolMapping(name, SIZE)
        try:
            assert mapping.is_current()
            assert first_bytes(mapping, name) == b"old!"
            # A restarted server unlinks the pool and creates it again.
            server.close()
            server.unlink()
            server = shared_memory.SharedMemory(name=name, create=True, size=SIZE)
            server.buf[:4] = b"new!"
            assert not mapping.is_current()
            mapping.remap()
            assert mapping.is_current()
            assert first_bytes(mapping, name) == b"new!"
        finally:
            mapping.close()
    finally:
        server.close()
        server.unlink()


@pytest.mark.skipif(not os.path.isdir("/dev/shm"), reason="requires POSIX /dev/shm")
def test_failed_remap_is_retried_by_the_next_lease() -> None:
    name = f"lmcache_checkpoint_pool_test_{uuid.uuid4().hex}"
    server = shared_memory.SharedMemory(name=name, create=True, size=SIZE)
    mapping = ShmPoolMapping(name, SIZE)
    try:
        # The old server is gone and the new one has not created its pool yet.
        server.close()
        server.unlink()
        assert not mapping.is_current()
        with pytest.raises(FileNotFoundError):
            mapping.remap()
        assert not mapping.is_current()
        server = shared_memory.SharedMemory(name=name, create=True, size=SIZE)
        server.buf[:4] = b"back"
        mapping.remap()
        assert mapping.is_current()
        assert first_bytes(mapping, name) == b"back"
    finally:
        mapping.close()
        server.close()
        server.unlink()
