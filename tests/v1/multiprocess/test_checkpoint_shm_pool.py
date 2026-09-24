# SPDX-License-Identifier: Apache-2.0
"""A worker's checkpoint SHM mapping notices a server-recreated pool."""

# Standard
from multiprocessing import shared_memory
import os
import uuid

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.transfer_context.shm import ShmPoolMapping

SIZE = 64 * 1024


@pytest.mark.skipif(not os.path.isdir("/dev/shm"), reason="requires POSIX /dev/shm")
def test_mapping_detects_a_recreated_pool_and_remaps() -> None:
    name = f"lmcache_checkpoint_pool_test_{uuid.uuid4().hex}"
    server = shared_memory.SharedMemory(name=name, create=True, size=SIZE)
    try:
        server.buf[:4] = b"old!"
        mapping = ShmPoolMapping(name, SIZE)
        try:
            assert mapping.is_current()
            # A restarted server unlinks the pool and creates it again.
            server.close()
            server.unlink()
            server = shared_memory.SharedMemory(name=name, create=True, size=SIZE)
            server.buf[:4] = b"new!"
            assert not mapping.is_current()
            mapping.remap()
            assert mapping.is_current()
            assert mapping._shm_buffer is not None
            assert bytes(mapping._shm_buffer[:4]) == b"new!"
        finally:
            mapping.close()
    finally:
        server.close()
        server.unlink()
