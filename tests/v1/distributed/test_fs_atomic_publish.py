# SPDX-License-Identifier: Apache-2.0
"""Atomic publication tests for filesystem L2 cache objects."""

# Standard
from concurrent.futures import ThreadPoolExecutor
import select
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import _publish_temp_file


def test_python_publish_keeps_one_complete_immutable_file(tmp_path):
    """Only one completed inode may acquire a shared cache key."""
    file_path = tmp_path / "shared.data"
    payloads = [bytes([value]) * (1024 * 1024) for value in range(8)]
    temp_paths = []
    for index, payload in enumerate(payloads):
        temp_path = tmp_path / f"shared.data.tmp.{index}"
        temp_path.write_bytes(payload)
        temp_paths.append(temp_path)

    with ThreadPoolExecutor(max_workers=len(temp_paths)) as executor:
        published = list(
            executor.map(
                lambda path: _publish_temp_file(path, file_path),
                temp_paths,
            )
        )

    assert published.count(True) == 1
    assert file_path.read_bytes() in payloads
    assert all(not path.exists() for path in temp_paths)


def _wait_for_completion(client, timeout: float = 10.0):
    poller = select.poll()
    poller.register(client.event_fd(), select.POLLIN)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if poller.poll(50):
            completions = client.drain_completions()
            if completions:
                return completions
    raise AssertionError("native connector completion timed out")


def test_native_duplicate_stores_publish_one_complete_file(tmp_path):
    """Native store workers must not share writable temporary storage.

    Distinct byte patterns make a torn publication observable. Production
    callers derive keys from content and submit identical bytes for duplicate
    keys, but correctness must not depend on worker timing.
    """
    lmcache_fs = pytest.importorskip("lmcache.lmcache_fs")
    client = lmcache_fs.LMCacheFSClient(str(tmp_path), 2)
    try:
        payload_size = 8 * 1024 * 1024
        first = bytearray(b"a" * payload_size)
        second = bytearray(b"b" * payload_size)
        key = "model@00000000@duplicate"

        future_id = client.submit_batch_set(
            [key, key],
            [memoryview(first), memoryview(second)],
        )
        completions = _wait_for_completion(client)

        assert len(completions) == 1
        completed_id, ok, error, results = completions[0]
        assert completed_id == future_id
        assert ok, error
        if results is not None:
            assert results == [True, True]

        published = (tmp_path / "model@0x00000000@duplicate.data").read_bytes()
        # The object file is the payload followed by the integrity trailer.
        assert published[: len(first)] in (first, second)
        assert list(tmp_path.glob("*.tmp.*")) == []
    finally:
        client.close()
