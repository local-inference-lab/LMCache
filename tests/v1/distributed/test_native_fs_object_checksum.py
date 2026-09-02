# SPDX-License-Identifier: Apache-2.0
"""Native FS connector object integrity trailer.

Every object file ends with a 16-byte trailer (magic, CRC-32C of the
payload, payload length) that a load verifies; a corrupted payload, a
truncated or oversized file and a trailer for another payload are reported
as per-key load failures. Objects without a trailer (file size == payload
size) are legacy objects and load unverified. These tests exercise the
compiled extension and are skipped when it is absent.
"""

# Standard
import os
import select
import struct
import time

# Third Party
import pytest

lmcache_fs = pytest.importorskip("lmcache.lmcache_fs")

TRAILER = struct.Struct("<IIQ")
MAGIC = 0x31434D4C
KEY = "model@00000000@0@" + "ab" * 32


def _wait(client, timeout: float = 5.0):
    poller = select.poll()
    poller.register(client.event_fd(), select.POLLIN)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if poller.poll(50):
            completions = client.drain_completions()
            if completions:
                return completions
    raise AssertionError("native connector completion timed out")


def _store(client, key: str, payload: bytes) -> None:
    buffer = bytearray(payload)  # must outlive the asynchronous write
    client.submit_batch_set([key], [memoryview(buffer)])
    _, ok, error, results = _wait(client)[0]
    assert ok, error
    assert results == [True]


def _load(client, key: str, size: int):
    """Return (per-key results, bytes read). A failed key is reported through
    the per-key results; the batch itself completes."""
    out = bytearray(size)
    client.submit_batch_get([key], [memoryview(out)])
    _, _ok, _error, results = _wait(client)[0]
    return results, bytes(out)


def _object_path(tmp_path):
    files = list(tmp_path.glob("*.data"))
    assert len(files) == 1
    return files[0]


def _crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return crc ^ 0xFFFFFFFF


def test_store_appends_verifiable_trailer_and_load_succeeds(tmp_path):
    client = lmcache_fs.LMCacheFSClient(str(tmp_path), 1)
    try:
        payload = bytes(range(256)) * 64  # 16 KiB
        _store(client, KEY, payload)
        path = _object_path(tmp_path)
        raw = path.read_bytes()
        assert len(raw) == len(payload) + TRAILER.size
        magic, crc, length = TRAILER.unpack(raw[len(payload):])
        assert magic == MAGIC
        assert length == len(payload)
        assert crc == _crc32c(payload)

        results, data = _load(client, KEY, len(payload))
        assert results == [True]
        assert data == payload
    finally:
        client.close()


def test_corrupted_payload_fails_the_load(tmp_path):
    client = lmcache_fs.LMCacheFSClient(str(tmp_path), 1)
    try:
        payload = b"\x5a" * 4096
        _store(client, KEY, payload)
        path = _object_path(tmp_path)
        with open(path, "r+b") as fh:
            fh.seek(1000)
            fh.write(b"\x00")

        results, _ = _load(client, KEY, len(payload))
        assert results == [False]
    finally:
        client.close()


def test_legacy_object_without_trailer_loads(tmp_path):
    client = lmcache_fs.LMCacheFSClient(str(tmp_path), 1)
    try:
        payload = b"\x33" * 2048
        _store(client, KEY, payload)
        path = _object_path(tmp_path)
        with open(path, "r+b") as fh:
            fh.truncate(len(payload))  # what an object written before trailers looks like

        results, data = _load(client, KEY, len(payload))
        assert results == [True]
        assert data == payload
    finally:
        client.close()


def test_truncated_and_oversized_objects_fail_the_load(tmp_path):
    client = lmcache_fs.LMCacheFSClient(str(tmp_path), 1)
    try:
        payload = b"\x11" * 4096
        _store(client, KEY, payload)
        path = _object_path(tmp_path)
        with open(path, "r+b") as fh:
            fh.truncate(len(payload) - 1)
        results, _ = _load(client, KEY, len(payload))
        assert results == [False]

        os.remove(path)
        _store(client, KEY, payload)
        with open(path, "ab") as fh:
            fh.write(b"\x00" * 8)
        results, _ = _load(client, KEY, len(payload))
        assert results == [False]
    finally:
        client.close()


def test_trailer_for_another_payload_length_fails_the_load(tmp_path):
    client = lmcache_fs.LMCacheFSClient(str(tmp_path), 1)
    try:
        payload = b"\x77" * 4096
        _store(client, KEY, payload)
        path = _object_path(tmp_path)
        raw = path.read_bytes()
        forged = raw[: len(payload)] + TRAILER.pack(MAGIC, _crc32c(payload), len(payload) - 16)
        path.write_bytes(forged)
        results, _ = _load(client, KEY, len(payload))
        assert results == [False]
    finally:
        client.close()
