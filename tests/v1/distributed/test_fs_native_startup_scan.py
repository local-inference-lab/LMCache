# SPDX-License-Identifier: Apache-2.0
"""Startup scan for the fs_native L2 adapter factory.

After a restart the backend still holds every previously stored file, but
byte accounting starts at zero, leaving L2 eviction blind to pre-existing
disk usage. ``_scan_existing_fs_native_files`` recovers ``(keys, sizes)``
from the on-disk files so the adapter can prime its accounting.
"""

# Standard
from pathlib import Path

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    _object_key_to_filename,
)
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
    _scan_existing_fs_native_files,
)


def _make_key(chunk_id: int, cache_salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
        cache_salt=cache_salt,
    )


def _write_data_file(base: Path, key: ObjectKey, num_bytes: int) -> None:
    (base / _object_key_to_filename(key)).write_bytes(b"x" * num_bytes)


def test_scan_returns_keys_and_sizes(tmp_path):
    keys = [_make_key(1), _make_key(2, cache_salt="salted")]
    _write_data_file(tmp_path, keys[0], 100)
    _write_data_file(tmp_path, keys[1], 200)

    scanned_keys, scanned_sizes = _scan_existing_fs_native_files(str(tmp_path))

    by_key = dict(zip(scanned_keys, scanned_sizes, strict=True))
    assert by_key == {keys[0]: 100, keys[1]: 200}


def test_scan_skips_foreign_and_empty_files(tmp_path):
    _write_data_file(tmp_path, _make_key(1), 100)
    # Unparseable name with the right extension.
    (tmp_path / "not-a-cache-file.data").write_bytes(b"x")
    # Wrong extension.
    (tmp_path / "temp.tmp").write_bytes(b"x")
    # Empty file.
    _write_data_file(tmp_path, _make_key(2), 0)

    scanned_keys, scanned_sizes = _scan_existing_fs_native_files(str(tmp_path))

    assert scanned_keys == [_make_key(1)]
    assert scanned_sizes == [100]


def test_scan_missing_directory_returns_empty(tmp_path):
    scanned_keys, scanned_sizes = _scan_existing_fs_native_files(
        str(tmp_path / "does-not-exist")
    )

    assert scanned_keys == []
    assert scanned_sizes == []
