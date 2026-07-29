# SPDX-License-Identifier: Apache-2.0
"""Reconstruct ``LocalDiskBackend``'s in-memory index from an existing cache directory.

LMCache 0.5.2 keeps the disk index only in memory. ``LocalDiskBackend.__init__``
starts with an empty ``dict`` and ``current_cache_size == 0`` and never enumerates
its directory, so chunks written by earlier server lifetimes are both unreachable
and uncounted. Two consequences:

  * the L2 (NVMe) tier loses everything on restart, defeating its purpose;
  * ``max_local_disk_size`` degrades into a per-process allowance, so the
    directory grows past the cap on every restart (measured: 522 GB on disk for a
    200 GiB cache, of which 345.7 GB was invisible to the running process).

A rebuild is possible because ``_key_to_path`` is deterministic and
self-describing::

    <model_name>@<world_size>@<worker_id>@<chunk_hash_hex>@<dtype>.pt   ('/' -> '-')

and ``CacheEngineKey.from_string`` parses that form. Chunks are stored as raw
bytes with no header, so shape/dtype/fmt cannot come from the file; they come from
a sidecar written by ``sync_layout`` for the layout the current stack produces
(``dcp_gather._dcp_store`` calls it once per process).

Registration requires *all* of:

  * the parsed key's canonical path is identical to the file found, which proves
    the ``'/' -> '-'`` mangling was inverted correctly;
  * the model name matches this run's metadata;
  * ``chunk_size`` matches this run's config;
  * file size == ``chunk_size * packed_bytes`` from the sidecar.

The key's world size, worker id and dtype are deliberately *not* compared against
``metadata``: under ``save_only_first_rank`` the writer stores TP-agnostic keys
(``world_size=1``, ``worker_id=0``, and the packed byte dtype rather than the KV
dtype), so a TP=4 run legitimately owns ``<model>@1@0@<hash>@uint8.pt``. Comparing
them to the run's topology rejected every file the server had just written.

A directory holding another model's chunks, or chunks with a different byte
layout, therefore contributes nothing instead of producing a false hit. As a final
guard, ``sync_layout`` purges everything this module registered if the running
stack turns out to produce a different byte layout than the sidecar promised.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any, Optional
import json
import os
import tempfile

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat

logger = init_logger(__name__)

LAYOUT_FILE = ".lmcache_layout.json"
_LAYOUT_VERSION = 1


def _layout_path(directory: str) -> str:
    return os.path.join(directory, LAYOUT_FILE)


def read_layout(directory: str) -> Optional[dict[str, Any]]:
    """Return the sidecar layout for ``directory``, or None if absent/unreadable."""
    try:
        with open(_layout_path(directory), "r") as handle:
            layout = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(layout, dict) or layout.get("version") != _LAYOUT_VERSION:
        return None
    return layout


def write_layout(
    directory: str, chunk_size: int, packed_bytes: int, dtype: torch.dtype, fmt: Any
) -> None:
    """Atomically record the byte layout of the chunks stored in ``directory``."""
    layout = {
        "version": _LAYOUT_VERSION,
        "chunk_size": int(chunk_size),
        "packed_bytes": int(packed_bytes),
        "dtype": str(dtype).removeprefix("torch."),
        "fmt": int(getattr(fmt, "value", fmt)),
    }
    try:
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".layout-")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(layout, handle)
            os.replace(tmp, _layout_path(directory))
        except BaseException:
            os.unlink(tmp)
            raise
    except OSError as exc:
        logger.warning("Could not write LMCache disk layout sidecar: %s", exc)


def _layout_matches(
    layout: dict[str, Any],
    chunk_size: int,
    packed_bytes: int,
    dtype: torch.dtype,
    fmt: Any,
) -> bool:
    return (
        int(layout.get("chunk_size", -1)) == int(chunk_size)
        and int(layout.get("packed_bytes", -1)) == int(packed_bytes)
        and str(layout.get("dtype")) == str(dtype).removeprefix("torch.")
        and int(layout.get("fmt", -1)) == int(getattr(fmt, "value", fmt))
    )


def _parse_key(backend: Any, filename: str, metadata: Any) -> Optional[CacheEngineKey]:
    """Parse a cache filename back into its key, or None if it is not ours."""
    parts = filename[: -len(".pt")].split("@")
    if len(parts) < 5:
        return None
    parts[0] = metadata.model_name
    try:
        key = CacheEngineKey.from_string("@".join(parts))
    except (ValueError, KeyError, IndexError):
        return None
    if key.model_name != metadata.model_name:
        return None
    if os.path.basename(backend._key_to_path(key)) != filename:
        return None
    return key


def _scan(backend: Any, metadata: Any, expected_size: int) -> tuple[list, int]:
    prefix = metadata.model_name.replace("/", "-") + "@"
    found, skipped = [], 0
    try:
        entries = list(os.scandir(backend.path))
    except OSError as exc:
        logger.warning(
            "LMCache disk index rebuild: cannot scan %s: %s", backend.path, exc
        )
        return [], 0
    for entry in entries:
        name = entry.name
        if not name.endswith(".pt"):
            continue
        if not name.startswith(prefix):
            skipped += 1
            continue
        try:
            stat = entry.stat()
        except OSError:
            continue
        if stat.st_size != expected_size:
            skipped += 1
            continue
        key = _parse_key(backend, name, metadata)
        if key is None:
            skipped += 1
            continue
        found.append((stat.st_mtime, key, stat.st_size, entry.path))
    return found, skipped


def _drop_file(path: str) -> None:
    try:
        os.remove(path)
    except OSError as exc:
        logger.warning("LMCache disk index rebuild: cannot remove %s: %s", path, exc)


def rebuild_disk_index(backend: Any, config: Any, metadata: Any) -> int:
    """Register the directory's existing chunks in ``backend``'s index.

    ``config`` must be passed in: ``LocalDiskBackend`` keeps it as a constructor
    local and never stores it on the instance, so it is unreachable from
    ``backend``.

    Returns the number of bytes adopted. Chunks beyond ``max_cache_size`` are
    deleted newest-first-kept, so the cap is enforced against real on-disk bytes
    for the first time. Registration order follows mtime, so LRU recency after a
    restart approximates the recency the cache had before it.
    """
    if metadata is None or config is None:
        return 0
    layout = read_layout(backend.path)
    if layout is None:
        return 0

    chunk_size = int(getattr(config, "chunk_size", 0) or 0)
    packed_bytes = int(layout.get("packed_bytes", 0))
    if chunk_size <= 0 or packed_bytes <= 0:
        return 0
    if int(layout.get("chunk_size", -1)) != chunk_size:
        logger.warning(
            "LMCache disk index rebuild: sidecar chunk_size=%s but config has %s; "
            "not adopting the existing cache.",
            layout.get("chunk_size"),
            chunk_size,
        )
        return 0
    dtype = getattr(torch, str(layout.get("dtype", "")), None)
    if not isinstance(dtype, torch.dtype):
        return 0
    try:
        fmt = MemoryFormat(int(layout.get("fmt", -1)))
    except ValueError:
        return 0

    expected_size = chunk_size * packed_bytes
    found, skipped = _scan(backend, metadata, expected_size)
    if not found and not skipped:
        return 0

    found.sort(key=lambda item: item[0], reverse=True)
    keep, evicted_bytes = [], 0
    total = 0
    for mtime, key, size, path in found:
        if total + size <= backend.max_cache_size:
            total += size
            keep.append((mtime, key, size))
        else:
            evicted_bytes += size
            _drop_file(path)

    shape = torch.Size([1, 1, chunk_size, packed_bytes])
    keep.sort(key=lambda item: item[0])
    adopted = 0
    rebuilt = set()
    for _, key, size in keep:
        backend.insert_key(key, size, shape, dtype, fmt)
        adopted += size
        rebuilt.add(key)

    backend.current_cache_size += adopted
    backend.usage = getattr(backend, "usage", 0) + adopted
    backend._rebuilt_keys = rebuilt
    stats = getattr(backend, "stats_monitor", None)
    if stats is not None:
        stats.update_local_storage_usage(backend.usage)

    logger.info(
        "LMCache disk index rebuilt from %s: adopted %d chunks (%.1f GB), "
        "dropped %.1f GB over the %.1f GB cap, skipped %d foreign files.",
        backend.path,
        len(rebuilt),
        adopted / 1e9,
        evicted_bytes / 1e9,
        backend.max_cache_size / 1e9,
        skipped,
    )
    return adopted


def purge_rebuilt(backend: Any, reason: str) -> int:
    """Delete everything ``rebuild_disk_index`` adopted, on-disk and in the index."""
    rebuilt = getattr(backend, "_rebuilt_keys", None)
    if not rebuilt:
        return 0
    freed = 0
    for key in rebuilt:
        meta = backend.dict.pop(key, None)
        if meta is None:
            continue
        freed += int(getattr(meta, "size", 0) or 0)
        _drop_file(backend._key_to_path(key))
    backend.current_cache_size = max(0.0, backend.current_cache_size - freed)
    backend.usage = max(0, getattr(backend, "usage", 0) - freed)
    backend._rebuilt_keys = set()
    logger.warning(
        "Discarded %.1f GB of adopted LMCache disk chunks: %s", freed / 1e9, reason
    )
    return freed


def find_disk_backend(storage_manager: Any) -> Optional[Any]:
    """Return the LocalDiskBackend held by ``storage_manager``, if any."""
    backends = getattr(storage_manager, "storage_backends", None)
    if not backends:
        return None
    for backend in backends.values():
        if type(backend).__name__ == "LocalDiskBackend":
            return backend
    return None


def sync_layout(
    storage_manager: Any,
    chunk_size: int,
    packed_bytes: int,
    dtype: torch.dtype,
    fmt: Any,
) -> None:
    """Record this run's chunk layout, discarding adopted chunks that disagree.

    Called once per process by the writer. The sidecar is what makes the next
    restart able to interpret these files; the purge is the last line of defence
    against adopting chunks whose byte layout differs from what this run produces
    (the key alone cannot distinguish two KV dtypes that share a torch dtype).
    """
    backend = find_disk_backend(storage_manager)
    if backend is None:
        return
    layout = read_layout(backend.path)
    if layout is not None and not _layout_matches(
        layout, chunk_size, packed_bytes, dtype, fmt
    ):
        purge_rebuilt(
            backend,
            f"on-disk layout {layout} does not match this run "
            f"(chunk_size={chunk_size}, packed_bytes={packed_bytes}, "
            f"dtype={dtype}, fmt={int(getattr(fmt, 'value', fmt))})",
        )
    write_layout(backend.path, chunk_size, packed_bytes, dtype, fmt)
