# SPDX-License-Identifier: Apache-2.0
"""Checkpoint restores whose RAM reservation fails keep intact checkpoints listed.

A restore from disk reserves RAM for all of a checkpoint's pages. When that
reservation fails, the checkpoint must stay listed so a later request restores
it; only pages that are really missing retire it.
"""

# Standard
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch
import hashlib
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from tests.v1.multiprocess.test_checkpoint_storage import (
    entry_keys,
    large_manifest,
    make_manifest,
    open_store,
    poll,
    publish_before_restart,
    restores,
)


@pytest.mark.parametrize("native", [False, True])
def test_ram_released_before_the_poll_keeps_checkpoint_listed(
    tmp_path: Path, native: bool
) -> None:
    """A writer holds all RAM while the load buffers are reserved and frees it
    before the store reads the result: the checkpoint stays listed."""
    entry = replace(make_manifest(), world_size=1)
    publish_before_restart(tmp_path, native, [entry])
    with open_store(tmp_path, native, admission_timeout_seconds=0.15) as (
        service,
        index,
        storage,
        mapping,
    ):
        filler = ObjectKey(hashlib.sha256(b"probe-writer").digest(), "filler", 0, 0)
        _, total = storage.get_l1_usage()
        submit = storage.submit_prefetch_task
        query = storage.query_prefetch_status_detailed
        attempts = 0

        def reserve_then_submit(*args: Any, **kwargs: Any) -> Any:
            nonlocal attempts
            attempts += 1
            result = storage.reserve_write_detailed(
                [filler], MemoryLayoutDesc([torch.Size([total])], [torch.uint8]), "new"
            )
            assert result[filler][1] is not None
            return submit(*args, **kwargs)

        def release_before_consumption(handle: Any) -> Any:
            result = query(handle)
            if result is not None:
                storage.abort_write([filler])
            return result

        try:
            with (
                patch.object(
                    storage, "submit_prefetch_task", side_effect=reserve_then_submit
                ),
                patch.object(
                    storage,
                    "query_prefetch_status_detailed",
                    side_effect=release_before_consumption,
                ),
            ):
                lease_id = service.begin_retrieve(entry, 0)
                assert lease_id is not None
                outcome = poll(service, lease_id)
            listed = index.get(entry.generation) == entry
            assert outcome is False
            assert listed, "intact disk checkpoint was delisted"
        finally:
            storage.abort_write([filler])
        assert restores(service, mapping, entry)


@pytest.mark.parametrize("native", [False, True])
def test_fragmented_ram_keeps_checkpoint_listed_without_spinning(
    tmp_path: Path, native: bool
) -> None:
    """Free RAM exceeds the unread pages, but no free block holds one page.

    No concurrency: 64 KiB holes between pinned 64 KiB blocks, 128 KiB
    pages. The reservation fails every time while get_l1_usage shows room, so
    the lookup is repeated with a pause until the admission timeout.
    """
    entry = large_manifest(500)
    publish_before_restart(tmp_path, native, [entry])
    with open_store(tmp_path, native, admission_timeout_seconds=1.0) as (
        service,
        index,
        storage,
        mapping,
    ):
        used, total = storage.get_l1_usage()
        assert used == 0
        block = 64 * 1024
        fillers = [
            ObjectKey(hashlib.sha256(b"frag%d" % i).digest(), "filler", 0, 0)
            for i in range(total // block)
        ]
        reserved = storage.reserve_write_detailed(
            fillers, MemoryLayoutDesc([torch.Size([block])], [torch.uint8]), "new"
        )
        assert all(obj is not None for _err, obj in reserved.values())
        storage.abort_write(fillers[::2])
        kept = fillers[1::2]
        used, total = storage.get_l1_usage()
        unread = len(entry_keys(entry)) * 128 * 1024
        assert total - used >= unread
        submit = storage.submit_prefetch_task
        attempts = 0

        def count(*args: Any, **kwargs: Any) -> Any:
            nonlocal attempts
            attempts += 1
            return submit(*args, **kwargs)

        try:
            with patch.object(storage, "submit_prefetch_task", side_effect=count):
                start = time.monotonic()
                lease_id = service.begin_retrieve(entry, 0)
                assert lease_id is not None
                outcome = poll(service, lease_id)
                elapsed = time.monotonic() - start
            listed = index.get(entry.generation) == entry
            assert outcome is False
            assert listed, "intact disk checkpoint was delisted"
            # Repeats after a RAM failure pause 20 ms: about 50 in one second.
            assert attempts <= 80, f"{attempts} lookups in {elapsed:.2f} s"
        finally:
            storage.abort_write(kept)
        assert restores(service, mapping, entry)


@pytest.mark.parametrize(
    "timeout,delay",
    [(0.2, 0.0), (0.2, 0.4), (0.0, 0.0)],
    ids=["fast-first-lookup", "first-lookup-slower-than-timeout", "timeout-0"],
)
def test_lost_page_retires_checkpoint(
    tmp_path: Path, timeout: float, delay: float
) -> None:
    """A page file deleted from disk (not a RAM problem) retires its checkpoint.

    Also when the first lookup is answered after the admission timeout, or the
    timeout is zero: the timeout bounds only repeats after a RAM failure.
    """
    entry = replace(make_manifest(), world_size=1)
    publish_before_restart(tmp_path, False, [entry])
    files = sorted((tmp_path / "payloads").rglob("*.data"))
    assert len(files) == len(entry_keys(entry))
    files[0].unlink()
    with open_store(tmp_path, False, admission_timeout_seconds=timeout) as (
        service,
        index,
        storage,
        _mapping,
    ):
        lease_id = service.begin_retrieve(entry, 0)
        assert lease_id is not None
        time.sleep(delay)
        outcome = poll(service, lease_id)
        listed = index.get(entry.generation) == entry
        assert outcome is False
        assert not listed, "checkpoint with a lost page stays listed"
