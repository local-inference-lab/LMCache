# SPDX-License-Identifier: Apache-2.0
"""Publication, isolation and process-restart contract of checkpoint manifests."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.checkpoint_index import (
    CheckpointIndex,
    CheckpointManifest,
    CheckpointPrefix,
)


def manifest(
    generation: str = "producer-a", tail: tuple[int, ...] = (1, 2)
) -> CheckpointManifest:
    return CheckpointManifest(
        generation,
        CheckpointPrefix("weights-layout-salt-a", 4096, b"a" * 32, tail),
        4,
        b'{"schema_version":1,"payload_keys":["immutable-generation-pages"]}',
    )


def publish(index: CheckpointIndex, entry: CheckpointManifest) -> None:
    assert index.begin(entry)
    for rank in range(entry.world_size):
        assert index.acknowledge(entry.generation, rank) == (
            rank == entry.world_size - 1
        )


def test_only_complete_distinct_rank_set_is_visible() -> None:
    index = CheckpointIndex()
    entry = manifest()
    assert index.begin(entry)
    assert not index.begin(entry)
    for rank in (0, 0, 1, 2, 2):
        assert not index.acknowledge(entry.generation, rank)
        assert index.find((entry.prefix,)) is None
    with pytest.raises(ValueError, match="rank"):
        index.acknowledge(entry.generation, 4)
    assert index.acknowledge(entry.generation, 3)
    assert index.find((entry.prefix,)) == entry
    assert not index.acknowledge(entry.generation, 3)
    index.close()


def test_identical_published_manifest_is_idempotent_across_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint-index.sqlite3"
    entry = manifest("recurrent-content-v1:" + "a" * 64)
    index = CheckpointIndex(path)
    publish(index, entry)
    assert not index.begin(entry)
    index.close()

    index = CheckpointIndex(path)
    assert not index.begin(entry)
    with pytest.raises(ValueError, match="changed after publication"):
        index.begin(replace(entry, payload=b'{"schema_version":1,"changed":true}'))
    assert index.find((entry.prefix,)) == entry
    index.close()


def test_restart_keeps_complete_manifest_but_discards_pending(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint-index.sqlite3"
    entry = manifest()
    partial = manifest("producer-b", (1, 2, 3))
    index = CheckpointIndex(path)
    publish(index, entry)
    assert index.begin(partial)
    assert not index.acknowledge(partial.generation, 0)
    assert index.report_status() == {
        "published_generations": 1,
        "pending_generations": 1,
    }
    index.close()

    index = CheckpointIndex(path)
    assert index.find((partial.prefix,)) == entry
    assert not index.acknowledge(partial.generation, 1)
    assert index.report_status() == {
        "published_generations": 1,
        "pending_generations": 0,
    }
    index.close()


def test_two_producers_cannot_mix_rank_acknowledgements() -> None:
    index = CheckpointIndex()
    first, second = manifest(), manifest("producer-b")
    assert index.begin(first)
    assert index.begin(second)
    for rank in (0, 1):
        assert not index.acknowledge(first.generation, rank)
    for rank in (2, 3):
        assert not index.acknowledge(second.generation, rank)
    assert index.find((first.prefix,)) is None
    index.abort(first.generation)
    for rank in (0, 1):
        index.acknowledge(second.generation, rank)
    assert index.find((first.prefix,)) == second
    index.close()


def test_replacement_is_atomic_and_stale_invalidation_is_generation_scoped() -> None:
    index = CheckpointIndex()
    first, second = manifest(), manifest("producer-b")
    publish(index, first)
    assert index.begin(second)
    assert not index.acknowledge(second.generation, 0)
    assert index.find((first.prefix,)) == first
    for rank in (1, 2, 3):
        index.acknowledge(second.generation, rank)
    index.invalidate(first.generation)
    assert index.find((first.prefix,)) == second
    index.invalidate(second.generation)
    assert index.find((first.prefix,)) is None
    index.close()


@pytest.mark.parametrize(
    "change",
    [
        {"namespace": "different-model-draft-layout-or-salt"},
        {"prefix_hash": b"b" * 32},
        {"start_tokens": 8192},
        {"tail_tokens": (1, 3)},
        {"tail_tokens": (1,)},
    ],
)
def test_incompatible_or_shorter_prefix_misses(change: dict) -> None:
    index = CheckpointIndex()
    entry = manifest()
    publish(index, entry)
    assert index.find((replace(entry.prefix, **change),)) is None
    index.close()


def test_longest_matching_prefix_and_lru_capacity() -> None:
    index = CheckpointIndex(max_entries=2)
    short, long = manifest(), manifest("producer-b", (1, 2, 3))
    publish(index, short)
    publish(index, long)
    query = replace(long.prefix, tail_tokens=(1, 2, 3, 4))
    assert index.find((query,)) == long
    other = manifest("producer-c", (9, 8))
    publish(index, other)
    assert index.find((short.prefix,)) is None
    assert index.find((long.prefix,)) == long
    index.close()


def test_longest_match_is_not_the_longest_query_tail() -> None:
    index = CheckpointIndex()
    short = manifest()
    long = replace(
        manifest("producer-b"),
        prefix=CheckpointPrefix(short.prefix.namespace, 4098, b"b" * 32, (3, 4)),
    )
    publish(index, short)
    publish(index, long)
    longer_query = replace(short.prefix, tail_tokens=(1, 2, 3, 4, 5))
    assert index.find((longer_query, long.prefix)) == long
    index.close()


def test_pending_capacity_abort_and_manifest_identity() -> None:
    index = CheckpointIndex(max_pending=1)
    entry = manifest()
    assert index.begin(entry)
    assert not index.begin(manifest("producer-b"))
    with pytest.raises(ValueError, match="changed"):
        index.begin(replace(entry, payload=b'{"schema_version":1,"changed":true}'))
    index.abort(entry.generation)
    assert index.begin(manifest("producer-b"))
    index.close()


def test_concurrent_rank_acknowledgements_publish_once() -> None:
    index = CheckpointIndex()
    entry = manifest()
    assert index.begin(entry)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda rank: index.acknowledge(entry.generation, rank),
                (0, 1, 2, 3, 0, 1, 2, 3),
            )
        )
    assert sum(results) == 1
    assert index.find((entry.prefix,)) == entry
    index.close()
