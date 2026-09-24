# SPDX-License-Identifier: Apache-2.0
"""Publication, isolation and process-restart contract of checkpoint manifests."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import sqlite3

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess import checkpoint_index
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


@pytest.mark.parametrize(
    ("stored", "query", "hit"),
    [
        ((1, 2), (1, 2), True),
        ((1, 2), (1, 2, 3), True),
        ((1, 2), (1, 20), False),
        ((1,), (12, 3), False),
        ((12,), (1, 2, 3), False),
    ],
)
def test_tail_matches_whole_tokens_only(
    stored: tuple[int, ...], query: tuple[int, ...], hit: bool
) -> None:
    index = CheckpointIndex()
    entry = manifest(tail=stored)
    publish(index, entry)
    found = index.find((replace(entry.prefix, tail_tokens=query),))
    assert found == (entry if hit else None)
    index.close()


def test_ancestors_match_whole_tokens_only() -> None:
    index = CheckpointIndex()
    kept = [manifest(f"kept-{tail}", tail) for tail in ((1,), (1, 2))]
    decoys = [manifest(f"decoy-{tail}", tail) for tail in ((12,), (1, 20))]
    for entry in kept + decoys:
        publish(index, entry)
    query = replace(kept[0].prefix, tail_tokens=(1, 2, 3))
    found = index.ancestors((query,), below_tokens=query.num_tokens)
    assert found == kept
    index.close()


def test_longest_match_among_many_tails_sharing_a_hash_block() -> None:
    # Every checkpoint shorter than one hash block shares a lookup bucket.
    index = CheckpointIndex()
    query_tail = tuple(range(1, 301))
    for length in range(1, 300, 7):
        publish(index, manifest(f"prefix-{length}", query_tail[:length]))
        decoy = query_tail[: length - 1] + (10_000 + length,)
        publish(index, manifest(f"decoy-{length}", decoy))
    found = index.find((replace(manifest().prefix, tail_tokens=query_tail),))
    assert found is not None
    assert found.generation == "prefix-295"
    index.close()


def test_capacity_keeps_recently_found_entries(tmp_path: Path) -> None:
    index = CheckpointIndex(tmp_path / "index.sqlite3", max_entries=3)
    entries = [manifest(f"producer-{tail}", (tail,)) for tail in range(3)]
    for entry in entries:
        publish(index, entry)
    assert index.find((entries[0].prefix,)) == entries[0]
    publish(index, manifest("producer-3", (3,)))
    assert index.find((entries[0].prefix,)) == entries[0]
    assert index.find((entries[1].prefix,)) is None
    assert index.report_status()["published_generations"] == 3
    assert tail_rows(index) == {"producer-0", "producer-2", "producer-3"}
    index.close()


def tail_rows(index: CheckpointIndex) -> set[str]:
    rows = index._db.execute("SELECT generation FROM checkpoint_tails")
    return {generation for (generation,) in rows}


def test_replaced_and_invalidated_generations_leave_no_lookup_rows() -> None:
    index = CheckpointIndex()
    first, second = manifest("producer-a"), manifest("producer-b")
    publish(index, first)
    publish(index, second)  # same exact prefix: replaces the first generation
    assert index.find((first.prefix,)) == second
    assert tail_rows(index) == {"producer-b"}
    index.invalidate(second.generation)
    assert index.find((first.prefix,)) is None
    assert tail_rows(index) == set()
    index.close()


def test_hash_candidates_are_confirmed_against_the_stored_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Every tail hashes alike, so lookups rely on the exact tail comparison.
    monkeypatch.setattr(
        checkpoint_index, "_prefix_hashes", lambda tokens: [7] * len(tokens)
    )
    index = CheckpointIndex()
    kept = manifest("kept", (1, 2))
    for entry in (kept, manifest("decoy-a", (1, 20)), manifest("decoy-b", (12,))):
        publish(index, entry)
    assert index.find((replace(kept.prefix, tail_tokens=(1, 2, 3)),)) == kept
    assert index.find((replace(kept.prefix, tail_tokens=(1, 3)),)) is None
    index.close()


def test_existing_directory_gains_recency_index(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite3"
    index = CheckpointIndex(path)
    entry = manifest()
    publish(index, entry)
    index.close()
    # As written by an older build: no recency index and no tail lookup table,
    # plus a lookup row whose manifest no longer exists.
    with sqlite3.connect(path) as db:
        db.execute("DROP INDEX checkpoints_access_order")
        db.execute("DROP TABLE checkpoint_tails")
        db.execute(
            "CREATE TABLE checkpoint_tails (generation TEXT PRIMARY KEY, "
            "namespace TEXT NOT NULL, start_tokens INTEGER NOT NULL, "
            "prefix_hash BLOB NOT NULL, tail_hash INTEGER NOT NULL)"
        )
        db.execute("INSERT INTO checkpoint_tails VALUES ('gone', 'ns', 0, x'', 1)")
    reopened = CheckpointIndex(path)
    assert reopened.find((entry.prefix,)) == entry
    assert tail_rows(reopened) == {entry.generation}
    reopened.close()
    with sqlite3.connect(path) as db:
        names = {row[0] for row in db.execute("SELECT name FROM sqlite_master")}
    assert {"checkpoints_access_order", "checkpoint_tails_lookup"} <= names
