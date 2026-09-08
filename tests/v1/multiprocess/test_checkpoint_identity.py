# SPDX-License-Identifier: Apache-2.0
"""Semantic checkpoint keys authenticate content, layout and exact shared tokens."""

# Standard
from dataclasses import replace
from typing import cast
import json
import uuid

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.checkpoint_identity import (
    CheckpointTokenRoots,
    checkpoint_generation,
    checkpoint_namespace,
    checkpoint_page_content_key,
)
from lmcache.v1.multiprocess.checkpoint_index import CheckpointIndex, CheckpointManifest


def test_namespace_separates_every_execution_discriminator() -> None:
    identity: dict[str, object] = dict(
        target_revision="target-content-digest",
        draft_revision="draft-content-digest",
        source_revision="source-tree-digest",
        layout={"dtype": "fp8", "page_bytes": 1024},
        parallel={"tp": 4, "dcp": 4},
    )
    baseline = checkpoint_namespace(identity, "tenant-a")
    assert baseline == checkpoint_namespace(
        json.loads(json.dumps(identity)), "tenant-a"
    )
    assert baseline != checkpoint_namespace(identity, "tenant-b")
    for field in identity:
        assert baseline != checkpoint_namespace(
            identity | {field: "different"}, "tenant-a"
        )
        with pytest.raises(ValueError, match="immutable"):
            checkpoint_namespace(
                {key: value for key, value in identity.items() if key != field},
                "tenant-a",
            )


def test_token_roots_match_arbitrary_boundaries_without_quadratic_tails() -> None:
    tokens = list(range(20000))
    roots = CheckpointTokenRoots.build("authenticated", tokens)
    assert sum(len(root.tail_tokens) for root in roots.roots) == len(tokens)
    directory = CheckpointIndex()
    try:
        for boundary in (17, 4096, 8193, 17408):
            manifest = CheckpointManifest(
                uuid.uuid4().hex, roots.prefix(boundary), 1, b'{"schema_version":1}'
            )
            assert directory.begin(manifest)
            assert directory.acknowledge(manifest.generation, 0)
        for shared, expected in ((18, 17), (4096, 4096), (8960, 8193), (18000, 17408)):
            sibling = CheckpointTokenRoots.build(
                "authenticated", tokens[:shared] + [777777] * 3
            )
            found = directory.find(sibling.roots)
            assert found is not None and found.prefix.num_tokens == expected
        changed = CheckpointTokenRoots.build("authenticated", [123456] + tokens[1:])
        assert directory.find(changed.roots) is None
        assert (
            directory.find(
                tuple(replace(root, namespace="other") for root in roots.roots)
            )
            is None
        )
    finally:
        directory.close()


@pytest.mark.parametrize("tokens", [[-1], [2**31], [True], [1.5]])
def test_token_root_rejects_non_token_wire_values(tokens: list[object]) -> None:
    with pytest.raises(ValueError, match="wire limits"):
        CheckpointTokenRoots.build("authenticated", cast(list[int], tokens))


def test_generation_is_content_stable_and_manifest_sensitive() -> None:
    prefix = CheckpointTokenRoots.build("authenticated", range(5000)).prefix(4500)
    payload = b'{"schema_version":1,"layout":"a"}'
    generation = checkpoint_generation(prefix, payload)
    assert generation == checkpoint_generation(prefix, payload)
    assert generation != checkpoint_generation(prefix, payload + b" ")
    assert generation != checkpoint_generation(
        replace(prefix, tail_tokens=prefix.tail_tokens[:-1]), payload
    )
    assert len(generation) <= 128


def test_page_content_key_is_prefix_stable_and_role_separated() -> None:
    roots = CheckpointTokenRoots.build("authenticated", range(9000))
    prefix = roots.prefix(8192)
    key = checkpoint_page_content_key(prefix, "attention:0")
    assert key == checkpoint_page_content_key(prefix, "attention:0")
    assert key != checkpoint_page_content_key(roots.prefix(8193), "attention:0")
    assert key != checkpoint_page_content_key(prefix, "recurrent:0")
    assert len(key) == 64


def test_token_roots_content_key_is_stable_at_full_and_partial_chunks() -> None:
    roots = CheckpointTokenRoots.build("authenticated", range(9000))
    for boundary in (4096, 4097, 8192, 9000):
        key = roots.content_key(boundary, "attention:0")
        assert key == roots.content_key(boundary, "attention:0")
        assert key != roots.content_key(boundary, "attention:1")
        assert len(key) == 64
    changed = CheckpointTokenRoots.build("authenticated", [*range(8999), 42])
    assert roots.content_key(8192, "attention:0") == changed.content_key(
        8192, "attention:0"
    )
    assert roots.content_key(9000, "attention:0") != changed.content_key(
        9000, "attention:0"
    )
    assert roots.content_keys(((9000, "attention:0"), (9000, "auxiliary"))) == (
        roots.content_key(9000, "attention:0"),
        roots.content_key(9000, "auxiliary"),
    )
