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
    checkpoint_namespace,
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
