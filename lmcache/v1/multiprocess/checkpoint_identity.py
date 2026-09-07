# SPDX-License-Identifier: Apache-2.0
"""Deterministic, restart-safe identities for recurrent checkpoint lookup."""

# Standard
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import sys

# First Party
from lmcache.v1.multiprocess.checkpoint_index import CheckpointPrefix


def checkpoint_namespace(identity: dict[str, object], cache_salt: str) -> str:
    """Authenticate the execution contract and a request's cache isolation salt.

    Args:
        identity: Immutable weights, source and layout contract. Required fields
            are target_revision, draft_revision (empty without speculation),
            source_revision, layout, and parallel. Revision values identify
            content, not a mutable branch name or a local directory path.
        cache_salt: Request isolation salt; an empty string is a distinct value.

    Returns:
        Versioned SHA256 namespace, stable across processes and pool capacities.

    Raises:
        ValueError: If any required identity field is missing or empty.
        TypeError: If the identity cannot be represented as canonical JSON.
    """
    required = {
        "target_revision",
        "draft_revision",
        "source_revision",
        "layout",
        "parallel",
    }
    if not required.issubset(identity) or any(
        not identity[key] for key in required - {"draft_revision"}
    ):
        raise ValueError(
            "Checkpoint identity requires immutable weights, source and layout"
        )
    encoded = json.dumps(
        {"schema_version": 1, "identity": identity, "cache_salt": cache_salt},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return "recurrent-sha256-v1:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CheckpointTokenRoots:
    """Restart-stable hash roots with exact token tails for longest-prefix lookup.

    Unlike process-seeded GPU cache hashes, this format uses an explicit SHA256
    domain and unsigned little-endian token IDs. A root covers at most one
    chunk's tail; all query tails together contain each token only once.
    """

    namespace: str
    roots: tuple[CheckpointPrefix, ...]
    chunk_tokens: int

    @classmethod
    def build(
        cls, namespace: str, token_ids: Sequence[int], *, chunk_tokens: int = 4096
    ) -> "CheckpointTokenRoots":
        """Build linear-size queries for an authenticated token sequence.

        Args:
            namespace: Result of checkpoint_namespace for this request.
            token_ids: Nonnegative token IDs below 2**31; empty inputs yield no roots.
            chunk_tokens: Hash unit, in 1..65536; part of the wire format.

        Returns:
            Roots matching checkpoints at any exact token offset in the sequence.

        Raises:
            ValueError: If the chunk size or any token is outside the wire domain.
        """
        if not 1 <= chunk_tokens <= 65536:
            raise ValueError("Checkpoint hash unit must be in 1..65536")
        chain = hashlib.sha256(
            b"lmcache-recurrent-token-roots-v1\0" + chunk_tokens.to_bytes(4, "little")
        ).digest()
        roots = []
        for start in range(0, len(token_ids), chunk_tokens):
            tail = tuple(token_ids[start : start + chunk_tokens])
            root = CheckpointPrefix(namespace, start, chain, tail)
            roots.append(root)
            packed = array("I", tail)
            if packed.itemsize != 4:
                raise ValueError("Checkpoint tokens require 32-bit unsigned integers")
            if sys.byteorder != "little":
                packed.byteswap()
            chain = hashlib.sha256(chain + packed.tobytes()).digest()
        return cls(namespace, tuple(roots), chunk_tokens)

    def prefix(self, num_tokens: int) -> CheckpointPrefix:
        """Return the exact stored key for a nonempty covered token prefix.

        Args:
            num_tokens: Exclusive token boundary within the sequence.

        Returns:
            Root plus a nonempty exact tail, including chunk-aligned endpoints.

        Raises:
            ValueError: If the requested boundary is empty or exceeds the sequence.
        """
        if num_tokens <= 0 or not self.roots or num_tokens > self.roots[-1].num_tokens:
            raise ValueError("Checkpoint boundary is outside the token sequence")
        root = self.roots[(num_tokens - 1) // self.chunk_tokens]
        return CheckpointPrefix(
            self.namespace,
            root.start_tokens,
            root.prefix_hash,
            root.tail_tokens[: num_tokens - root.start_tokens],
        )
