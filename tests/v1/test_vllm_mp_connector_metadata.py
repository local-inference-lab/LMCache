# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``LMCacheMPRequestMetadata.GetRetrieveMetadata``.

The retrieve op must only be emitted when the tracker's allocated block ids
actually cover the retrieve token range in every engine group:
``slice_block_ids_per_group`` validates chunk alignment only, and its plain
Python slice truncates silently, so an uncovered range would otherwise emit
an op with too few block ids.
"""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
    _recurrent_safe_lookup_end,
)

CHUNK_TOKENS = 64


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [
        (0, 0),
        (1, 0),
        (63, 0),
        (64, 0),
        (65, 64),
        (128, 64),
        (129, 128),
    ],
)
def test_recurrent_lookup_excludes_final_prompt_token(
    num_tokens: int, expected: int
) -> None:
    """A recurrent restore ends before the token vLLM recomputes for logits."""
    assert _recurrent_safe_lookup_end(num_tokens, CHUNK_TOKENS) == expected


def test_recurrent_lookup_rejects_non_positive_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_tokens must be positive"):
        _recurrent_safe_lookup_end(128, 0)


def _tracker(
    num_tokens: int,
    lmcache_hit_tokens: int,
    vllm_hit_tokens: int,
    allocated_block_ids: dict[int, list[int]],
) -> LMCacheMPRequestTracker:
    """Build a tracker that is ready for retrieving over *num_tokens*."""
    request = SimpleNamespace(
        request_id="req-1",
        cache_salt="",
        all_token_ids=list(range(num_tokens)),
    )
    tracker = LMCacheMPRequestTracker(request)
    tracker.num_lmcache_hit_tokens = lmcache_hit_tokens
    tracker.num_vllm_hit_tokens = vllm_hit_tokens
    tracker.allocated_block_ids = allocated_block_ids
    tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD
    return tracker


@pytest.mark.parametrize(
    ("group_tokens_per_block", "allocated_block_ids", "expected_block_ids"),
    [
        # Single group, plain geometry: 64 tokens / 16 tokens per block.
        ([16], {0: [0, 1, 2, 3]}, [[0, 1, 2, 3]]),
        # Single group, DCP-scaled: one manager block id covers 64 tokens.
        ([64], {0: [5]}, [[5]]),
        # Hybrid geometries: each group sliced by its own tokens-per-block.
        ([16, 32], {0: [0, 1, 2, 3], 1: [10, 11]}, [[0, 1, 2, 3], [10, 11]]),
        # Extra allocated blocks beyond the range are fine (and not emitted).
        ([16], {0: [0, 1, 2, 3, 4, 5]}, [[0, 1, 2, 3]]),
    ],
)
def test_retrieve_metadata_emitted_when_allocation_covers_range(
    group_tokens_per_block: list[int],
    allocated_block_ids: dict[int, list[int]],
    expected_block_ids: list[list[int]],
) -> None:
    tracker = _tracker(
        num_tokens=CHUNK_TOKENS,
        lmcache_hit_tokens=CHUNK_TOKENS,
        vllm_hit_tokens=0,
        allocated_block_ids=allocated_block_ids,
    )

    metadata = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        tracker, CHUNK_TOKENS, group_tokens_per_block
    )

    assert metadata is not None
    assert metadata.direction == "RETRIEVE"
    assert metadata.op.start == 0
    assert metadata.op.end == CHUNK_TOKENS
    assert metadata.op.block_ids == expected_block_ids


@pytest.mark.parametrize(
    ("group_tokens_per_block", "allocated_block_ids"),
    [
        # One block short in the only group.
        ([16], {0: [0, 1, 2]}),
        # DCP-scaled group with no allocation at all.
        ([64], {0: []}),
        # Group 0 covered, group 1 one block short.
        ([16, 32], {0: [0, 1, 2, 3], 1: [10]}),
        # Group key missing entirely (counts as zero blocks).
        ([16, 16], {0: [0, 1, 2, 3]}),
    ],
)
def test_retrieve_metadata_suppressed_when_allocation_short(
    group_tokens_per_block: list[int],
    allocated_block_ids: dict[int, list[int]],
) -> None:
    """Any engine group whose allocation does not cover the retrieve range
    suppresses the op entirely: the request falls back to recompute instead
    of retrieving over silently truncated block-id lists."""
    tracker = _tracker(
        num_tokens=CHUNK_TOKENS,
        lmcache_hit_tokens=CHUNK_TOKENS,
        vllm_hit_tokens=0,
        allocated_block_ids=allocated_block_ids,
    )

    metadata = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        tracker, CHUNK_TOKENS, group_tokens_per_block
    )

    assert metadata is None


def test_retrieve_metadata_skip_tokens_preserved_on_emitted_op() -> None:
    """vLLM-hit tokens inside the first chunk round the start down and are
    carried as skip_first_n_tokens; coverage is still checked against the
    chunk-aligned end."""
    tracker = _tracker(
        num_tokens=CHUNK_TOKENS,
        lmcache_hit_tokens=CHUNK_TOKENS,
        vllm_hit_tokens=16,
        allocated_block_ids={0: [0, 1, 2, 3]},
    )

    metadata = LMCacheMPRequestMetadata.GetRetrieveMetadata(tracker, CHUNK_TOKENS, [16])

    assert metadata is not None
    assert metadata.op.start == 0
    assert metadata.op.end == CHUNK_TOKENS
    assert metadata.op.skip_first_n_tokens == 16


def test_suppressed_retrieve_is_reported_to_the_worker_as_failed_load() -> None:
    """A tracker in WAITING_FOR_LOAD whose retrieve is suppressed must not be
    left for the scheduler to wait on: the connector metadata carries the
    request id and its allocated blocks so the worker reports a failed load
    and vLLM recomputes."""
    # First Party
    from lmcache.integration.vllm.lmcache_mp_connector import (
        LMCacheMPConnector,
        LMCacheMPConnectorMetadata,
    )

    short = _tracker(
        num_tokens=CHUNK_TOKENS,
        lmcache_hit_tokens=CHUNK_TOKENS,
        vllm_hit_tokens=0,
        allocated_block_ids={0: [7, 8], 1: [30]},  # 2 x 16 < 64 tokens
    )
    covered = _tracker(
        num_tokens=CHUNK_TOKENS,
        lmcache_hit_tokens=CHUNK_TOKENS,
        vllm_hit_tokens=0,
        allocated_block_ids={0: [0, 1, 2, 3], 1: [10, 11]},
    )
    covered.request_id = "req-2"
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector.scheduler_adapter = SimpleNamespace(lmcache_tokens_per_chunk=CHUNK_TOKENS)
    connector._group_tokens_per_block = [16, 32]
    connector.request_trackers = {"req-1": short, "req-2": covered}
    metadata = LMCacheMPConnectorMetadata()

    connector._process_retrieve_requests(metadata)

    assert [m.request_id for m in metadata.requests] == ["req-2"]
    assert metadata.suppressed_retrieves == [("req-1", [7, 8, 30])]
    assert short.state == LMCacheMPRequestState.READY
    assert covered.state == LMCacheMPRequestState.READY


def test_anchored_scheduled_tokens_exclude_rejected_draft_tokens() -> None:
    """The computed-token bound follows the scheduler's num_computed_tokens
    (rewound for rejected drafts) instead of accumulating scheduled counts."""
    request = SimpleNamespace(
        request_id="req-1", cache_salt="", all_token_ids=list(range(100))
    )
    tracker = LMCacheMPRequestTracker(request)
    tracker.num_lmcache_hit_tokens = 64
    tracker.num_vllm_hit_tokens = 0

    # Step 1: 64 hit tokens computed, 4 scheduled (1 + 3 drafts).
    tracker.anchor_num_scheduled_tokens(engine_num_computed_tokens=64, num_new_tokens=4)
    assert tracker.num_scheduled_tokens + 64 == 68
    # Only 2 of the 3 drafts were accepted: the scheduler reports 66.
    tracker.anchor_num_scheduled_tokens(engine_num_computed_tokens=66, num_new_tokens=4)
    assert tracker.num_scheduled_tokens + 64 == 70
    # Accumulation would have claimed 72 computed tokens here.
    tracker.increase_num_scheduled_tokens(0)
    assert tracker.num_scheduled_tokens == 6
    # A count below the hit prefix (fresh tracker after preemption) clamps at 0.
    tracker.anchor_num_scheduled_tokens(engine_num_computed_tokens=0, num_new_tokens=16)
    assert tracker.num_scheduled_tokens == 0


def test_store_window_stops_at_verified_tokens_when_steps_overlap() -> None:
    """With asynchronous scheduling the scheduler's computed-token count
    includes the placeholders of a step still in flight; the store window
    must stop at the tokens the scheduler has appended (verified)."""
    verified = 2 * CHUNK_TOKENS - 2
    request = SimpleNamespace(
        request_id="req-1", cache_salt="", all_token_ids=list(range(verified))
    )
    tracker = LMCacheMPRequestTracker(request)
    tracker.num_lmcache_hit_tokens = 0
    tracker.num_vllm_hit_tokens = 0
    # 16-token blocks covering three chunks: allocation is not the bound.
    tracker.allocated_block_ids = {0: list(range(3 * CHUNK_TOKENS // 16))}
    # The in-flight step counts its 4 placeholders (1 + 3 drafts) as
    # computed: 2 * CHUNK_TOKENS + 2 > verified.
    tracker.anchor_num_scheduled_tokens(
        engine_num_computed_tokens=verified, num_new_tokens=4
    )

    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(tracker, CHUNK_TOKENS, [16])

    assert metadata is not None
    assert metadata.direction == "STORE"
    assert metadata.op.start == 0
    # Only the chunk fully inside the verified prefix is stored; the chunk
    # whose last two tokens are still placeholders is not.
    assert metadata.op.end == CHUNK_TOKENS
    assert len(metadata.op.token_ids) == verified
    assert tracker.num_stored_tokens == CHUNK_TOKENS
