# SPDX-License-Identifier: Apache-2.0
"""Unit tests for multimodal-aware LMCache keys in the MP connector.

vLLM tokenizes every image into the same placeholder token ids, so two
images of equal size at the same prompt position have identical token ids.
The connector keys every lookup, retrieve, store and lock release through
``LMCacheMPRequestTracker.all_token_ids``, a view that replaces placeholder
positions with tokens derived from the item's content identifier; these
tests pin the view's semantics and its use by the emitted metadata.
"""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (
    MM_KEY_TOKEN_FLAG,
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
    MultimodalKeyTokenIds,
    multimodal_key_tokens,
)

CHUNK_TOKENS = 64


def _feature(identifier: str, offset: int, length: int) -> SimpleNamespace:
    return SimpleNamespace(
        identifier=identifier,
        mm_position=SimpleNamespace(offset=offset, length=length),
    )


def _request(
    num_tokens: int,
    features: list[SimpleNamespace] | None = None,
    cache_salt: str = "",
) -> SimpleNamespace:
    request = SimpleNamespace(
        request_id="req-1",
        cache_salt=cache_salt,
        all_token_ids=list(range(num_tokens)),
    )
    if features is not None:
        request.mm_features = features
    return request


def test_key_tokens_are_flagged_and_cover_the_identifier() -> None:
    tokens = multimodal_key_tokens("a" * 64, 20)

    assert len(tokens) == 20
    assert all(token & MM_KEY_TOKEN_FLAG for token in tokens)
    assert all(token < (1 << 41) for token in tokens)
    # Eight 32-bit words cover the 256-bit digest, then repeat.
    assert len(set(tokens[:8])) == 8
    assert tokens[8:16] == tokens[:8]
    assert tokens[16:20] == tokens[:4]
    assert multimodal_key_tokens("a" * 64, 0) == []


def test_key_tokens_follow_the_identifier() -> None:
    assert multimodal_key_tokens("blue", 8) == multimodal_key_tokens("blue", 8)
    assert multimodal_key_tokens("blue", 8) != multimodal_key_tokens("green", 8)
    assert multimodal_key_tokens("blue", 3) == multimodal_key_tokens("blue", 8)[:3]


def test_view_without_items_is_a_pass_through() -> None:
    tokens = list(range(10))
    view = MultimodalKeyTokenIds(tokens)

    assert not view.has_multimodal_items
    assert list(view) == tokens
    assert view[3] == 3
    assert view[-1] == 9
    assert view[2:5] == [2, 3, 4]
    assert view[::3] == [0, 3, 6, 9]


def test_view_substitutes_placeholder_positions_only() -> None:
    tokens = list(range(20))
    view = MultimodalKeyTokenIds(
        tokens,
        ["img-a", "img-b"],
        [SimpleNamespace(offset=2, length=4), SimpleNamespace(offset=10, length=3)],
    )
    expected = list(tokens)
    expected[2:6] = multimodal_key_tokens("img-a", 4)
    expected[10:13] = multimodal_key_tokens("img-b", 3)

    assert view.has_multimodal_items
    assert len(view) == 20
    assert list(view) == expected
    assert [view[i] for i in range(20)] == expected
    assert view[-16] == expected[4]
    # Slices that cut through a placeholder keep the item's own token stream.
    assert view[4:12] == expected[4:12]
    assert view[0:2] == [0, 1]
    assert view[6:10] == [6, 7, 8, 9]
    assert view[::2] == expected[::2]


def test_view_follows_the_growing_token_list() -> None:
    tokens = list(range(8))
    view = MultimodalKeyTokenIds(tokens, ["img"], [SimpleNamespace(offset=1, length=2)])
    assert len(view) == 8

    tokens.extend([100, 101])

    assert len(view) == 10
    assert list(view)[8:] == [100, 101]
    assert list(view)[1:3] == multimodal_key_tokens("img", 2)


def test_view_clips_a_placeholder_past_the_token_list() -> None:
    tokens = list(range(4))
    view = MultimodalKeyTokenIds(tokens, ["img"], [SimpleNamespace(offset=2, length=6)])

    assert list(view) == [0, 1, *multimodal_key_tokens("img", 6)[:2]]


def test_tracker_keys_differ_only_where_the_image_content_differs() -> None:
    image_offset, image_length = 8, 24
    blue = LMCacheMPRequestTracker(
        _request(CHUNK_TOKENS, [_feature("blue", image_offset, image_length)])
    )
    blue_again = LMCacheMPRequestTracker(
        _request(CHUNK_TOKENS, [_feature("blue", image_offset, image_length)])
    )
    green = LMCacheMPRequestTracker(
        _request(CHUNK_TOKENS, [_feature("green", image_offset, image_length)])
    )
    text_only = LMCacheMPRequestTracker(_request(CHUNK_TOKENS))

    blue_ids = list(blue.all_token_ids)
    green_ids = list(green.all_token_ids)
    text_ids = list(text_only.all_token_ids)

    assert blue_ids == list(blue_again.all_token_ids)
    assert blue_ids != green_ids
    # Prefix before the image is shared; the placeholder range is not.
    assert (
        blue_ids[:image_offset] == green_ids[:image_offset] == text_ids[:image_offset]
    )
    assert (
        blue_ids[image_offset + image_length :]
        == text_ids[image_offset + image_length :]
    )
    assert (
        blue_ids[image_offset : image_offset + image_length]
        != text_ids[image_offset : image_offset + image_length]
    )
    assert all(
        token & MM_KEY_TOKEN_FLAG
        for token in blue_ids[image_offset : image_offset + image_length]
    )


def test_tracker_accepts_legacy_mm_fields_and_absent_items() -> None:
    legacy = SimpleNamespace(
        request_id="req-legacy",
        cache_salt="",
        all_token_ids=list(range(16)),
        mm_hashes=["img"],
        mm_positions=[SimpleNamespace(offset=4, length=4)],
    )
    tracker = LMCacheMPRequestTracker(legacy)
    assert list(tracker.all_token_ids)[4:8] == multimodal_key_tokens("img", 4)

    plain = LMCacheMPRequestTracker(_request(16))
    assert list(plain.all_token_ids) == list(range(16))


@pytest.mark.parametrize("direction", ["STORE", "RETRIEVE"])
def test_metadata_ops_carry_the_keyed_tokens(direction: str) -> None:
    tracker = LMCacheMPRequestTracker(
        _request(CHUNK_TOKENS, [_feature("img", 0, CHUNK_TOKENS)])
    )
    tracker.allocated_block_ids = {0: [1, 2, 3, 4]}
    keyed = list(tracker.all_token_ids)
    assert keyed == multimodal_key_tokens("img", CHUNK_TOKENS)

    if direction == "STORE":
        tracker.state = LMCacheMPRequestState.READY
        tracker.num_scheduled_tokens = CHUNK_TOKENS
        metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
            tracker, CHUNK_TOKENS, [16]
        )
    else:
        tracker.num_lmcache_hit_tokens = CHUNK_TOKENS
        tracker.num_vllm_hit_tokens = 0
        tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD
        metadata = LMCacheMPRequestMetadata.GetRetrieveMetadata(
            tracker, CHUNK_TOKENS, [16]
        )

    assert metadata is not None
    assert metadata.direction == direction
    assert metadata.op.token_ids == keyed
