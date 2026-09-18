# SPDX-License-Identifier: Apache-2.0
"""Exercise multimodal identity through lookup and the actual wire hasher."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

pytest.importorskip("vllm", reason="The MP connector requires vLLM")

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPRequestTracker,
)
from lmcache.v1.multiprocess.token_hasher import TokenHasher  # noqa: E402


def _request(identifier: str) -> SimpleNamespace:
    return SimpleNamespace(
        request_id="image-key-contract",
        cache_salt="identity-test",
        resumable=False,
        prompt_token_ids=[10, 11, 12, 13] + [99] * 4 + [20, 21, 22, 23],
        all_token_ids=[10, 11, 12, 13] + [99] * 4 + [20, 21, 22, 23],
        mm_features=[
            SimpleNamespace(
                identifier=identifier,
                mm_position=SimpleNamespace(offset=4, length=4),
            )
        ],
    )


def test_identifiers_sharing_low_bits_have_distinct_wire_keys() -> None:
    """Keep the pre-image prefix shared without aliasing image-dependent KV."""
    hasher = TokenHasher(chunk_size=4, hash_algorithm="blake3")
    first = LMCacheMPRequestTracker(_request("0x1234")).get_token_ids()
    second = LMCacheMPRequestTracker(_request("0x11234")).get_token_ids()
    repeated = LMCacheMPRequestTracker(_request("0x1234")).get_token_ids()
    first_keys = hasher.compute_chunk_hashes(first)
    second_keys = hasher.compute_chunk_hashes(second)
    assert first_keys == hasher.compute_chunk_hashes(repeated)
    assert first_keys[0] == second_keys[0]
    assert first_keys[1] != second_keys[1]
    assert first_keys[2] != second_keys[2]


@pytest.mark.parametrize("recurrent", [False, True])
def test_eager_lookup_uses_adjusted_identity_before_boundary_truncation(
    recurrent: bool,
) -> None:
    request = _request("image-with-content-hash")
    tracker = LMCacheMPRequestTracker(request)
    adapter = SimpleNamespace(
        lmcache_tokens_per_chunk=4,
        maybe_submit_lookup_request=MagicMock(),
    )
    # Isolate the public lifecycle hook from service/device initialization.
    connector = SimpleNamespace(
        role=KVConnectorRole.SCHEDULER,
        _eager_prefetch=True,
        _has_recurrent_cache=recurrent,
        _get_or_create_request_tracker=lambda value: tracker,
        scheduler_adapter=adapter,
    )
    LMCacheMPConnector.on_new_request(connector, request)
    expected = tracker.get_token_ids()[:8] if recurrent else tracker.get_token_ids()
    adapter.maybe_submit_lookup_request.assert_called_once_with(
        request.request_id, token_ids=expected, cache_salt=tracker.cache_salt
    )
