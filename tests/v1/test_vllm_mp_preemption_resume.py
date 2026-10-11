# SPDX-License-Identifier: Apache-2.0
"""A preempted request reloads its prefix from LMCache when it resumes."""

# Standard
from typing import Any

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module load")

# Third Party
from vllm.v1.request import RequestStatus  # noqa: E402
from vllm.v1.utils import ConstantList  # noqa: E402

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
    LMCacheMPRequestState,
)

CHUNK = 3072


class _Request:
    """Duck-typed text request carrying the fields the connector reads."""

    def __init__(self, request_id: str, num_tokens: int, status: object) -> None:
        self.request_id = request_id
        self.cache_salt = ""
        self.prompt_token_ids = list(range(num_tokens))
        self.all_token_ids = ConstantList(self.prompt_token_ids)
        self.mm_features: list[object] = []
        self.status = status
        self.kv_transfer_params = None
        self.resumable = False


class _SchedulerAdapter:
    """Lookup stub that reports a fixed external hit."""

    lmcache_tokens_per_chunk = CHUNK

    def __init__(self, lookup_tokens: int) -> None:
        self.lookup_tokens = lookup_tokens
        self.submitted_token_counts: list[int] = []

    def maybe_submit_lookup_request(
        self, request_id: str, token_ids: list[int], cache_salt: str
    ) -> None:
        del request_id, cache_salt
        self.submitted_token_counts.append(len(token_ids))

    def check_lookup_result(self, request_id: str) -> int:
        del request_id
        return self.lookup_tokens


def _connector(lookup_tokens: int, has_recurrent_cache: bool) -> LMCacheMPConnector:
    connector: Any = object.__new__(LMCacheMPConnector)
    connector.request_trackers = {}
    connector._has_recurrent_cache = has_recurrent_cache
    connector._prefill_replay_tokens = 0
    connector._hit_alignment_tokens = CHUNK
    connector.scheduler_adapter = _SchedulerAdapter(lookup_tokens)
    connector.lazy_offload = False
    return connector


@pytest.mark.parametrize("has_recurrent_cache", [False, True])
def test_preempted_request_reloads_its_prefix(has_recurrent_cache: bool) -> None:
    """Before, a preempted request reported no external hit and recomputed
    its whole prompt even when LMCache held the prefix."""
    connector = _connector(4 * CHUNK, has_recurrent_cache)
    request = _Request("preempted", 5 * CHUNK + 100, RequestStatus.PREEMPTED)

    assert connector.get_num_new_matched_tokens(request, 0) == (4 * CHUNK, True)
    tracker = connector.request_trackers[request.request_id]
    assert tracker.num_lmcache_hit_tokens == 4 * CHUNK
    assert tracker.num_stored_tokens == 4 * CHUNK


def test_preempted_request_loads_only_beyond_its_local_prefix() -> None:
    connector = _connector(4 * CHUNK, has_recurrent_cache=False)
    request = _Request("preempted-local", 5 * CHUNK, RequestStatus.PREEMPTED)

    assert connector.get_num_new_matched_tokens(request, CHUNK) == (3 * CHUNK, True)


def test_preemption_restarts_the_tracker_of_an_admitted_request() -> None:
    """The tracker from before the preemption must not carry its stored range
    into the resumed request's accounting."""
    connector = _connector(2 * CHUNK, has_recurrent_cache=False)
    request = _Request("admitted", 3 * CHUNK, RequestStatus.RUNNING)
    connector.get_num_new_matched_tokens(request, 0)
    before = connector.request_trackers[request.request_id]
    before.state = LMCacheMPRequestState.READY
    before.num_stored_tokens = 3 * CHUNK

    request.status = RequestStatus.PREEMPTED
    connector.get_num_new_matched_tokens(request, 0)

    after = connector.request_trackers[request.request_id]
    assert after is not before
    assert after.num_stored_tokens == 2 * CHUNK


def test_repeated_polling_does_not_double_count_the_stored_range() -> None:
    """The scheduler polls a completed lookup again while it waits for free
    blocks; the stored range must stay the lookup's hit."""
    connector = _connector(2 * CHUNK, has_recurrent_cache=False)
    request = _Request("polled", 3 * CHUNK, RequestStatus.WAITING)

    connector.get_num_new_matched_tokens(request, 0)
    connector.get_num_new_matched_tokens(request, 0)

    tracker = connector.request_trackers[request.request_id]
    assert tracker.num_stored_tokens == 2 * CHUNK
