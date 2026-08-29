# SPDX-License-Identifier: Apache-2.0
"""Fail-closed mixed-prefix tests for the vLLM MP connector."""

# Standard
from typing import Any

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm.v1.utils import ConstantList  # noqa: E402

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
    LMCacheMPRequestState,
)


class _Request:
    """Duck-typed text request carrying the fields the connector reads."""

    def __init__(self, request_id: str, num_tokens: int) -> None:
        self.request_id = request_id
        self.cache_salt = ""
        self.prompt_token_ids = list(range(num_tokens))
        self.all_token_ids = ConstantList(self.prompt_token_ids)
        self.mm_features: list[object] = []
        self.status = object()


class _SchedulerAdapter:
    lmcache_tokens_per_chunk = 3072

    def __init__(self, lookup_tokens: int) -> None:
        self.lookup_tokens = lookup_tokens
        self.cleaned_request_ids: list[str] = []
        self.freed_ranges: list[tuple[int, int, str]] = []

    def maybe_submit_lookup_request(
        self,
        request_id: str,
        token_ids: list[int],
        cache_salt: str,
    ) -> None:
        del request_id, token_ids, cache_salt

    def check_lookup_result(self, request_id: str) -> int:
        del request_id
        return self.lookup_tokens

    def cleanup_lookup_result(self, request_id: str) -> None:
        self.cleaned_request_ids.append(request_id)

    def free_lookup_locks(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        cache_salt: str,
    ) -> None:
        del token_ids, cache_salt
        self.freed_ranges.append((start, end, request_id))


class _NoBlocks:
    @staticmethod
    def get_block_ids() -> tuple[list[int], ...]:
        return ()


def _scheduler_connector(
    *, has_recurrent_cache: bool, lookup_tokens: int
) -> LMCacheMPConnector:
    connector: Any = object.__new__(LMCacheMPConnector)
    connector.request_trackers = {}
    connector._has_recurrent_cache = has_recurrent_cache
    connector._hit_alignment_tokens = 3072
    connector.scheduler_adapter = _SchedulerAdapter(lookup_tokens)
    return connector


def test_recurrent_mixed_local_and_external_prefix_recomputes_tail() -> None:
    """A deeper external tail must not be spliced onto local recurrent state."""
    connector = _scheduler_connector(
        has_recurrent_cache=True,
        lookup_tokens=46080,
    )
    request = _Request("mixed-recurrent", 47963)

    matched = connector.get_num_new_matched_tokens(
        request,
        num_computed_tokens=36864,
    )

    assert matched == (0, False)
    tracker = connector.request_trackers[request.request_id]
    assert tracker.num_vllm_hit_tokens == 36864
    assert tracker.num_lmcache_hit_tokens == 46080

    connector.update_state_after_alloc(request, _NoBlocks(), 0)

    assert tracker.state == LMCacheMPRequestState.READY
    adapter = connector.scheduler_adapter
    assert isinstance(adapter, _SchedulerAdapter)
    assert adapter.cleaned_request_ids == [request.request_id]
    assert adapter.freed_ranges == [(0, 46080, request.request_id)]


@pytest.mark.parametrize(
    ("has_recurrent_cache", "num_computed_tokens", "expected"),
    [
        (True, 0, (46080, True)),
        (False, 36864, (9216, True)),
    ],
)
def test_mixed_recurrent_guard_preserves_qualified_retrievals(
    has_recurrent_cache: bool,
    num_computed_tokens: int,
    expected: tuple[int, bool],
) -> None:
    """Full external resume and non-recurrent mixed retrieval remain enabled."""
    connector = _scheduler_connector(
        has_recurrent_cache=has_recurrent_cache,
        lookup_tokens=46080,
    )
    request = _Request("qualified-retrieve", 47963)

    matched = connector.get_num_new_matched_tokens(
        request,
        num_computed_tokens=num_computed_tokens,
    )

    assert matched == expected
    tracker = connector.request_trackers[request.request_id]
    assert tracker.needs_retrieve() is True
