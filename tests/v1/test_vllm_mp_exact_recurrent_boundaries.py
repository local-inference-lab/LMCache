# SPDX-License-Identifier: Apache-2.0
"""Exact recurrent-boundary tests for the vLLM MP connector."""

# Standard
from types import SimpleNamespace
from typing import cast

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module load")

# Third Party
from vllm.v1.core.sched.output import SchedulerOutput  # noqa: E402

# First Party
import lmcache.integration.vllm.lmcache_mp_connector as connector_mod  # noqa: E402
import lmcache.integration.vllm.lmcache_mp_metadata as metadata_mod  # noqa: E402
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
    _is_mamba_group_spec,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPConnectorMetadata,
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
)

GROUP_TOKENS_PER_BLOCK = [512, 512, 512, 512, 2048, 512]
CHUNK_TOKENS = 4096
RECURRENT_GROUP_IDS = {0, 1, 2, 3}
ATTENTION_GROUP_ID = 4
DFLASH_GROUP_ID = 5


class MambaSpec:
    """Compatibility stand-in identified by vLLM's recurrent spec name."""


class AttentionSpec:
    """Non-recurrent compatibility stand-in."""


class UniformTypeKVCacheSpecs:
    """Compatibility stand-in for vLLM's per-layer group wrapper."""

    def __init__(self, kv_cache_specs: dict[str, object]) -> None:
        self.kv_cache_specs = kv_cache_specs


class RecordingBlockPool:
    """Minimal block-pool contract for scheduler reference tests."""

    class Blocks:
        def __getitem__(self, block_id: int) -> SimpleNamespace:
            return SimpleNamespace(block_id=block_id)

    def __init__(self) -> None:
        self.blocks = self.Blocks()
        self.touched: list[list[int]] = []
        self.freed: list[list[int]] = []

    def touch(self, blocks: list[SimpleNamespace]) -> None:
        self.touched.append([block.block_id for block in blocks])

    def free_blocks(self, blocks: object) -> None:
        self.freed.append([block.block_id for block in blocks])


def _bind_recording_block_pool(connector: LMCacheMPConnector) -> RecordingBlockPool:
    pool = RecordingBlockPool()
    connector._gpu_block_pool = pool  # type: ignore[assignment]
    connector._next_store_job_id = 0
    connector._pinned_store_jobs = {}
    connector._num_workers = 4
    return pool


def test_live_six_group_detection_finds_only_recurrent_groups() -> None:
    layer_counts = [9, 9, 8, 8]
    groups = [
        SimpleNamespace(
            kv_cache_spec=UniformTypeKVCacheSpecs(
                {
                    f"model.layers.{layer_id}": MambaSpec()
                    for layer_id in range(layer_count)
                }
            )
        )
        for layer_count in layer_counts
    ]
    groups.extend(
        [
            SimpleNamespace(
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    {
                        f"model.layers.{layer_id}": AttentionSpec()
                        for layer_id in range(11)
                    }
                )
            ),
            SimpleNamespace(
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    {
                        f"draft.layers.{layer_id}": AttentionSpec()
                        for layer_id in range(5)
                    }
                )
            ),
        ]
    )

    detected_group_ids = {
        group_id
        for group_id, group in enumerate(groups)
        if _is_mamba_group_spec(group.kv_cache_spec)
    }

    assert detected_group_ids == RECURRENT_GROUP_IDS


@pytest.mark.parametrize(
    ("has_recurrent_cache", "is_kv_producer", "expected"),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_local_prefill_serialization_requires_recurrent_producer(
    has_recurrent_cache: bool,
    is_kv_producer: bool,
    expected: bool,
) -> None:
    connector = cast(LMCacheMPConnector, object.__new__(LMCacheMPConnector))
    connector._has_recurrent_cache = has_recurrent_cache
    connector._kv_transfer_config = SimpleNamespace(is_kv_producer=is_kv_producer)

    assert connector.requires_local_prefill_serialization is expected


def _store_tracker(num_chunks: int = 2) -> LMCacheMPRequestTracker:
    """Build a tracker with complete positional tables for DFlash geometry."""
    num_tokens = num_chunks * CHUNK_TOKENS
    token_ids = list(range(num_tokens))
    request = SimpleNamespace(
        request_id="dflash-store",
        cache_salt="",
        prompt_token_ids=token_ids,
        all_token_ids=token_ids,
        mm_features=[],
    )
    tracker = LMCacheMPRequestTracker(request)
    tracker.allocated_block_ids = {
        group_id: list(
            range(
                1000 * (group_id + 1),
                1000 * (group_id + 1) + num_tokens // group_tokens_per_block,
            )
        )
        for group_id, group_tokens_per_block in enumerate(GROUP_TOKENS_PER_BLOCK)
    }
    tracker.num_scheduled_tokens = num_tokens
    return tracker


def _current_block_table(
    tracker: LMCacheMPRequestTracker,
) -> tuple[list[int], ...]:
    """Return an isolated scheduler-style snapshot of one tracker table."""
    return tuple(
        list(tracker.allocated_block_ids[group_id])
        for group_id in range(len(GROUP_TOKENS_PER_BLOCK))
    )


def test_tracker_initializes_exact_recurrent_boundary_map() -> None:
    tracker = _store_tracker()

    assert tracker.exact_mamba_boundary_blocks == {}


def test_store_uses_exact_boundaries_for_six_group_dflash_geometry() -> None:
    tracker = _store_tracker()
    tracker.exact_mamba_boundary_blocks = {
        group_id: {
            CHUNK_TOKENS: 100 + group_id,
            2 * CHUNK_TOKENS: 200 + group_id,
        }
        for group_id in RECURRENT_GROUP_IDS
    }
    positional_attention_ids = list(tracker.allocated_block_ids[ATTENTION_GROUP_ID])
    positional_dflash_ids = list(tracker.allocated_block_ids[DFLASH_GROUP_ID])

    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        CHUNK_TOKENS,
        GROUP_TOKENS_PER_BLOCK,
        RECURRENT_GROUP_IDS,
    )

    assert metadata is not None
    for group_id in RECURRENT_GROUP_IDS:
        assert metadata.op.block_ids[group_id] == (
            [0] * 7 + [100 + group_id] + [0] * 7 + [200 + group_id]
        )
    assert metadata.op.block_ids[ATTENTION_GROUP_ID] == positional_attention_ids
    assert metadata.op.block_ids[DFLASH_GROUP_ID] == positional_dflash_ids
    assert len(metadata.op.block_ids[ATTENTION_GROUP_ID]) == 4
    assert len(metadata.op.block_ids[DFLASH_GROUP_ID]) == 16
    assert tracker.num_stored_tokens == 2 * CHUNK_TOKENS


@pytest.mark.parametrize(
    "group_id",
    list(RECURRENT_GROUP_IDS),
)
def test_store_fails_closed_when_any_exact_boundary_is_missing(
    group_id: int,
) -> None:
    tracker = _store_tracker()
    tracker.exact_mamba_boundary_blocks = {
        recurrent_group_id: {
            CHUNK_TOKENS: 100 + recurrent_group_id,
            2 * CHUNK_TOKENS: 200 + recurrent_group_id,
        }
        for recurrent_group_id in RECURRENT_GROUP_IDS
    }
    del tracker.exact_mamba_boundary_blocks[group_id][CHUNK_TOKENS]
    positional_tables = {
        group_id: list(block_ids)
        for group_id, block_ids in tracker.allocated_block_ids.items()
    }

    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        CHUNK_TOKENS,
        GROUP_TOKENS_PER_BLOCK,
        RECURRENT_GROUP_IDS,
    )

    assert metadata is None
    assert tracker.num_stored_tokens == 0
    assert tracker.allocated_block_ids == positional_tables


def test_exact_store_remains_enabled_after_mixed_recurrent_suppression() -> None:
    tracker = _store_tracker(num_chunks=1)
    tracker.suppress_mixed_recurrent_retrieve = True
    tracker.exact_mamba_boundary_blocks = {
        group_id: {CHUNK_TOKENS: 100 + group_id} for group_id in RECURRENT_GROUP_IDS
    }

    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        CHUNK_TOKENS,
        GROUP_TOKENS_PER_BLOCK,
        RECURRENT_GROUP_IDS,
    )

    assert metadata is not None
    for group_id in RECURRENT_GROUP_IDS:
        assert metadata.op.block_ids[group_id] == [0] * 7 + [100 + group_id]


def test_connector_ingests_only_exact_handoffs_for_recurrent_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = _store_tracker(num_chunks=1)
    tracker.state = LMCacheMPRequestState.READY
    info_messages: list[str] = []

    def spy_info(message: str, *args: object) -> None:
        info_messages.append(message % args)

    monkeypatch.setattr(connector_mod.logger, "info", spy_info)
    connector = cast(LMCacheMPConnector, object.__new__(LMCacheMPConnector))
    connector.request_trackers = {"known": tracker}
    connector._mamba_group_ids = RECURRENT_GROUP_IDS
    connector._group_tokens_per_block = GROUP_TOKENS_PER_BLOCK
    connector.lazy_offload = False
    _bind_recording_block_pool(connector)
    connector.scheduler_adapter = SimpleNamespace(  # type: ignore[assignment]
        lmcache_tokens_per_chunk=CHUNK_TOKENS,
        report_block_allocations=lambda records: None,
    )
    scheduler_output = SimpleNamespace(
        kv_connector_block_state=SimpleNamespace(
            block_ids={"known": _current_block_table(tracker)},
            boundary_state_offloads={
                "unknown": [(0, 71, CHUNK_TOKENS)],
                "known": [
                    (0, 91, CHUNK_TOKENS),
                    (1, 92, CHUNK_TOKENS),
                    (2, 93, CHUNK_TOKENS),
                    (3, 94, CHUNK_TOKENS),
                    (ATTENTION_GROUP_ID, 95, CHUNK_TOKENS),
                    (DFLASH_GROUP_ID, 96, CHUNK_TOKENS),
                    (0, 0, 2 * CHUNK_TOKENS),
                    (1, -1, 2 * CHUNK_TOKENS),
                ],
            },
        ),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["known"],
            new_block_ids=[None],
            resumed_req_ids=[],
        ),
        num_scheduled_tokens={"known": 0},
        total_num_scheduled_tokens=0,
        preempted_req_ids=[],
    )

    connector.build_connector_meta(scheduler_output)

    assert tracker.exact_mamba_boundary_blocks == {
        0: {CHUNK_TOKENS: 91},
        1: {CHUNK_TOKENS: 92},
        2: {CHUNK_TOKENS: 93},
        3: {CHUNK_TOKENS: 94},
    }
    assert info_messages == [
        "Ingested exact recurrent handoffs: request_id=known, "
        "received_handoff_boundaries={0: [4096], 1: [4096], "
        "2: [4096], 3: [4096]}"
    ]


def test_store_uses_authoritative_scheduler_block_table() -> None:
    """A store must not use an append-only block mirror after table mutation."""
    tracker = _store_tracker(num_chunks=1)
    tracker.state = LMCacheMPRequestState.READY
    authoritative = tuple(
        [block_id + 10_000 for block_id in tracker.allocated_block_ids[group_id]]
        for group_id in range(len(GROUP_TOKENS_PER_BLOCK))
    )
    connector = cast(LMCacheMPConnector, object.__new__(LMCacheMPConnector))
    connector.request_trackers = {"store": tracker}
    connector._mamba_group_ids = RECURRENT_GROUP_IDS
    connector._group_tokens_per_block = GROUP_TOKENS_PER_BLOCK
    connector.lazy_offload = False
    _bind_recording_block_pool(connector)
    connector.scheduler_adapter = SimpleNamespace(  # type: ignore[assignment]
        lmcache_tokens_per_chunk=CHUNK_TOKENS,
        report_block_allocations=lambda records: None,
    )
    scheduler_output = SimpleNamespace(
        kv_connector_block_state=SimpleNamespace(
            block_ids={"store": authoritative},
            boundary_state_offloads={
                "store": [
                    (group_id, 20_000 + group_id, CHUNK_TOKENS)
                    for group_id in RECURRENT_GROUP_IDS
                ]
            },
        ),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["store"],
            new_block_ids=[None],
            resumed_req_ids=[],
        ),
        num_scheduled_tokens={"store": 0},
        total_num_scheduled_tokens=0,
        preempted_req_ids=[],
    )

    metadata = connector.build_connector_meta(cast(SchedulerOutput, scheduler_output))

    assert metadata.requests[0].op.block_ids[ATTENTION_GROUP_ID] == list(
        authoritative[ATTENTION_GROUP_ID]
    )
    assert metadata.requests[0].op.block_ids[DFLASH_GROUP_ID] == list(
        authoritative[DFLASH_GROUP_ID]
    )
    for group_id in RECURRENT_GROUP_IDS:
        assert metadata.requests[0].op.block_ids[group_id] == [0] * 7 + [
            20_000 + group_id
        ]


def test_store_diagnostic_logs_are_bounded_without_changing_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = _store_tracker()
    tracker.exact_mamba_boundary_blocks = {
        group_id: {CHUNK_TOKENS: 100 + group_id} for group_id in RECURRENT_GROUP_IDS
    }
    info_messages: list[str] = []
    debug_messages: list[str] = []

    def spy_info(message: str, *args: object) -> None:
        info_messages.append(message % args)

    def spy_debug(message: str, *args: object) -> None:
        debug_messages.append(message % args)

    monkeypatch.setattr(metadata_mod.logger, "info", spy_info)
    monkeypatch.setattr(metadata_mod.logger, "debug", spy_debug)

    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        CHUNK_TOKENS,
        GROUP_TOKENS_PER_BLOCK,
        RECURRENT_GROUP_IDS,
    )

    assert metadata is not None
    assert metadata.op.start == 0
    assert metadata.op.end == CHUNK_TOKENS
    assert tracker.num_stored_tokens == CHUNK_TOKENS
    assert info_messages == [
        "Truncating recurrent store to handoff-ready prefix: "
        "request_id=dflash-store, recurrent_groups=[0, 1, 2, 3], "
        "received_handoff_boundaries={0: [4096], 1: [4096], "
        "2: [4096], 3: [4096]}, "
        "required_store_boundaries=[4096, 8192], store_range=[0, 4096), "
        "reason=later exact recurrent boundary unavailable",
        "Emitting store metadata: request_id=dflash-store, "
        "store_range=[0, 4096), "
        "block_counts_by_group={0: 8, 1: 8, 2: 8, 3: 8, 4: 2, 5: 8}",
    ]

    suppressed_metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        CHUNK_TOKENS,
        GROUP_TOKENS_PER_BLOCK,
        RECURRENT_GROUP_IDS,
    )

    assert suppressed_metadata is None
    assert tracker.num_stored_tokens == CHUNK_TOKENS
    assert info_messages == [
        "Truncating recurrent store to handoff-ready prefix: "
        "request_id=dflash-store, recurrent_groups=[0, 1, 2, 3], "
        "received_handoff_boundaries={0: [4096], 1: [4096], "
        "2: [4096], 3: [4096]}, "
        "required_store_boundaries=[4096, 8192], store_range=[0, 4096), "
        "reason=later exact recurrent boundary unavailable",
        "Emitting store metadata: request_id=dflash-store, "
        "store_range=[0, 4096), "
        "block_counts_by_group={0: 8, 1: 8, 2: 8, 3: 8, 4: 2, 5: 8}",
    ]
    assert debug_messages == [
        "Suppressing recurrent store: request_id=dflash-store, "
        "recurrent_groups=[0, 1, 2, 3], "
        "received_handoff_boundaries={0: [4096], 1: [4096], "
        "2: [4096], 3: [4096]}, required_store_boundaries=[8192], "
        "reason=no contiguous exact recurrent boundary"
    ]


def test_connector_retains_new_request_handoff_before_first_store() -> None:
    tracker = _store_tracker(num_chunks=1)
    tracker.num_scheduled_tokens = 0

    class NewRequestConnector(LMCacheMPConnector):
        def _process_new_requests(
            self,
            scheduler_output: SchedulerOutput,
            metadata: LMCacheMPConnectorMetadata,
        ) -> None:
            self.request_trackers["new-request"] = tracker
            super()._process_new_requests(scheduler_output, metadata)

    connector = cast(NewRequestConnector, object.__new__(NewRequestConnector))
    connector.request_trackers = {}
    connector._mamba_group_ids = RECURRENT_GROUP_IDS
    connector._group_tokens_per_block = GROUP_TOKENS_PER_BLOCK
    connector.lazy_offload = False
    pool = _bind_recording_block_pool(connector)
    connector.scheduler_adapter = SimpleNamespace(  # type: ignore[assignment]
        lmcache_tokens_per_chunk=CHUNK_TOKENS,
        report_block_allocations=lambda records: None,
    )
    scheduler_output = SimpleNamespace(
        kv_connector_block_state=SimpleNamespace(
            block_ids={"new-request": _current_block_table(tracker)},
            boundary_state_offloads={
                "new-request": [
                    (group_id, 100 + group_id, CHUNK_TOKENS)
                    for group_id in RECURRENT_GROUP_IDS
                ]
            },
        ),
        scheduled_new_reqs=[SimpleNamespace(req_id="new-request")],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            new_block_ids=[],
            resumed_req_ids=[],
        ),
        num_scheduled_tokens={"new-request": CHUNK_TOKENS},
        total_num_scheduled_tokens=CHUNK_TOKENS,
        preempted_req_ids=[],
    )

    metadata = connector.build_connector_meta(cast(SchedulerOutput, scheduler_output))

    assert isinstance(metadata, LMCacheMPConnectorMetadata)
    assert tracker.exact_mamba_boundary_blocks == {
        group_id: {CHUNK_TOKENS: 100 + group_id} for group_id in RECURRENT_GROUP_IDS
    }
    assert len(metadata.requests) == 1
    assert metadata.requests[0].direction == "STORE"
    assert metadata.requests[0].store_job_id == 0
    assert len(pool.touched) == 1
    for group_id in RECURRENT_GROUP_IDS:
        assert metadata.requests[0].op.block_ids[group_id] == (
            [0] * 7 + [100 + group_id]
        )


def test_multistep_store_emits_handoff_ready_prefix() -> None:
    tracker = _store_tracker(num_chunks=8)
    tracker.num_scheduled_tokens = 0
    tracker.state = LMCacheMPRequestState.READY
    connector = cast(LMCacheMPConnector, object.__new__(LMCacheMPConnector))
    connector.request_trackers = {"multistep": tracker}
    connector._mamba_group_ids = RECURRENT_GROUP_IDS
    connector._group_tokens_per_block = GROUP_TOKENS_PER_BLOCK
    connector.lazy_offload = False
    _bind_recording_block_pool(connector)
    connector.scheduler_adapter = SimpleNamespace(  # type: ignore[assignment]
        lmcache_tokens_per_chunk=CHUNK_TOKENS,
        report_block_allocations=lambda records: None,
    )

    first_step = SimpleNamespace(
        kv_connector_block_state=SimpleNamespace(
            block_ids={"multistep": _current_block_table(tracker)},
            boundary_state_offloads={},
        ),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["multistep"],
            new_block_ids=[None],
            resumed_req_ids=[],
        ),
        num_scheduled_tokens={"multistep": 8192},
        total_num_scheduled_tokens=8192,
        preempted_req_ids=[],
    )
    first_metadata = connector.build_connector_meta(cast(SchedulerOutput, first_step))

    assert isinstance(first_metadata, LMCacheMPConnectorMetadata)
    assert first_metadata.requests == []
    assert tracker.num_stored_tokens == 0

    for batch_id in range(4):
        batch_start = batch_id * 8192
        num_scheduled_tokens = 8192 if batch_id < 3 else 0
        handoff_step = SimpleNamespace(
            kv_connector_block_state=SimpleNamespace(
                block_ids={"multistep": _current_block_table(tracker)},
                boundary_state_offloads={
                    "multistep": [
                        (
                            group_id,
                            1000 * (batch_id + 1) + group_id,
                            boundary_tokens,
                        )
                        for boundary_tokens in (
                            batch_start + CHUNK_TOKENS,
                            batch_start + 2 * CHUNK_TOKENS,
                        )
                        for group_id in RECURRENT_GROUP_IDS
                    ]
                },
            ),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=["multistep"],
                new_block_ids=[None],
                resumed_req_ids=[],
            ),
            num_scheduled_tokens={"multistep": num_scheduled_tokens},
            total_num_scheduled_tokens=num_scheduled_tokens,
            preempted_req_ids=[],
        )
        metadata = connector.build_connector_meta(cast(SchedulerOutput, handoff_step))

        assert isinstance(metadata, LMCacheMPConnectorMetadata)
        assert len(metadata.requests) == 1
        assert metadata.requests[0].op.start == batch_start
        assert metadata.requests[0].op.end == batch_start + 8192
        assert tracker.num_stored_tokens == batch_start + 8192

    assert tracker.num_scheduled_tokens == 32768
    assert tracker.num_stored_tokens == 32768


def test_store_blocks_remain_referenced_until_every_worker_completes() -> None:
    connector = cast(LMCacheMPConnector, object.__new__(LMCacheMPConnector))
    connector.lazy_offload = False
    pool = _bind_recording_block_pool(connector)
    metadata = LMCacheMPConnectorMetadata()
    metadata.add_request_metadata(
        LMCacheMPRequestMetadata(
            request_id="store",
            direction="STORE",
            op=metadata_mod.LoadStoreOp(
                token_ids=list(range(CHUNK_TOKENS)),
                block_ids=[[0, 71, 71], [72]],
                start=0,
                end=CHUNK_TOKENS,
            ),
        )
    )

    connector._reference_store_blocks(metadata)

    assert metadata.requests[0].store_job_id == 0
    assert pool.touched == [[71, 72]]
    connector.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=metadata_mod.LMCacheMPWorkerMetadata(
                completed_store_jobs={0: 3}
            )
        )
    )
    assert pool.freed == []
    connector.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=metadata_mod.LMCacheMPWorkerMetadata(
                completed_store_jobs={0: 1}
            )
        )
    )
    assert pool.freed == [[72, 71]]
    assert connector.has_pending_push_work() is False
