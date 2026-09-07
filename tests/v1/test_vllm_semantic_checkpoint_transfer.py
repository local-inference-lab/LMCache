# SPDX-License-Identifier: Apache-2.0
"""Atomic external checkpoint ownership with the real vLLM allocator and LMCache RPC."""

# Standard
from types import SimpleNamespace
import time

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# Third Party
from vllm.lora.request import LoRARequest  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.utils.hashing import sha256  # noqa: E402
from vllm.v1.core.kv_cache_manager import KVCacheManager  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request  # noqa: E402

# First Party
from lmcache.integration.vllm.checkpoint_scheduler import (  # noqa: E402
    CheckpointEngineTask,
    CheckpointSchedulerBridge,
)
from lmcache.integration.vllm.recurrent_checkpoint_connector import (  # noqa: E402
    LMCacheRecurrentCheckpointConnector,
    RecurrentCheckpointWorkerMetadata,
)
from lmcache.v1.multiprocess.checkpoint_transfer import (  # noqa: E402
    CheckpointTransferJob,
    CheckpointTransferWorker,
)
from tests.v1.multiprocess.test_checkpoint_storage import (  # noqa: E402
    open_checkpoint_rpc,
)

pytestmark = pytest.mark.skipif(
    not hasattr(KVCacheManager, "reserve_external_boundary_checkpoint"),
    reason="vLLM requires the atomic boundary import allocator API",
)


def test_connector_capability_rejects_mutable_revision_names() -> None:
    identity = {
        "target_revision": "a" * 40,
        "source_revision": "b" * 40,
        "draft_revision": "c" * 40,
    }
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(get_from_extra_config=lambda *_: identity),
        speculative_config=SimpleNamespace(method="dflash"),
    )
    assert LMCacheRecurrentCheckpointConnector.supports_request_boundary_checkpoints(
        config
    )
    for key in tuple(identity):
        original = identity[key]
        identity[key] = "main"
        supported = (
            LMCacheRecurrentCheckpointConnector.supports_request_boundary_checkpoints(
                config
            )
        )
        assert not supported
        identity[key] = original


def test_worker_acknowledgements_preserve_rank_identity() -> None:
    left = RecurrentCheckpointWorkerMetadata(
        {0: {"page_bytes": 128}}, {"copy": {0: True}}
    )
    right = RecurrentCheckpointWorkerMetadata(
        {1: {"page_bytes": 128}}, {"copy": {1: False}}
    )
    merged = left.aggregate(right)
    assert merged.results == {"copy": {0: True, 1: False}}
    assert set(merged.layouts) == {0, 1}
    with pytest.raises(ValueError, match="Duplicate"):
        merged.aggregate(left)


def test_lora_requests_do_not_read_or_publish_the_base_weight_namespace() -> None:
    """Adapter IDs do not authenticate immutable adapter bytes across restarts."""
    with open_checkpoint_rpc() as (client, module, _mapping, _name):
        manager = make_manager()
        bridge = CheckpointSchedulerBridge(manager, client, {}, 4)
        layout = {"schema_version": 1, "page_bytes": 128}
        bridge.accept_layouts({rank: layout for rank in range(4)})
        request = make_request("adapter-request")
        request.lora_request = LoRARequest("adapter", 1, "/adapter")
        checkpoint = manager.reserve_external_boundary_checkpoint(
            request,
            11,
            manager.boundary_checkpoint_page_positions(11),
            draft_prefix_len=11,
            kind="prompt",
            num_ranks=4,
        )
        assert checkpoint is not None
        for rank in range(4):
            manager.acknowledge_external_boundary_checkpoint(
                checkpoint.checkpoint_id, rank
            )
        assert not bridge.handles(request)
        assert bridge.poll_prefix(request)
        bridge.store(request, checkpoint)
        assert bridge.take_tasks() == []
        assert manager.reset_prefix_cache()
        assert (
            module.report_status()["recurrent_checkpoints"]["pending_generations"] == 0
        )


def make_request(name: str, *, first_token: int = 0) -> Request:
    """Construct an exact eleven-token prompt with ordinary cache authentication."""
    params = SamplingParams(max_tokens=1)
    params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=name,
        prompt_token_ids=list(range(first_token, first_token + 11)),
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(4, sha256),
    )


def make_manager() -> KVCacheManager:
    """Mix full attention and an endpoint-only recurrent group in one block pool."""
    init_none_hash(sha256)
    config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["target.attention"],
                FullAttentionSpec(
                    block_size=4,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["target.recurrent"],
                MambaSpec(
                    block_size=4,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                ),
            ),
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=128,
        hash_block_size=4,
        scheduler_block_size=4,
        enable_boundary_checkpoints=True,
    )


@pytest.mark.parametrize(
    "outcome",
    [
        "success",
        "rank-miss",
        "cancelled",
        "rejected-store-reused-request-id",
        "inflight-store-reused-request-id",
    ],
)
def test_semantic_roundtrip_collective_visibility_and_cancellation(
    outcome: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with open_checkpoint_rpc() as (client, module, mapping, _name):
        manager = make_manager()
        bridge = CheckpointSchedulerBridge(
            manager,
            client,
            {
                "target_revision": "target-content",
                "draft_revision": "",
                "source_revision": "source-content",
                "parallel": {"tp": 4, "dcp": 1},
            },
            4,
        )
        layout = {"schema_version": 1, "page_bytes": 128}
        bridge.accept_layouts({rank: layout for rank in range(4)})
        producer = make_request("producer")
        checkpoint = manager.reserve_external_boundary_checkpoint(
            producer,
            11,
            manager.boundary_checkpoint_page_positions(11),
            draft_prefix_len=11,
            kind="prompt",
            num_ranks=4,
        )
        assert checkpoint is not None
        for rank in range(4):
            manager.acknowledge_external_boundary_checkpoint(
                checkpoint.checkpoint_id, rank
            )
        if outcome == "rejected-store-reused-request-id":
            # A producer can disconnect before the server rejects its store.
            # Reusing its request ID must not cancel an independent import.
            rejected = make_request("consumer")
            with monkeypatch.context() as patch:
                patch.setattr(
                    client,
                    "submit_request",
                    lambda *_: SimpleNamespace(
                        query=lambda: True, result=lambda: False
                    ),
                )
                bridge.store(rejected, checkpoint)
            bridge.finish_request(rejected.request_id)
            assert bridge.take_tasks() == []
            assert not bridge.has_pending
        bridge.store(producer, checkpoint)
        bridge.finish_request(producer.request_id)
        assert not manager.reset_prefix_cache()
        deadline = time.monotonic() + 5
        tasks: list[CheckpointEngineTask] = []
        while not tasks and time.monotonic() < deadline:
            tasks = bridge.take_tasks()
            time.sleep(0.001)
        assert len(tasks) == 1 and not bridge.take_tasks()
        store_task = tasks[0]

        def copy_pages(job, lease) -> None:
            for group_id, group in enumerate(lease.slots):
                for page_id, (offset, size) in enumerate(group):
                    pattern = bytes([job.rank * 16 + group_id * 4 + page_id]) * size
                    if job.direction == "STORE":
                        mapping[offset : offset + size] = pattern
                    else:
                        assert mapping[offset : offset + size] == pattern

        worker = CheckpointTransferWorker(client, copy_pages)
        try:
            for rank in range(4):
                future = worker.submit(
                    CheckpointTransferJob(
                        store_task.manifest,
                        rank,
                        "STORE",
                        store_task.block_ids,
                    )
                )
                assert future is not None and future.result(timeout=5)
                bridge.complete({store_task.task_id: {rank: True}})
                assert manager.reset_prefix_cache() == (rank == 3)
            assert not bridge.has_pending
            consumer = make_request("consumer")
            inflight_store = None
            if outcome == "inflight-store-reused-request-id":
                # This cancelled producer has different tokens but the same
                # public ID as the waiting consumer. Its admitted copy is still
                # live while the consumer looks up the first producer's data.
                predecessor = make_request("consumer", first_token=20)
                predecessor_checkpoint = manager.reserve_external_boundary_checkpoint(
                    predecessor,
                    11,
                    manager.boundary_checkpoint_page_positions(11),
                    draft_prefix_len=11,
                    kind="prompt",
                    num_ranks=4,
                )
                assert predecessor_checkpoint is not None
                for rank in range(4):
                    manager.acknowledge_external_boundary_checkpoint(
                        predecessor_checkpoint.checkpoint_id, rank
                    )
                bridge.store(predecessor, predecessor_checkpoint)
                bridge.finish_request(predecessor.request_id)
                deadline = time.monotonic() + 5
                pending: list[CheckpointEngineTask] = []
                while not pending and time.monotonic() < deadline:
                    pending = bridge.take_tasks()
                    time.sleep(0.001)
                assert len(pending) == 1
                inflight_store = pending[0]

            def drain_predecessor() -> None:
                nonlocal inflight_store
                assert inflight_store is not None
                for rank in range(4):
                    future = worker.submit(
                        CheckpointTransferJob(
                            inflight_store.manifest,
                            rank,
                            "STORE",
                            inflight_store.block_ids,
                        )
                    )
                    assert future is not None and future.result(timeout=5)
                    bridge.complete({inflight_store.task_id: {rank: True}})
                inflight_store = None

            before = manager.block_pool.get_num_free_blocks()
            deadline = time.monotonic() + 5
            drain_after = time.monotonic() + 0.1
            tasks = []
            while not tasks and time.monotonic() < deadline:
                assert not bridge.poll_prefix(consumer)
                tasks = bridge.take_tasks()
                if (
                    not tasks
                    and inflight_store is not None
                    and time.monotonic() >= drain_after
                ):
                    # Deferring reuse until the predecessor drains is valid.
                    # Admitting it sooner must not inherit cancellation or lose
                    # its lookup when the predecessor completes.
                    drain_predecessor()
                    before = manager.block_pool.get_num_free_blocks()
                time.sleep(0.001)
            assert len(tasks) == 1
            restore_task = tasks[0]
            assert manager.get_computed_blocks(consumer)[1] == 0
            if outcome == "cancelled":
                bridge.finish_request(consumer.request_id)
            for rank in range(4):
                future = worker.submit(
                    CheckpointTransferJob(
                        restore_task.manifest,
                        rank,
                        "RETRIEVE",
                        restore_task.block_ids,
                    )
                )
                assert future is not None and future.result(timeout=5)
                bridge.complete(
                    {
                        restore_task.task_id: {
                            rank: not (outcome == "rank-miss" and rank == 2)
                        }
                    }
                )
                if rank < 3:
                    assert manager.get_computed_blocks(consumer)[1] == 0
                    assert manager.block_pool.get_num_free_blocks() < before
            assert manager.block_pool.get_num_free_blocks() == before
            expected_tokens = 0 if outcome in ("rank-miss", "cancelled") else 11
            assert manager.get_computed_blocks(consumer)[1] == expected_tokens
            assert bridge.external_tokens(consumer) == expected_tokens
            if inflight_store is not None:
                drain_predecessor()
            assert not bridge.has_pending
            assert (
                module.report_status()["recurrent_checkpoints"]["retrieve_leases"] == 0
            )
        finally:
            worker.close()
