# SPDX-License-Identifier: Apache-2.0
"""Async device<->CPU copies must complete before their buffers are reused.

Pickle serialization must wait for gathered data, and raw-pointer scatter must
retain locally pinned buffers until its copies complete, including on errors.
Caller-owned pinned buffers keep their asynchronous transfer contract.

These tests assert the ordering contract without a GPU, so they run in the
CPU-only unit CI. The hardware reproductions live alongside them in
``test_engine_driven_transfer.py`` and skip without CUDA.
"""

# Standard
from contextlib import nullcontext
from typing import cast
from unittest.mock import MagicMock, patch

# Third Party
import pytest
import torch


def test_scatter_syncs_before_releasing_dynamically_pinned_chunks() -> None:
    """Unpinned input must be synced before scatter returns.

    ``scatter_cpu_to_paged_kv`` pins unpinned chunks into temporaries and
    launches async H2D reads on them through raw pointers, which torch's
    stream tracking cannot see. Returning drops the last reference, so the
    caching host allocator can hand that memory to the next caller while the
    copies are in flight. The documented caller-side synchronize cannot cover
    it -- by then the temporaries are gone -- so scatter must sync itself.
    """
    # First Party
    from lmcache.v1.multiprocess.transfer_context import base

    kv = {f"layer_{i}": torch.zeros(2, 4, 4, 2, 8) for i in range(2)}

    # Mock chunks, not real tensors: pin_memory() needs an accelerator, so the
    # ptr-only branch cannot execute for real on the CPU-only unit CI. What we
    # are pinning down is the ordering contract, not the copy itself.
    def _unpinned_chunk() -> MagicMock:
        c = MagicMock()
        c.is_pinned.return_value = False
        c.pin_memory.return_value = c
        c.data_ptr.return_value = 0
        return c

    chunks = [_unpinned_chunk()]

    # Pin the ptr-only path: only that branch pins temporaries, and whether the
    # compiled op takes tensors varies by build. scatter imports device_ops
    # inside the function, so patch it at source.
    with (
        patch.object(base, "_LMC_OPS_BLOCK_TRANSFER_ACCEPTS_TENSOR", False),
        patch.object(base, "torch_dev") as dev,
        patch("lmcache.device_ops") as ops,
    ):
        # cast: the mocks stand in for tensors on purpose (see above).
        base.scatter_cpu_to_paged_kv(
            kv, list(range(4)), cast(list[torch.Tensor], chunks), 4
        )
        assert ops.multi_layer_block_kv_transfer.called, (
            "fixture must reach the async H2D launches"
        )
        assert dev.synchronize.called, (
            "scatter must complete async H2D before releasing the temporaries "
            "it pinned; otherwise the host allocator reuses them mid-copy"
        )


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("fail_batch", [None, 1, 2])
def test_scatter_drains_owned_temporaries_when_a_transfer_fails(
    pinned: bool, fail_batch: int | None
) -> None:
    """Raw-pointer transfers cannot outlive locally pinned source buffers."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import base

    kv = {f"layer_{i}": torch.zeros(2, 4, 4, 2, 8) for i in range(2)}
    chunks = [MagicMock() for _ in range(5)]
    for chunk in chunks:
        chunk.is_pinned.return_value = pinned
        chunk.pin_memory.return_value = chunk
        chunk.data_ptr.return_value = 0
    order: list[str] = []
    launches = 0

    def transfer(*_args: object, **_kwargs: object) -> None:
        nonlocal launches
        launches += 1
        order.append("launch")
        if launches == fail_batch:
            raise RuntimeError("transfer batch failed")

    with (
        patch.object(base, "_LMC_OPS_BLOCK_TRANSFER_ACCEPTS_TENSOR", False),
        patch.object(base, "torch_dev") as dev,
        patch("lmcache.device_ops") as ops,
    ):
        ops.multi_layer_block_kv_transfer.side_effect = transfer
        dev.synchronize.side_effect = lambda: order.append("sync")
        expected_error = (
            pytest.raises(RuntimeError, match="transfer batch failed")
            if fail_batch is not None
            else nullcontext()
        )
        with expected_error:
            base.scatter_cpu_to_paged_kv(
                kv, [0, 1, 2, 3] * 5, cast(list[torch.Tensor], chunks), 4
            )
        order.append("caller")

    expected_launches = fail_batch if fail_batch is not None else 2
    assert order == (
        ["launch"] * expected_launches + ([] if pinned else ["sync"]) + ["caller"]
    )


def test_scatter_does_not_sync_when_every_block_is_skipped() -> None:
    """An entirely cached prefix does not launch or synchronize a transfer."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import base

    kv = {f"layer_{i}": torch.zeros(2, 4, 4, 2, 8) for i in range(2)}
    chunk = MagicMock()
    chunk.is_pinned.return_value = False
    chunk.pin_memory.return_value = chunk
    chunk.data_ptr.return_value = 0

    with (
        patch.object(base, "_LMC_OPS_BLOCK_TRANSFER_ACCEPTS_TENSOR", False),
        patch.object(base, "torch_dev") as dev,
        patch("lmcache.device_ops") as ops,
    ):
        base.scatter_cpu_to_paged_kv(
            kv,
            list(range(4)),
            cast(list[torch.Tensor], [chunk]),
            4,
            skip_first_n_tokens=16,
        )
        ops.multi_layer_block_kv_transfer.assert_not_called()
        dev.synchronize.assert_not_called()


def test_pickle_store_syncs_before_commit_serializes() -> None:
    """The pickle path must sync before commit_store reads the buffers.

    Gather issues async device->CPU copies into fresh buffers, and the pickle
    transport serializes them immediately in ``commit_store``. Syncing only
    when ``out_buffers`` is given (the SHM path) leaves pickle serializing a
    buffer that is still being written.
    """
    # First Party
    from lmcache.v1.multiprocess.transfer_context import worker_transfer

    order: list[str] = []
    ctx = worker_transfer.EngineDrivenTransferContext()
    ctx._engine_driven_context = MagicMock()
    ctx._engine_driven_context.prepare_store.return_value = None  # pickle mode

    def _commit(*_a: object, **_k: object) -> bool:
        order.append("commit")
        return True

    ctx._engine_driven_context.commit_store.side_effect = _commit
    ctx._layout_hints = None
    ctx._engine_kv_format = None

    def _gather(*_a: object, **_k: object) -> list[torch.Tensor]:
        order.append("gather")
        return [torch.zeros(1)]

    with (
        patch.object(worker_transfer, "torch_dev") as dev,
        patch.object(worker_transfer, "gather_paged_kv_to_cpu", side_effect=_gather),
    ):
        dev.synchronize.side_effect = lambda *a, **k: order.append("sync")
        ctx.submit_store(
            "req",
            MagicMock(),  # key
            1,  # instance_id
            {"layer_0": torch.zeros(2, 4, 4, 2, 8)},
            [[0, 1, 2, 3]],
            MagicMock(),  # event (unused on this transport)
            4,  # blocks_in_chunk
        )

    # A sync must fall BETWEEN gather and commit. submit_store also syncs
    # before prepare_store, so merely finding a "sync" proves nothing -- that
    # earlier one is why guarding this on out_buffers went unnoticed.
    gathered, committed = order.index("gather"), order.index("commit")
    assert any(
        i for i, step in enumerate(order) if step == "sync" and gathered < i < committed
    ), (
        "pickle store must synchronize after gather and before commit_store "
        f"serializes the buffers, got {order}"
    )
