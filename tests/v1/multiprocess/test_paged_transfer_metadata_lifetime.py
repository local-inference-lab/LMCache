# SPDX-License-Identifier: Apache-2.0
"""Paged gather metadata must survive deferred host-to-device consumption."""

# Standard
from collections.abc import Callable
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache import device_ops
from lmcache.v1.multiprocess.transfer_context import base


@pytest.mark.parametrize("num_objects", [0, 1, 4, 5, 8, 11])
@pytest.mark.parametrize("blocks_per_window", [1, 2])
def test_gather_preserves_block_ids_until_async_copy_completes(
    monkeypatch: pytest.MonkeyPatch,
    num_objects: int,
    blocks_per_window: int,
) -> None:
    """Every native launch observes its selected IDs even if DMA is delayed.

    A CPU stream double defers both H2D reads and native launches until after
    gather submission. This is a legal CUDA execution order and exposes host
    metadata mutation without requiring a particular GPU scheduling delay.
    """
    source = {"layer": torch.zeros(32, 4, 16)}
    block_ids = list(reversed(range(22)))
    selected_chunks = list(reversed(range(num_objects)))
    expected = [
        block_ids[(chunk + 1) * 2 - blocks_per_window : (chunk + 1) * 2]
        for chunk in selected_chunks
    ]
    host_ids = torch.empty(4 * blocks_per_window, dtype=torch.int64)
    device_ids = torch.empty_like(host_ids)
    workspace = base.PagedKVTransferWorkspace(
        paged_buffer_ptrs=torch.tensor([source["layer"].data_ptr()]),
        block_ids_host=(host_ids,),
        block_ids_device=(device_ids,),
    )
    outputs = [torch.empty(1, blocks_per_window * 4, 16) for _ in selected_chunks]
    pending: list[Callable[[], None]] = []
    observed: list[list[int]] = []
    original_copy = torch.Tensor.copy_
    original_empty = torch.empty

    def deferred_copy(
        target: torch.Tensor, source: torch.Tensor, non_blocking: bool = False
    ) -> torch.Tensor:
        if target.data_ptr() == device_ids.data_ptr() and non_blocking:
            pending.append(lambda: original_copy(target, source))
            return target
        return original_copy(target, source, non_blocking=non_blocking)

    def cpu_empty(*args: Any, **kwargs: Any) -> torch.Tensor:
        kwargs.pop("pin_memory", None)
        return original_empty(*args, **kwargs)

    def deferred_transfer(
        _pointers: torch.Tensor,
        _outputs: list[int],
        ids: torch.Tensor,
        *_args: Any,
    ) -> None:
        def consume() -> None:
            observed.extend(ids.reshape(-1, blocks_per_window).tolist())

        pending.append(consume)

    monkeypatch.setattr(base, "_LMC_OPS_BLOCK_TRANSFER_ACCEPTS_TENSOR", False)
    monkeypatch.setattr(torch.Tensor, "copy_", deferred_copy)
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda _self: True)
    monkeypatch.setattr(torch, "empty", cpu_empty)
    monkeypatch.setattr(device_ops, "multi_layer_block_kv_transfer", deferred_transfer)

    returned = base.gather_paged_kv_to_cpu(
        source,
        block_ids,
        blocks_per_chunk=2,
        blocks_per_window=blocks_per_window,
        chunk_indices=selected_chunks,
        out=outputs,
        transfer_workspace=workspace,
    )
    for operation in pending:
        operation()

    assert returned is outputs
    assert observed == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA transfer test")
@pytest.mark.parametrize("blocks_per_window", [1, 2])
def test_native_gather_with_delayed_cuda_stream(blocks_per_window: int) -> None:
    """Native CUDA copies retain exact MLA bytes across three launch batches."""
    source_cpu = torch.arange(32 * 4 * 16, dtype=torch.float32).reshape(32, 4, 16)
    source = {"layer": source_cpu.cuda()}
    block_ids = list(reversed(range(22)))
    selected_chunks = list(reversed(range(11)))
    outputs = [
        torch.empty(1, blocks_per_window * 4, 16, pin_memory=True)
        for _ in selected_chunks
    ]
    workspace = base.create_paged_kv_transfer_workspace(
        source, max_block_ids=4 * blocks_per_window, num_slots=1
    )
    # Load the native kernel before deliberately delaying its execution stream.
    warmup = base.gather_paged_kv_to_cpu(
        source, block_ids[:2], 2, blocks_per_window=blocks_per_window
    )
    torch.cuda.synchronize()
    del warmup
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # CUDA's test-only sleep provides stream-local delay without a host sync.
        torch.cuda._sleep(500_000_000)  # noqa: SLF001
        returned = base.gather_paged_kv_to_cpu(
            source,
            block_ids,
            2,
            blocks_per_window=blocks_per_window,
            chunk_indices=selected_chunks,
            out=outputs,
            transfer_workspace=workspace,
        )
    stream.synchronize()
    assert returned is outputs
    for chunk_idx, output in zip(selected_chunks, outputs, strict=True):
        ids = block_ids[(chunk_idx + 1) * 2 - blocks_per_window : (chunk_idx + 1) * 2]
        expected = source_cpu[ids].reshape(1, blocks_per_window * 4, 16)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
