# SPDX-License-Identifier: Apache-2.0
"""Preserve opaque recurrent pages when their token span does not divide bytes."""

# Standard
from typing import Literal
import os

# Third Party
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
)
import pytest
import torch

# First Party
from lmcache.integration.vllm.kv_cache_group_edits import apply_kv_cache_group_edits
from lmcache.integration.vllm.kv_cache_groups import create_engine_group_infos_from_vllm
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.transfer_context.base import (
    compute_kv_layout,
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)


def _config(page_bytes: int, cadence: int) -> KVCacheConfig:
    """Describe one opaque recurrent layer and one ordinary MLA layer."""
    return KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["recurrent"],
                kv_cache_spec=MambaSpec(
                    block_size=cadence,
                    shapes=((page_bytes,),),
                    dtypes=(torch.uint8,),
                    mamba_cache_mode="align",
                ),
            ),
            KVCacheGroupSpec(
                layer_names=["mla"],
                kv_cache_spec=MLAAttentionSpec(
                    block_size=8, num_kv_heads=1, head_size=16, dtype=torch.float16
                ),
            ),
        ],
    )


def _views(
    raw: torch.Tensor, cadence: int, layout: Literal["NHD", "HND"]
) -> tuple[KVCacheConfig, dict[str, torch.Tensor]]:
    config = _config(raw.shape[-1], cadence)
    caches = {
        "recurrent": raw,
        "mla": torch.empty(8, 8, 16, device=raw.device, dtype=torch.float16),
    }
    registered = apply_kv_cache_group_edits(config, caches, {"kv_layout": layout})
    edited: dict[str, torch.Tensor] = {}
    for name, tensor in registered.items():
        assert isinstance(tensor, torch.Tensor)
        edited[name] = tensor
    return config, edited


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize(
    ("cadence", "page_bytes", "slots", "width"),
    [(1536, 1007616, 1536, 656), (4608, 1007616, 1536, 656), (6, 32, 2, 16)],
)
def test_recurrent_view_preserves_page_and_logical_span(
    layout: Literal["NHD", "HND"],
    cadence: int,
    page_bytes: int,
    slots: int,
    width: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lmcache.v1.gpu_connector.kv_format.detectors.vllm.torch_device_type", "cuda"
    )
    raw = torch.arange(2 * page_bytes, dtype=torch.int32).remainder_(251)
    raw = raw.to(torch.uint8).view(2, 1, 1, page_bytes)
    config, edited = _views(raw, cadence, layout)
    view = edited["recurrent"]
    expected = (2, slots, 1, width) if layout == "NHD" else (2, 1, slots, width)
    assert tuple(view.shape) == expected
    assert view.data_ptr() == raw.data_ptr()
    assert view.untyped_storage().nbytes() == raw.untyped_storage().nbytes()
    assert torch.equal(view.flatten(), raw.flatten())
    groups = create_engine_group_infos_from_vllm(config, edited, {"kv_layout": layout})
    assert groups[0].tokens_per_block == cadence
    assert groups[0].sw_size_tokens == cadence
    assert config.kv_cache_groups[0].kv_cache_spec.block_size == cadence


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.skipif(
    os.getenv("LMCACHE_RUN_RECURRENT_PAGE_GPU_TEST") != "1",
    reason="requires an idle GPU and LMCACHE_RUN_RECURRENT_PAGE_GPU_TEST=1",
)
def test_recurrent_page_roundtrip_preserves_tail_and_block_ids(
    layout: Literal["NHD", "HND"], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "lmcache.v1.gpu_connector.kv_format.detectors.vllm.torch_device_type", "cuda"
    )
    page_bytes = 1007616
    source = torch.arange(4 * page_bytes, dtype=torch.int32).remainder_(251)
    source = source.to(device="cuda", dtype=torch.uint8).view(4, 1, 1, page_bytes)
    destination = torch.full_like(source, 255)
    _, src = _views(source, 4608, layout)
    _, dst = _views(destination, 4608, layout)
    hints: LayoutHints = {"kv_layout": layout}
    src_state, dst_state = (
        {"recurrent": src["recurrent"]},
        {"recurrent": dst["recurrent"]},
    )
    physical_slots, _, hidden, _, _, _ = compute_kv_layout(src_state, hints)
    assert (physical_slots, hidden) == (1536, 656)
    chunks = gather_paged_kv_to_cpu(src_state, [2, 0], 1, layout_hints=hints)
    torch.cuda.synchronize()
    assert sum(chunk.numel() for chunk in chunks) == 2 * page_bytes
    assert torch.equal(chunks[0].flatten(), source[2].cpu().flatten())
    assert torch.equal(chunks[1].flatten(), source[0].cpu().flatten())
    scatter_cpu_to_paged_kv(dst_state, [1, 3], chunks, 1, layout_hints=hints)
    torch.cuda.synchronize()
    assert torch.equal(destination[1], source[2])
    assert torch.equal(destination[3], source[0])
    assert (destination[[0, 2]] == 255).all()


@pytest.mark.skipif(
    os.getenv("LMCACHE_RUN_RECURRENT_PAGE_GPU_TEST") != "1",
    reason="requires 2.1 GiB free VRAM and the recurrent-page GPU gate",
)
def test_recurrent_page_roundtrip_uses_64_bit_page_offsets() -> None:
    page_bytes, source_id, destination_id = 1007616, 2198, 2199
    assert source_id * page_bytes > 2**31
    raw = torch.empty((2200, 1, 1, page_bytes), dtype=torch.uint8, device="cuda")
    pattern = (
        torch.arange(page_bytes, dtype=torch.int32).remainder_(251).to(torch.uint8)
    )
    raw[source_id].copy_(pattern.view(1, 1, -1))
    raw[destination_id].fill_(255)
    _, views = _views(raw, 4608, "NHD")
    state = {"recurrent": views["recurrent"]}
    chunks = gather_paged_kv_to_cpu(
        state, [source_id], 1, layout_hints={"kv_layout": "NHD"}
    )
    torch.cuda.synchronize()
    assert torch.equal(chunks[0].flatten(), pattern)
    scatter_cpu_to_paged_kv(
        state, [destination_id], chunks, 1, layout_hints={"kv_layout": "NHD"}
    )
    torch.cuda.synchronize()
    assert torch.equal(raw[destination_id].cpu().flatten(), pattern)
