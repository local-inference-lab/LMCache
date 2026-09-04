# SPDX-License-Identifier: Apache-2.0

# Standard
import random

# Third Party
import pytest
import torch

pytest.importorskip(
    "lmcache.c_ops",
    reason="Requires CUDA extension lmcache.c_ops",
)

# First Party
import lmcache.c_ops as lmc_ops

# Skip all tests if cuda is unavailable
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() == 0,
    reason="No CUDA GPU present",
)

# ---------------------------------------------------------------------------
# Tensor factories (ported from kernel harness)
# ---------------------------------------------------------------------------


def _create_random_tensor(
    shape: list, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if dtype == torch.float8_e4m3fn:
        return torch.rand(shape, dtype=torch.bfloat16, device=device).to(dtype)
    if dtype == torch.uint8:
        return torch.randint(0, 256, shape, dtype=dtype, device=device)
    return torch.rand(shape, dtype=dtype, device=device)


def _create_zero_tensor(
    shape: list, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if dtype == torch.float8_e4m3fn:
        return torch.zeros(shape, dtype=torch.bfloat16, device=device).to(dtype)
    return torch.zeros(shape, dtype=dtype, device=device)


# Format enum values from c_ops
FMT_NORMAL = lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
FMT_CROSS_LAYER = lmc_ops.EngineKVFormat.NB_NL_TWO_BS_NH_HS
FMT_FLASH_INFER = lmc_ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS
FMT_MLA = lmc_ops.EngineKVFormat.NL_X_NB_BS_HS
FMT_SGLANG_MHA = lmc_ops.EngineKVFormat.TWO_X_NL_X_NBBS_NH_HS
FMT_SGLANG_MLA = lmc_ops.EngineKVFormat.NL_X_NBBS_ONE_HS
FMT_NORMAL_HND = lmc_ops.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS
FMT_FLASH_INFER_HND = lmc_ops.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS
FMT_VLLM_FUSED_HND = lmc_ops.EngineKVFormat.NL_X_NB_NH_BS_TWO_HS
FMT_VLLM_FUSED_NHD = lmc_ops.EngineKVFormat.NL_X_NB_BS_NH_TWO_HS

# Format parameters: (engine_kv_format, num_layers, num_heads, head_size, is_mla)
# The is_mla column really means "kv_size == 1": the fused-K/V HND format is
# not MLA but transfers with kv_size == 1 and hs already doubled (kv-packed D).
# Use small layer counts to keep GPU memory usage low in CI
FORMAT_PARAMS = [
    (FMT_NORMAL, 4, 8, 128, False),
    (FMT_CROSS_LAYER, 4, 8, 128, False),
    (FMT_FLASH_INFER, 4, 8, 128, False),
    (FMT_MLA, 4, 1, 576, True),
    (FMT_SGLANG_MHA, 4, 8, 128, False),
    (FMT_SGLANG_MLA, 4, 1, 576, True),
    (FMT_NORMAL_HND, 4, 8, 128, False),
    (FMT_FLASH_INFER_HND, 4, 8, 128, False),
    (FMT_VLLM_FUSED_HND, 4, 8, 256, True),
    (FMT_VLLM_FUSED_NHD, 4, 8, 256, True),
]


def create_vllm_tensors(
    engine_kv_format,
    nl: int,
    nb: int,
    bs: int,
    nh: int,
    hs: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list[torch.Tensor]:
    nbbs = nb * bs
    if engine_kv_format == FMT_NORMAL:
        shape = [2, nb, bs, nh, hs]
        return [_create_random_tensor(shape, dtype, device) for _ in range(nl)]
    elif engine_kv_format == FMT_NORMAL_HND:
        shape = [2, nb, nh, bs, hs]
        return [_create_random_tensor(shape, dtype, device) for _ in range(nl)]
    elif engine_kv_format == FMT_CROSS_LAYER:
        shape = [nb, nl, 2, bs, nh, hs]
        return [_create_random_tensor(shape, dtype, device)]
    elif engine_kv_format == FMT_FLASH_INFER:
        shape = [nb, 2, bs, nh, hs]
        return [_create_random_tensor(shape, dtype, device) for _ in range(nl)]
    elif engine_kv_format == FMT_FLASH_INFER_HND:
        shape = [nb, 2, nh, bs, hs]
        return [_create_random_tensor(shape, dtype, device) for _ in range(nl)]
    elif engine_kv_format == FMT_VLLM_FUSED_HND:
        shape = [nb, nh, bs, hs]  # hs is the fused 2 * head_size
        return [_create_random_tensor(shape, dtype, device) for _ in range(nl)]
    elif engine_kv_format == FMT_VLLM_FUSED_NHD:
        shape = [nb, bs, nh, hs]  # hs is the fused 2 * head_size
        return [_create_random_tensor(shape, dtype, device) for _ in range(nl)]
    elif engine_kv_format == FMT_MLA:
        shape = [nb, bs, hs]
        return [_create_random_tensor(shape, dtype, device) for _ in range(nl)]
    elif engine_kv_format == FMT_SGLANG_MHA:
        shape = [nbbs, nh, hs]
        return [_create_random_tensor(shape, dtype, device) for _ in range(2 * nl)]
    elif engine_kv_format == FMT_SGLANG_MLA:
        shape = [nbbs, 1, hs]
        return [_create_random_tensor(shape, dtype, device) for _ in range(nl)]
    raise ValueError(f"Unknown format: {engine_kv_format}")


def create_zero_vllm_tensors(
    engine_kv_format,
    nl: int,
    nb: int,
    bs: int,
    nh: int,
    hs: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list[torch.Tensor]:
    nbbs = nb * bs
    if engine_kv_format == FMT_NORMAL:
        shape = [2, nb, bs, nh, hs]
        return [_create_zero_tensor(shape, dtype, device) for _ in range(nl)]
    elif engine_kv_format == FMT_NORMAL_HND:
        shape = [2, nb, nh, bs, hs]
        return [_create_zero_tensor(shape, dtype, device) for _ in range(nl)]
    elif engine_kv_format == FMT_CROSS_LAYER:
        shape = [nb, nl, 2, bs, nh, hs]
        return [_create_zero_tensor(shape, dtype, device)]
    elif engine_kv_format == FMT_FLASH_INFER:
        shape = [nb, 2, bs, nh, hs]
        return [_create_zero_tensor(shape, dtype, device) for _ in range(nl)]
    elif engine_kv_format == FMT_FLASH_INFER_HND:
        shape = [nb, 2, nh, bs, hs]
        return [_create_zero_tensor(shape, dtype, device) for _ in range(nl)]
    elif engine_kv_format == FMT_VLLM_FUSED_HND:
        shape = [nb, nh, bs, hs]  # hs is the fused 2 * head_size
        return [_create_zero_tensor(shape, dtype, device) for _ in range(nl)]
    elif engine_kv_format == FMT_VLLM_FUSED_NHD:
        shape = [nb, bs, nh, hs]  # hs is the fused 2 * head_size
        return [_create_zero_tensor(shape, dtype, device) for _ in range(nl)]
    elif engine_kv_format == FMT_MLA:
        shape = [nb, bs, hs]
        return [_create_zero_tensor(shape, dtype, device) for _ in range(nl)]
    elif engine_kv_format == FMT_SGLANG_MHA:
        shape = [nbbs, nh, hs]
        return [_create_zero_tensor(shape, dtype, device) for _ in range(2 * nl)]
    elif engine_kv_format == FMT_SGLANG_MLA:
        shape = [nbbs, 1, hs]
        return [_create_zero_tensor(shape, dtype, device) for _ in range(nl)]
    raise ValueError(f"Unknown format: {engine_kv_format}")


def create_memory_objects(
    kv_dim: int,
    nl: int,
    tokens_per_object: int,
    hidden_dim: int,
    num_objects: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list[torch.Tensor]:
    shape = [kv_dim, nl, tokens_per_object, hidden_dim]
    objects = []
    for _ in range(num_objects):
        if dtype == torch.float8_e4m3fn:
            t = torch.zeros(shape, dtype=torch.bfloat16, device=device)
            t = t.to(dtype)
        else:
            t = torch.zeros(shape, dtype=dtype, device=device)
        if device.type == "cpu":
            t = t.pin_memory()
        objects.append(t)
    return objects


def get_block_data(
    vllm_tensors: list[torch.Tensor],
    engine_kv_format,
    nl: int,
    bs: int,
    nh: int,
    block_idx: int,
) -> list[torch.Tensor]:
    """Extract all layer data for a given block."""
    results = []
    for layer_idx in range(nl):
        if engine_kv_format == FMT_NORMAL:
            results.append(vllm_tensors[layer_idx][:, block_idx, :, :, :].clone())
        elif engine_kv_format == FMT_NORMAL_HND:
            results.append(vllm_tensors[layer_idx][:, block_idx, :, :, :].clone())
        elif engine_kv_format == FMT_CROSS_LAYER:
            results.append(vllm_tensors[0][block_idx, layer_idx, :, :, :, :].clone())
        elif engine_kv_format == FMT_FLASH_INFER:
            results.append(vllm_tensors[layer_idx][block_idx, :, :, :, :].clone())
        elif engine_kv_format == FMT_FLASH_INFER_HND:
            results.append(vllm_tensors[layer_idx][block_idx, :, :, :, :].clone())
        elif engine_kv_format == FMT_VLLM_FUSED_HND:
            results.append(vllm_tensors[layer_idx][block_idx, :, :, :].clone())
        elif engine_kv_format == FMT_VLLM_FUSED_NHD:
            results.append(vllm_tensors[layer_idx][block_idx, :, :, :].clone())
        elif engine_kv_format == FMT_MLA:
            results.append(vllm_tensors[layer_idx][block_idx, :, :].clone())
        elif engine_kv_format == FMT_SGLANG_MHA:
            ts, ed = block_idx * bs, (block_idx + 1) * bs
            k = vllm_tensors[layer_idx][ts:ed, :, :].clone()
            v = vllm_tensors[nl + layer_idx][ts:ed, :, :].clone()
            results.append(torch.stack([k, v], dim=0))
        elif engine_kv_format == FMT_SGLANG_MLA:
            ts, ed = block_idx * bs, (block_idx + 1) * bs
            results.append(vllm_tensors[layer_idx][ts:ed, 0, :].clone())
    return results


# ---------------------------------------------------------------------------
# Kernel call helper
# ---------------------------------------------------------------------------


def call_block_kernel(
    vllm_tensors: list[torch.Tensor],
    mem_objects: list[torch.Tensor],
    block_ids: list[int],
    engine_kv_format,
    direction,
    nl: int,
    nb: int,
    bs: int,
    nh: int,
    hs: int,
    is_mla: bool,
    tokens_per_object: int,
    skip_prefix_n_blocks: int = 0,
    block_stride_elems: int = 0,
) -> None:
    device = vllm_tensors[0].device

    shape_desc = lmc_ops.PageBufferShapeDesc()
    shape_desc.kv_size = 1 if is_mla else 2
    shape_desc.nl = nl
    shape_desc.nb = nb
    shape_desc.bs = bs
    shape_desc.nh = nh
    shape_desc.hs = hs
    shape_desc.element_size = vllm_tensors[0].element_size()
    shape_desc.block_stride_elems = block_stride_elems

    ptrs = [t.data_ptr() for t in vllm_tensors]
    paged_buffer_ptrs_tensor = torch.tensor(ptrs, dtype=torch.int64, device=device)
    lmcache_objects_ptrs = [m.data_ptr() for m in mem_objects]

    block_ids_gpu = torch.tensor(block_ids, dtype=torch.int64, device=device)
    lmc_ops.multi_layer_block_kv_transfer(
        paged_buffer_ptrs_tensor,
        lmcache_objects_ptrs,
        block_ids_gpu,
        device,
        direction,
        shape_desc,
        tokens_per_object,
        engine_kv_format,
        skip_prefix_n_blocks,
    )


def test_fused_nhd_transfer_supports_64_heads() -> None:
    """Compressed-state pages with 64 heads use a legal CUDA block shape."""
    device = torch.device("cuda")
    num_layers, num_blocks = 2, 8
    block_size, num_heads, fused_head_size = 1, 64, 132
    source = create_vllm_tensors(
        FMT_VLLM_FUSED_NHD,
        num_layers,
        num_blocks,
        block_size,
        num_heads,
        fused_head_size,
        torch.uint8,
        device,
    )
    destination = create_zero_vllm_tensors(
        FMT_VLLM_FUSED_NHD,
        num_layers,
        num_blocks,
        block_size,
        num_heads,
        fused_head_size,
        torch.uint8,
        device,
    )
    objects = create_memory_objects(
        1,
        num_layers,
        2,
        num_heads * fused_head_size,
        1,
        torch.uint8,
        device,
    )

    call_block_kernel(
        source,
        objects,
        [1, 6],
        FMT_VLLM_FUSED_NHD,
        lmc_ops.TransferDirection.D2H,
        num_layers,
        num_blocks,
        block_size,
        num_heads,
        fused_head_size,
        True,
        2,
    )
    call_block_kernel(
        destination,
        objects,
        [2, 7],
        FMT_VLLM_FUSED_NHD,
        lmc_ops.TransferDirection.H2D,
        num_layers,
        num_blocks,
        block_size,
        num_heads,
        fused_head_size,
        True,
        2,
    )
    torch.cuda.synchronize()

    for layer in range(num_layers):
        assert torch.equal(source[layer][1], destination[layer][2])
        assert torch.equal(source[layer][6], destination[layer][7])


def test_engine_driven_transfer_supports_interleaved_compressed_pages() -> None:
    """Engine-driven transfer preserves interleaved 64-head cache pages."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context.base import (
        gather_paged_kv_to_cpu,
        scatter_cpu_to_paged_kv,
    )

    device = torch.device("cuda")
    num_blocks, num_layers = 8, 2
    pool_shape = (num_blocks, num_layers, 1, 64, 132)
    source_pool = (
        torch.arange(
            int(torch.tensor(pool_shape).prod()), dtype=torch.int64, device=device
        )
        .to(torch.uint8)
        .reshape(pool_shape)
    )
    destination_pool = torch.full(pool_shape, 0xA5, dtype=torch.uint8, device=device)
    source = {f"layer_{layer}": source_pool[:, layer] for layer in range(num_layers)}
    destination = {
        f"layer_{layer}": destination_pool[:, layer] for layer in range(num_layers)
    }
    logical_page_elems = source_pool[:, 0][0].numel()
    assert source_pool[:, 0].stride(0) == num_layers * logical_page_elems

    chunks = gather_paged_kv_to_cpu(
        source,
        [1, 6],
        blocks_per_chunk=2,
        layout_hints={"kv_layout": "NHD"},
    )
    torch.cuda.synchronize()
    scatter_cpu_to_paged_kv(
        destination,
        [2, 7],
        chunks,
        blocks_per_chunk=2,
        layout_hints={"kv_layout": "NHD"},
    )
    torch.cuda.synchronize()

    assert torch.equal(destination_pool[2], source_pool[1])
    assert torch.equal(destination_pool[7], source_pool[6])
    untouched = set(range(num_blocks)) - {2, 7}
    for block_idx in untouched:
        assert torch.all(destination_pool[block_idx] == 0xA5)


@pytest.mark.parametrize(
    "engine_kv_format,pool_shape",
    [
        (FMT_VLLM_FUSED_NHD, (12, 3, 4, 2, 16)),
        (FMT_VLLM_FUSED_HND, (12, 3, 2, 4, 16)),
    ],
    ids=["nhd", "hnd"],
)
def test_blocks_first_fused_kv_padded_stride_roundtrip(engine_kv_format, pool_shape):
    """CUDA transfers address each layer through the physical block stride."""
    device = torch.device("cuda")
    num_blocks, num_layers = pool_shape[:2]
    block_size = 4
    num_heads = 2
    fused_head_size = 16
    source_pool = (
        torch.arange(
            int(torch.tensor(pool_shape).prod()), dtype=torch.int64, device=device
        )
        .to(torch.uint8)
        .reshape(pool_shape)
    )
    target_pool = torch.full(pool_shape, 0xA5, dtype=torch.uint8, device=device)
    source_layers = [source_pool[:, layer_idx] for layer_idx in range(num_layers)]
    target_layers = [target_pool[:, layer_idx] for layer_idx in range(num_layers)]
    block_stride = source_layers[0].stride(0)
    logical_page_elems = source_layers[0][0].numel()
    assert block_stride == num_layers * logical_page_elems

    tokens_per_object = block_size * 2
    memory_objects = create_memory_objects(
        1,
        num_layers,
        tokens_per_object,
        num_heads * fused_head_size,
        1,
        torch.uint8,
        device,
    )
    source_block_ids = [1, 7]
    target_block_ids = [3, 9]
    call_block_kernel(
        source_layers,
        memory_objects,
        source_block_ids,
        engine_kv_format,
        lmc_ops.TransferDirection.D2H,
        num_layers,
        num_blocks,
        block_size,
        num_heads,
        fused_head_size,
        True,
        tokens_per_object,
        block_stride_elems=block_stride,
    )
    # The helper creates a temporary device pointer array. Production retains
    # that array for the connector lifetime; synchronize here before the local
    # tensor can be reclaimed and reused by the following launch.
    torch.cuda.synchronize()
    call_block_kernel(
        target_layers,
        memory_objects,
        target_block_ids,
        engine_kv_format,
        lmc_ops.TransferDirection.H2D,
        num_layers,
        num_blocks,
        block_size,
        num_heads,
        fused_head_size,
        True,
        tokens_per_object,
        block_stride_elems=block_stride,
    )
    torch.cuda.synchronize()

    for source_block, target_block in zip(
        source_block_ids, target_block_ids, strict=True
    ):
        assert torch.equal(target_pool[target_block], source_pool[source_block])
    untouched = set(range(num_blocks)) - set(target_block_ids)
    for block_idx in untouched:
        assert torch.all(target_pool[block_idx] == 0xA5)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

NB = 1000
BS = 16
NUM_MEMORY_OBJECTS = 4
TOKENS_PER_OBJECT = 256
BLOCKS_PER_OBJECT = TOKENS_PER_OBJECT // BS  # 16
TOTAL_BLOCKS = NUM_MEMORY_OBJECTS * BLOCKS_PER_OBJECT  # 64


@pytest.mark.parametrize(
    "engine_kv_format,nl,nh,hs,is_mla",
    FORMAT_PARAMS,
    ids=[
        "normal",
        "cross_layer",
        "flash_infer",
        "mla",
        "sglang_mha",
        "sglang_mla",
        "normal_hnd",
        "flash_infer_hnd",
        "vllm_fused_hnd",
        "vllm_fused_nhd",
    ],
)
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"]
)
@pytest.mark.parametrize("mem_device", ["cuda", "cpu"], ids=["mem_gpu", "mem_cpu"])
def test_block_transfer_roundtrip(
    engine_kv_format, nl, nh, hs, is_mla, dtype, mem_device
):
    """
    D2H -> H2D roundtrip with different block IDs proves data flows through
    memory objects.
    """
    device = torch.device("cuda")
    mem_dev = torch.device(mem_device)
    kv_dim = 1 if is_mla else 2
    hidden_dim = nh * hs

    # Create tensors
    source_vllm = create_vllm_tensors(
        engine_kv_format, nl, NB, BS, nh, hs, dtype, device
    )
    target_vllm = create_zero_vllm_tensors(
        engine_kv_format, nl, NB, BS, nh, hs, dtype, device
    )
    mem_objects = create_memory_objects(
        kv_dim,
        nl,
        TOKENS_PER_OBJECT,
        hidden_dim,
        NUM_MEMORY_OBJECTS,
        dtype,
        mem_dev,
    )

    # Disjoint block IDs for D2H and H2D
    rng_d2h = random.Random(42)
    block_ids_d2h = rng_d2h.sample(range(NB), TOTAL_BLOCKS)
    excluded = set(block_ids_d2h)
    available = [i for i in range(NB) if i not in excluded]
    rng_h2d = random.Random(123)
    block_ids_h2d = rng_h2d.sample(available, TOTAL_BLOCKS)

    # D2H: source -> mem_objects
    call_block_kernel(
        source_vllm,
        mem_objects,
        block_ids_d2h,
        engine_kv_format,
        lmc_ops.TransferDirection.D2H,
        nl,
        NB,
        BS,
        nh,
        hs,
        is_mla,
        TOKENS_PER_OBJECT,
    )
    torch.cuda.synchronize()

    # H2D: mem_objects -> target
    call_block_kernel(
        target_vllm,
        mem_objects,
        block_ids_h2d,
        engine_kv_format,
        lmc_ops.TransferDirection.H2D,
        nl,
        NB,
        BS,
        nh,
        hs,
        is_mla,
        TOKENS_PER_OBJECT,
    )
    torch.cuda.synchronize()

    # Verify: target[h2d_block] == source[d2h_block]
    for i in range(TOTAL_BLOCKS):
        src_data = get_block_data(
            source_vllm, engine_kv_format, nl, BS, nh, block_ids_d2h[i]
        )
        tgt_data = get_block_data(
            target_vllm, engine_kv_format, nl, BS, nh, block_ids_h2d[i]
        )
        for layer_idx in range(nl):
            assert torch.equal(src_data[layer_idx], tgt_data[layer_idx]), (
                f"Mismatch at block index {i}, layer {layer_idx}"
            )


@pytest.mark.parametrize(
    "engine_kv_format,nl,nh,hs,is_mla",
    FORMAT_PARAMS,
    ids=[
        "normal",
        "cross_layer",
        "flash_infer",
        "mla",
        "sglang_mha",
        "sglang_mla",
        "normal_hnd",
        "flash_infer_hnd",
        "vllm_fused_hnd",
        "vllm_fused_nhd",
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
def test_block_transfer_skip_prefix(engine_kv_format, nl, nh, hs, is_mla, dtype):
    """Verify skip_prefix_n_blocks=4 skips the first 4 blocks globally."""
    device = torch.device("cuda")
    kv_dim = 1 if is_mla else 2
    hidden_dim = nh * hs
    skip = 4

    source_vllm = create_vllm_tensors(
        engine_kv_format, nl, NB, BS, nh, hs, dtype, device
    )
    target_vllm = create_zero_vllm_tensors(
        engine_kv_format, nl, NB, BS, nh, hs, dtype, device
    )
    mem_objects = create_memory_objects(
        kv_dim,
        nl,
        TOKENS_PER_OBJECT,
        hidden_dim,
        NUM_MEMORY_OBJECTS,
        dtype,
        device,
    )

    rng_d2h = random.Random(42)
    block_ids_d2h = rng_d2h.sample(range(NB), TOTAL_BLOCKS)
    excluded = set(block_ids_d2h)
    available = [i for i in range(NB) if i not in excluded]
    rng_h2d = random.Random(123)
    block_ids_h2d = rng_h2d.sample(available, TOTAL_BLOCKS)

    # D2H with skip
    call_block_kernel(
        source_vllm,
        mem_objects,
        block_ids_d2h,
        engine_kv_format,
        lmc_ops.TransferDirection.D2H,
        nl,
        NB,
        BS,
        nh,
        hs,
        is_mla,
        TOKENS_PER_OBJECT,
        skip_prefix_n_blocks=skip,
    )
    torch.cuda.synchronize()

    # H2D with skip
    call_block_kernel(
        target_vllm,
        mem_objects,
        block_ids_h2d,
        engine_kv_format,
        lmc_ops.TransferDirection.H2D,
        nl,
        NB,
        BS,
        nh,
        hs,
        is_mla,
        TOKENS_PER_OBJECT,
        skip_prefix_n_blocks=skip,
    )
    torch.cuda.synchronize()

    # Non-skipped blocks should match
    for i in range(skip, TOTAL_BLOCKS):
        src_data = get_block_data(
            source_vllm, engine_kv_format, nl, BS, nh, block_ids_d2h[i]
        )
        tgt_data = get_block_data(
            target_vllm, engine_kv_format, nl, BS, nh, block_ids_h2d[i]
        )
        for layer_idx in range(nl):
            assert torch.equal(src_data[layer_idx], tgt_data[layer_idx]), (
                f"Mismatch at block index {i}, layer {layer_idx}"
            )

    # Skipped blocks in target should remain zero
    for i in range(skip):
        tgt_data = get_block_data(
            target_vllm, engine_kv_format, nl, BS, nh, block_ids_h2d[i]
        )
        for layer_idx in range(nl):
            block = tgt_data[layer_idx]
            if block.dtype == torch.float8_e4m3fn:
                block = block.to(torch.float32)
            assert block.abs().sum().item() == 0, (
                f"Skipped block {i}, layer {layer_idx} is not zero"
            )
