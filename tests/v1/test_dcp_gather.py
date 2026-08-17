# SPDX-License-Identifier: Apache-2.0
# Standard
import itertools

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.dcp_gather import (
    _deinterleave,
    _extract_shard,
    _interleave,
    _packed_row_bytes,
    _scatter_shard,
    dcp_local_token_count,
)


def _gathered(world: int, blocks: int, width: int) -> torch.Tensor:
    local_tokens = blocks * width
    gathered = torch.empty(1, 1, world * local_tokens, 1)
    for rank in range(world):
        for block in range(blocks):
            for offset in range(width):
                gathered[0, 0, rank * local_tokens + block * width + offset, 0] = (
                    block * world + rank
                ) * width + offset
    return gathered


@pytest.mark.parametrize(
    "world,blocks,width", list(itertools.product([2, 4], [1, 3], [1, 4]))
)
def test_interleave_produces_global_order(world, blocks, width):
    tokens = world * blocks * width
    full = _interleave(_gathered(world, blocks, width), world, width, tokens)
    expected = torch.arange(tokens, dtype=torch.float32).view(1, 1, tokens, 1)
    torch.testing.assert_close(full, expected)


@pytest.mark.parametrize(
    "world,blocks,width", list(itertools.product([2, 4], [1, 3], [1, 4]))
)
def test_deinterleave_inverts_interleave(world, blocks, width):
    local_tokens = blocks * width
    tokens = world * local_tokens
    gathered = _gathered(world, blocks, width)
    full = _interleave(gathered, world, width, tokens)
    for rank in range(world):
        shard = _deinterleave(full, world, rank, width, local_tokens)
        expected = gathered[:, :, rank * local_tokens : (rank + 1) * local_tokens]
        torch.testing.assert_close(shard, expected)


def test_deinterleave_selects_strided_blocks():
    world, blocks, width = 3, 2, 4
    local_tokens = blocks * width
    tokens = world * local_tokens
    full = torch.arange(tokens, dtype=torch.float32).view(1, 1, tokens, 1)
    for rank in range(world):
        shard = _deinterleave(full, world, rank, width, local_tokens).flatten()
        owned = {int(value) // width for value in shard}
        assert owned == {
            block for block in range(tokens // width) if block % world == rank
        }


@pytest.mark.parametrize(
    "global_tokens,world_size,local_tokens",
    [(3072, 4, 768), (46336, 4, 11584), (46337, 4, 11585), (256, 1, 256)],
)
def test_dcp_local_token_count(global_tokens, world_size, local_tokens):
    assert dcp_local_token_count(global_tokens, world_size) == local_tokens


def _assert_pack_round_trip(caches: list[torch.Tensor], slots: torch.Tensor) -> None:
    packed = _extract_shard(caches, slots)
    assert packed.dtype == torch.uint8
    assert packed.shape == (1, 1, len(slots), _packed_row_bytes(caches))

    destination = [torch.zeros_like(cache) for cache in caches]
    _scatter_shard(destination, slots, packed)
    for actual, expected in zip(destination, caches, strict=True):
        actual_flat = actual.view(-1, actual.shape[-1])
        expected_flat = expected.view(-1, expected.shape[-1])
        torch.testing.assert_close(actual_flat[slots], expected_flat[slots])
        untouched = torch.ones(len(actual_flat), dtype=torch.bool)
        untouched[slots] = False
        assert torch.count_nonzero(actual_flat[untouched]) == 0


def test_packed_round_trip_heterogeneous_widths():
    caches = [
        torch.arange(8 * 132, dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .view(8, 1, 132),
        torch.arange(8 * 432, dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .view(8, 1, 432),
    ]
    assert _packed_row_bytes(caches) == 132 + 432
    _assert_pack_round_trip(caches, torch.tensor([1, 4, 7]))


def test_packed_round_trip_mixed_dtypes():
    caches = [
        torch.arange(12, dtype=torch.float16).view(4, 1, 3),
        torch.arange(20, dtype=torch.uint8).view(4, 1, 5),
    ]
    assert _packed_row_bytes(caches) == 3 * 2 + 5
    _assert_pack_round_trip(caches, torch.tensor([0, 2]))


def test_packed_dcp4_round_trip():
    world, tokens = 4, 16
    caches = [
        torch.arange(tokens * 2, dtype=torch.uint8).view(tokens, 1, 2),
        torch.arange(tokens * 3, dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .view(tokens, 1, 3),
    ]
    rank_slots = [torch.arange(rank, tokens, world) for rank in range(world)]
    gathered = torch.cat([_extract_shard(caches, slots) for slots in rank_slots], dim=2)
    full = _interleave(gathered, world, 1, tokens)

    destination = [torch.zeros_like(cache) for cache in caches]
    for rank, slots in enumerate(rank_slots):
        shard = _deinterleave(full, world, rank, 1, len(slots))
        _scatter_shard(destination, slots, shard)
    for actual, expected in zip(destination, caches, strict=True):
        torch.testing.assert_close(actual, expected)


class _CountingGroup:
    def __init__(self):
        self.all_gather_calls = 0

    def all_gather(self, shard, dim):
        self.all_gather_calls += 1
        return torch.zeros(
            1, 1, shard.shape[dim] * 4, shard.shape[-1], dtype=shard.dtype
        )


class _Storage:
    def __init__(self, capacity):
        self.capacity = capacity
        self.puts = []

    def allocate(self, shape, dtype, fmt=None, busy_loop=False):
        if self.capacity <= 0:
            return None
        self.capacity -= 1
        obj = type("Alloc", (), {})()
        obj.tensor = torch.zeros(shape, dtype=dtype)
        return obj

    def batched_put(self, keys, mobjs, location=None):
        self.puts.append(len(keys))


def _store_with(monkeypatch, storage, n_chunks=6):
    # First Party
    import lmcache.integration.vllm.dcp_gather as dg

    group = _CountingGroup()
    chunk = 256
    monkeypatch.setattr(dg, "_shape_logged", True)
    monkeypatch.setattr(dg, "_dcp_group", lambda: (group, 4, 0))
    monkeypatch.setattr(dg, "_interleave_size", lambda: 1)
    monkeypatch.setattr(dg, "_packed_row_bytes", lambda kv: 8)
    monkeypatch.setattr(
        dg,
        "_extract_shard",
        lambda kv, rslots: torch.zeros(1, 1, chunk // 4, 8, dtype=torch.uint8),
    )
    monkeypatch.setattr(
        dg, "_interleave", lambda g, w, b, n: torch.zeros(1, 1, n, 8, dtype=torch.uint8)
    )

    engine = type("Engine", (), {})()
    engine.storage_manager = storage
    engine.fmt = "vllm"
    engine.store_location = None
    engine.config = type(
        "Cfg", (), {"get_extra_config_value": staticmethod(lambda *a: False)}
    )()
    engine.token_database = type(
        "TokDB",
        (),
        {
            "process_tokens": staticmethod(
                lambda toks, mask=None: [
                    (i * chunk, (i + 1) * chunk, f"k{i}") for i in range(n_chunks)
                ]
            )
        },
    )()
    impl = type("Impl", (), {})()
    impl.lmcache_engine = engine

    kvcaches = [torch.zeros(1, 1, chunk, 8, dtype=torch.uint8)]
    dg._dcp_store(
        impl,
        torch.arange(n_chunks * chunk),
        None,
        kvcaches,
        torch.arange(n_chunks * chunk),
    )
    return group.all_gather_calls


def test_dcp_store_collective_count_is_rank_symmetric(monkeypatch):
    """rank 0 running out of CPU cache must not change the all_gather count.

    storage is non-None only on the global first rank (save_only_first_rank),
    so any early exit driven by allocate() returning None strands the other
    DCP ranks inside a collective and deadlocks the engine.
    """
    non_rank0 = _store_with(monkeypatch, None)
    rank0_roomy = _store_with(monkeypatch, _Storage(capacity=99))
    rank0_full = _store_with(monkeypatch, _Storage(capacity=2))

    assert non_rank0 == rank0_roomy == rank0_full == 6
