# SPDX-License-Identifier: Apache-2.0
"""Shared-memory capacity accounting for engine-driven cache restarts."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess import server


@pytest.mark.parametrize(
    ("configured_name", "expected_path"),
    [
        ("restart-test", "/dev/shm/lmcache_l1_pool_restart-test"),
        (
            "lmcache_l1_pool_restart-test",
            "/dev/shm/lmcache_l1_pool_restart-test",
        ),
        (
            "/lmcache_l1_pool_restart-test",
            "/dev/shm/lmcache_l1_pool_restart-test",
        ),
    ],
)
def test_unlink_configured_l1_shm_uses_allocator_name(
    monkeypatch: pytest.MonkeyPatch,
    configured_name: str,
    expected_path: str,
) -> None:
    """Capacity checks remove the segment that the allocator will replace."""
    observed: list[str] = []

    def _unlink(shm_name: str) -> None:
        observed.append(shm_name)

    monkeypatch.setattr(server, "_unlink_stale_shm", _unlink)

    server._unlink_configured_l1_shm(configured_name)
    assert observed == [expected_path.removeprefix("/dev/shm/")]


@pytest.mark.parametrize("configured_name", ["nested/pool", "nested\\pool"])
def test_unlink_configured_l1_shm_passes_path_validation_to_allocator(
    monkeypatch: pytest.MonkeyPatch,
    configured_name: str,
) -> None:
    """The allocator cleanup remains the authority for name validation."""
    observed: list[str] = []

    monkeypatch.setattr(server, "_unlink_stale_shm", observed.append)

    server._unlink_configured_l1_shm(configured_name)
    assert observed == [f"lmcache_l1_pool_{configured_name}"]


def test_available_l1_shm_bytes_is_measured_after_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The capacity value reflects tmpfs state after stale-pool cleanup."""
    calls: list[str] = []
    monkeypatch.setattr(
        server,
        "_unlink_configured_l1_shm",
        lambda shm_name: calls.append(f"unlink:{shm_name}"),
    )

    def _disk_usage(path: str) -> SimpleNamespace:
        calls.append(f"usage:{path}")
        return SimpleNamespace(free=70 * 1024**3)

    monkeypatch.setattr(
        server.shutil,
        "disk_usage",
        _disk_usage,
    )

    available = server._available_l1_shm_bytes_after_cleanup("restart-test")

    assert available == 70 * 1024**3
    assert calls == ["unlink:restart-test", "usage:/dev/shm"]


@pytest.mark.parametrize(
    ("use_lazy", "devdax_path"),
    [(True, None), (False, "/dev/dax0.0")],
)
def test_non_posix_l1_modes_do_not_unlink_named_shm(
    monkeypatch: pytest.MonkeyPatch,
    use_lazy: bool,
    devdax_path: str | None,
) -> None:
    """Restart cleanup is limited to configurations that replace POSIX SHM."""
    calls: list[str] = []
    monkeypatch.setattr(
        server,
        "_available_l1_shm_bytes_after_cleanup",
        lambda shm_name: calls.append(shm_name) or 1,
    )
    mem_cfg = SimpleNamespace(
        shm_name="pool-owned-by-another-process",
        use_lazy=use_lazy,
        devdax_path=devdax_path,
    )

    assert server._available_configured_l1_shm_bytes(mem_cfg) is None
    assert calls == []
