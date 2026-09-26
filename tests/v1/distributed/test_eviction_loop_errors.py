# SPDX-License-Identifier: Apache-2.0
"""An exception in one eviction pass must not stop eviction for good."""

# Standard
from types import SimpleNamespace
from typing import Any, Generic, TypeVar, cast
import threading
import time

# First Party
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.l2_adapters.base import AdapterUsage
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L1EvictionController,
    L2EvictionController,
)


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


UsageT = TypeVar("UsageT")


class FlakyUsage(Generic[UsageT]):
    """Raise on the first call, then report an idle tier."""

    def __init__(self, idle: UsageT) -> None:
        self.calls = 0
        self._idle = idle
        self._lock = threading.Lock()

    def __call__(self) -> UsageT:
        with self._lock:
            self.calls += 1
            first = self.calls == 1
        if first:
            raise RuntimeError("transient failure")
        return self._idle


def test_l1_eviction_loop_survives_a_failed_pass() -> None:
    usage = FlakyUsage((0, 100))
    manager = SimpleNamespace(
        register_listener=lambda listener: None,
        get_memory_usage=usage,
        reclaim_abandoned_writes=lambda: 0,
    )
    controller = L1EvictionController(
        cast(Any, manager), EvictionConfig(eviction_policy="LRU")
    )
    controller.start()
    try:
        assert wait_until(lambda: usage.calls >= 2)
        assert controller.report_status()["thread_alive"]
    finally:
        controller.stop()


def test_l2_eviction_loop_survives_a_failed_pass() -> None:
    usage = FlakyUsage(AdapterUsage(total_bytes_used=0, total_capacity_bytes=100))
    state = SimpleNamespace(
        adapter_id=0,
        adapter=SimpleNamespace(get_usage=usage),
        eviction_policy=SimpleNamespace(support_isolation=False),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )
    controller = L2EvictionController([cast(Any, state)])
    controller.start()
    try:
        assert wait_until(lambda: usage.calls >= 2)
        assert controller.report_status()["thread_alive"]
    finally:
        controller.stop()
