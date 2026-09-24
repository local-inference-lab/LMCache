# SPDX-License-Identifier: Apache-2.0
"""
Decide which recurrent checkpoint pages deserve L1 memory and L2 storage.

A conversation publishes new request-boundary checkpoints every turn, and
each one supersedes the previous turn's endpoint state. Pages shared with the
newer checkpoint stay alive through it, so only the pages unique to the older
checkpoint (recurrent endpoint state, a partial attention page, auxiliary
state) become dead weight. This module keeps that knowledge:

* Superseded pages are dropped first from L1 and L2 and are never written to
  L2 when they leave L1.
* With write-on-evict storage, a current checkpoint page is written to L2
  once, when L1 is about to evict it, instead of on every request.
* L2 residency per adapter is tracked so a page that already reached L2 is
  not written again.

Everything is bounded in memory and safe to call from the store, eviction and
request-handler threads. Forgetting an entry only costs an extra write or a
later eviction; correctness never depends on this state, because restores
validate every page and fall back to a shorter checkpoint when one is gone.
"""

# Standard
from collections import OrderedDict
from collections.abc import Callable, Iterable
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey, is_recurrent_checkpoint_key
from lmcache.v1.distributed.internal_api import L2AdapterListener

logger = init_logger(__name__)


class _BoundedSet:
    """Insertion-ordered set that forgets its oldest entries beyond a bound."""

    def __init__(self, bound: int) -> None:
        self._bound = bound
        self._items: OrderedDict[object, None] = OrderedDict()

    def add(self, item: object) -> None:
        self._items[item] = None
        self._items.move_to_end(item)
        while len(self._items) > self._bound:
            self._items.popitem(last=False)

    def discard(self, item: object) -> None:
        self._items.pop(item, None)

    def __contains__(self, item: object) -> bool:
        return item in self._items

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> list:
        return list(self._items)


class _AdapterResidencyListener(L2AdapterListener):
    """Record which checkpoint pages one L2 adapter holds."""

    def __init__(self, retention: "CheckpointRetention", adapter_id: int) -> None:
        self._retention = retention
        self._adapter_id = adapter_id

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        self._retention.record_l2_present(self._adapter_id, keys, sizes)

    def on_l2_keys_accessed(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l2_keys_deleted(self, keys: list[ObjectKey]) -> None:
        self._retention.record_l2_absent(self._adapter_id, keys)


class CheckpointRetention:
    """
    Track superseded and persisted recurrent checkpoint pages.

    Args:
        write_on_evict: Whether current checkpoint pages are written to L2
            only when L1 evicts them. Without it, checkpoint pages follow the
            store policy and this object only orders eviction.
        persist: Callback that asynchronously stores keys to L2; installed by
            the storage manager once its store controller exists.
        max_superseded: Bound on remembered superseded pages.
        max_tracked: Bound on remembered L2-resident pages per adapter.
        persist_timeout: Seconds after which a page whose write never
            completed may be evicted from L1 without reaching L2.
    """

    def __init__(
        self,
        *,
        write_on_evict: bool = False,
        persist: Callable[[list[ObjectKey]], None] | None = None,
        max_superseded: int = 262144,
        max_tracked: int = 1048576,
        max_generations: int = 65536,
        persist_timeout: float = 120.0,
    ) -> None:
        self._write_on_evict = write_on_evict
        self._persist = persist
        self._max_tracked = max_tracked
        self._persist_timeout = persist_timeout
        self._lock = threading.Lock()
        self._superseded = _BoundedSet(max_superseded)
        self._superseded_generations = _BoundedSet(max_generations)
        # adapter id -> key -> size in bytes
        self._resident: dict[int, OrderedDict[ObjectKey, int]] = {}
        # key -> monotonic time its write was requested
        self._pending: dict[ObjectKey, float] = {}
        self._stats = {
            "superseded_checkpoints": 0,
            "superseded_pages": 0,
            "l1_superseded_drops": 0,
            "l2_superseded_evictions": 0,
            "l2_superseded_eviction_bytes": 0,
            "write_on_evict_requests": 0,
            "write_on_evict_persisted": 0,
            "write_on_evict_timeouts": 0,
        }

    @property
    def write_on_evict(self) -> bool:
        return self._write_on_evict

    def set_persist(self, persist: Callable[[list[ObjectKey]], None]) -> None:
        """Install the asynchronous L2 store callback."""
        self._persist = persist

    def listener_for(self, adapter_id: int) -> L2AdapterListener:
        """Return a listener that records one adapter's checkpoint pages."""
        with self._lock:
            self._resident.setdefault(adapter_id, OrderedDict())
        return _AdapterResidencyListener(self, adapter_id)

    def forget_adapter(self, adapter_id: int) -> None:
        with self._lock:
            self._resident.pop(adapter_id, None)

    # ----- supersession ---------------------------------------------------

    def is_superseded_generation(self, generation: str) -> bool:
        with self._lock:
            return generation in self._superseded_generations

    def mark_superseded(self, generation: str, keys: Iterable[ObjectKey]) -> int:
        """Mark one older checkpoint's unique pages as superseded.

        Args:
            generation: The older checkpoint's generation, remembered so the
                same ancestor is not processed again.
            keys: Pages that no current checkpoint references.

        Returns:
            Number of pages newly marked.
        """
        added = 0
        with self._lock:
            self._superseded_generations.add(generation)
            for key in keys:
                if key not in self._superseded:
                    added += 1
                self._superseded.add(key)
            self._stats["superseded_checkpoints"] += 1
            self._stats["superseded_pages"] += added
        return added

    def mark_current(self, keys: Iterable[ObjectKey]) -> None:
        """Clear the superseded mark from pages a new checkpoint references."""
        with self._lock:
            for key in keys:
                self._superseded.discard(key)

    def is_superseded(self, key: ObjectKey) -> bool:
        with self._lock:
            return key in self._superseded

    def superseded_keys(self) -> list[ObjectKey]:
        with self._lock:
            return self._superseded.items()

    # ----- L2 residency ---------------------------------------------------

    def record_l2_present(
        self, adapter_id: int, keys: list[ObjectKey], sizes: list[int]
    ) -> None:
        with self._lock:
            resident = self._resident.setdefault(adapter_id, OrderedDict())
            for key, size in zip(keys, sizes, strict=False):
                if not is_recurrent_checkpoint_key(key):
                    continue
                resident[key] = size
                resident.move_to_end(key)
                if self._pending.pop(key, None) is not None:
                    self._stats["write_on_evict_persisted"] += 1
            while len(resident) > self._max_tracked:
                resident.popitem(last=False)

    def record_l2_absent(self, adapter_id: int, keys: list[ObjectKey]) -> None:
        with self._lock:
            resident = self._resident.get(adapter_id)
            if resident is None:
                return
            for key in keys:
                resident.pop(key, None)

    def is_l2_resident(self, key: ObjectKey) -> bool:
        with self._lock:
            return any(key in resident for resident in self._resident.values())

    def superseded_in_adapter(
        self, adapter_id: int, max_bytes: int
    ) -> tuple[list[ObjectKey], int]:
        """Return superseded pages one adapter holds, up to ``max_bytes``."""
        victims: list[ObjectKey] = []
        total = 0
        with self._lock:
            resident = self._resident.get(adapter_id, {})
            for key in self._superseded.items():
                size = resident.get(key)
                if size is None:
                    continue
                victims.append(key)
                total += size
                if total >= max_bytes:
                    break
        return victims, total

    def record_l2_superseded_evictions(self, count: int, size: int) -> None:
        with self._lock:
            self._stats["l2_superseded_evictions"] += count
            self._stats["l2_superseded_eviction_bytes"] += size

    def record_l1_superseded_drops(self, count: int) -> None:
        with self._lock:
            self._stats["l1_superseded_drops"] += count

    # ----- write on evict -------------------------------------------------

    def needs_persist_before_evict(self, key: ObjectKey) -> bool:
        """Whether L1 must keep ``key`` until its L2 write completes.

        True for a current checkpoint page that is not in L2 yet, while its
        write is still expected to finish. A superseded page, an ordinary KV
        chunk or a page whose write timed out may be evicted.
        """
        if not self._write_on_evict or not is_recurrent_checkpoint_key(key):
            return False
        now = time.monotonic()
        with self._lock:
            if key in self._superseded:
                return False
            if any(key in resident for resident in self._resident.values()):
                self._pending.pop(key, None)
                return False
            requested = self._pending.get(key)
            if requested is not None and now - requested > self._persist_timeout:
                del self._pending[key]
                self._stats["write_on_evict_timeouts"] += 1
                logger.warning(
                    "Checkpoint page write to L2 did not complete within %.0f s; "
                    "evicting it from L1 without an L2 copy",
                    self._persist_timeout,
                )
                return False
            return True

    def request_persist(self, keys: list[ObjectKey]) -> int:
        """Ask the store controller to write pages not already requested."""
        now = time.monotonic()
        with self._lock:
            fresh = [key for key in keys if key not in self._pending]
            for key in fresh:
                self._pending[key] = now
            self._stats["write_on_evict_requests"] += len(fresh)
        if fresh and self._persist is not None:
            self._persist(fresh)
        return len(fresh)

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def l2_checkpoint_bytes(self) -> int:
        """Bytes of tracked checkpoint pages held across L2 adapters."""
        with self._lock:
            return sum(sum(resident.values()) for resident in self._resident.values())

    def observations(self) -> list[tuple[int | float, dict[str, object]]]:
        """Counters and sizes in OTel-observation shape, one per ``stat``."""
        status = self.report_status()
        status.pop("write_on_evict")
        status["l2_checkpoint_bytes"] = self.l2_checkpoint_bytes()
        return [(value, {"stat": name}) for name, value in status.items()]

    def report_status(self) -> dict:
        with self._lock:
            return {
                "write_on_evict": self._write_on_evict,
                "superseded_pages_tracked": len(self._superseded),
                "l2_resident_pages_tracked": sum(
                    len(resident) for resident in self._resident.values()
                ),
                "write_on_evict_pending": len(self._pending),
                **self._stats,
            }
