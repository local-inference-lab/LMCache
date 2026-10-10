# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for selecting coherent distributed-cache eviction victims."""

# Standard
from collections import OrderedDict
from collections.abc import Callable, Collection, Iterable
import time

# First Party
from lmcache.v1.distributed.api import ObjectKey

KVTopology = tuple[int, int] | None
TopologyNamespace = tuple[str, KVTopology]
ChunkFamilyId = tuple[bytes, str, str, KVTopology]

# How long a chunk family missing a rank/group sibling waits for it before
# its present members become evictable together. Per-rank store batches land
# within milliseconds of each other, while some families stay partial for
# good: a prefix reloaded from L2 brings sliding-window groups back only for
# its final window.
INCOMPLETE_FAMILY_GRACE_SECONDS = 30.0


def _kv_topology(kv_rank: int) -> KVTopology:
    """Decode the stable world/local-world topology from a packed KV rank."""
    world_size = (kv_rank >> 24) & 0xFF
    global_rank = (kv_rank >> 16) & 0xFF
    local_world_size = (kv_rank >> 8) & 0xFF
    local_rank = kv_rank & 0xFF
    if (
        world_size > 0
        and local_world_size > 0
        and global_rank < world_size
        and local_rank < local_world_size
        and global_rank % local_world_size == local_rank
    ):
        return world_size, local_world_size
    # Tests and legacy callers may use an unpacked integer rank. Preserve
    # their observed-coordinate behavior rather than guessing a topology.
    return None


def _topology_namespace(key: ObjectKey) -> TopologyNamespace:
    """Return the cache-model and parallel-layout namespace for ``key``."""
    return key.model_name, _kv_topology(key.kv_rank)


def _logical_chunk_family(
    key: ObjectKey,
) -> ChunkFamilyId:
    """Return the identity shared by every rank/group object for one chunk."""
    return (
        key.chunk_hash,
        key.model_name,
        key.cache_salt,
        _kv_topology(key.kv_rank),
    )


class ChunkFamilyTopology:
    """Track the observed rank/group coordinates of logical chunk families.

    Eviction policies already serialize key lifecycle notifications and victim
    selection under their own lock. Recording only the small topology, plus
    the families that gained a member within the grace period, keeps selection
    proportional to the number of victim families without rebuilding or
    retaining a second full-cache map.
    """

    def __init__(
        self,
        incomplete_family_grace_seconds: float = INCOMPLETE_FAMILY_GRACE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create an empty topology.

        Args:
            incomplete_family_grace_seconds: How long a family missing an
                expected rank/group sibling waits for it before its present
                members become evictable together.
            clock: Monotonic time source in seconds.

        Raises:
            ValueError: If ``incomplete_family_grace_seconds`` is negative.
        """
        if incomplete_family_grace_seconds < 0:
            raise ValueError(
                "incomplete_family_grace_seconds must be non-negative, got "
                f"{incomplete_family_grace_seconds}"
            )
        self._coordinates: dict[TopologyNamespace, set[tuple[int, int]]] = {}
        self._ordered_coordinates: dict[
            TopologyNamespace,
            tuple[tuple[int, int], ...],
        ] = {}
        self._grace_seconds = incomplete_family_grace_seconds
        self._clock = clock
        # Families that gained a member within the grace period, oldest
        # arrival first, so the map stays bounded by the recent store rate
        # rather than by the cache size.
        self._recent_families: OrderedDict[ChunkFamilyId, float] = OrderedDict()

    def observe(self, keys: Iterable[ObjectKey]) -> None:
        """Record the coordinates of new keys and when their families grew.

        Args:
            keys: Keys that just became tracked by the calling policy.
        """
        now = self._clock()
        self._expire_recent_families(now)
        changed_namespaces = set()
        for key in keys:
            family_id = _logical_chunk_family(key)
            self._recent_families[family_id] = now
            self._recent_families.move_to_end(family_id)
            namespace = _topology_namespace(key)
            coordinates = self._coordinates.setdefault(namespace, set())
            previous_count = len(coordinates)
            coordinates.add((key.kv_rank, key.object_group_id))
            if len(coordinates) != previous_count:
                changed_namespaces.add(namespace)
        for namespace in changed_namespaces:
            observed = self._coordinates[namespace]
            topology = namespace[1]
            if topology is None:
                expected = observed
            else:
                world_size, local_world_size = topology
                group_ids = {object_group_id for _, object_group_id in observed}
                expected = {
                    (
                        ObjectKey.ComputeKVRank(
                            world_size=world_size,
                            global_rank=global_rank,
                            local_world_size=local_world_size,
                            local_rank=global_rank % local_world_size,
                        ),
                        object_group_id,
                    )
                    for global_rank in range(world_size)
                    for object_group_id in group_ids
                }
            self._ordered_coordinates[namespace] = tuple(sorted(expected))

    def _expire_recent_families(self, now: float) -> None:
        """Drop arrival times that are older than the grace period."""
        recent = self._recent_families
        while recent:
            family_id, updated = next(iter(recent.items()))
            if now - updated < self._grace_seconds:
                return
            del recent[family_id]

    def members(
        self,
        key: ObjectKey,
        tracked_keys: Collection[ObjectKey],
    ) -> tuple[ObjectKey, ...]:
        """Return the tracked members of ``key``'s family, or empty to wait.

        A complete family is returned at once. A family missing an expected
        rank/group sibling waits until the grace period has passed since it
        last gained a member, because a late store may still complete it;
        after that its present members are returned together. Families can
        stay partial for good: a prefix reloaded from L2 brings sliding-window
        groups back only for its final window, and a store can fail on one
        rank. Waiting for such siblings forever would leave the present
        members unevictable.

        Args:
            key: Any tracked member of the family.
            tracked_keys: Keys currently tracked by the calling policy.

        Returns:
            The family's tracked members in coordinate order, or an empty
            tuple while an incomplete family is within its grace period.
        """
        coordinates = self._ordered_coordinates.get(_topology_namespace(key), ())
        if coordinates == ((key.kv_rank, key.object_group_id),):
            return (key,)

        members = []
        incomplete = False
        for kv_rank, object_group_id in coordinates:
            candidate = ObjectKey(
                chunk_hash=key.chunk_hash,
                model_name=key.model_name,
                kv_rank=kv_rank,
                object_group_id=object_group_id,
                cache_salt=key.cache_salt,
            )
            if candidate in tracked_keys:
                members.append(candidate)
            else:
                incomplete = True
        if incomplete:
            updated = self._recent_families.get(_logical_chunk_family(key))
            if updated is not None and self._clock() - updated < self._grace_seconds:
                # Stores complete asynchronously across rank/group batches.
                # Evicting the visible subset this early lets a late sibling
                # land after this pass as an orphan of an evicted family.
                return ()
        return tuple(members)


def select_chunk_coherent_victims(
    ordered_keys: Collection[ObjectKey],
    target_count: int,
    family_topology: ChunkFamilyTopology,
    key_eligible_filter: Callable[[ObjectKey], bool] | None = None,
) -> list[ObjectKey]:
    """Select LRU victims without splitting a logical chunk family.

    A distributed or hybrid chunk is persisted as multiple ``ObjectKey``
    instances that differ only by ``kv_rank`` and ``object_group_id``. Deleting
    an arbitrary subset makes a later lookup appear promising while the load
    cannot reconstruct any tokens. Once the LRU boundary selects one member,
    return every tracked member of that logical family.

    When an eligibility filter is present, a family is selected only if every
    member is eligible. This prevents eviction from splitting a family around
    a read/write-locked object. A family missing an expected sibling waits for
    the topology's grace period, then its present members are selected
    together. The returned count may exceed ``target_count`` by at most one
    family; eviction ratios are approximate by contract.

    Args:
        ordered_keys: Keys in least-to-most-recently-used order.
        target_count: Approximate number of object keys to select.
        family_topology: Rank/group coordinates observed by the calling
            eviction policy.
        key_eligible_filter: Optional per-key eligibility predicate.

    Returns:
        Selected keys in family/LRU order, or an empty list when no eligible
        family can be selected. An incomplete family is skipped while it is
        within its grace period.
    """
    if target_count <= 0:
        return []

    selected: list[ObjectKey] = []
    visited: set[ChunkFamilyId] = set()
    for key in ordered_keys:
        family_id = _logical_chunk_family(key)
        if family_id in visited:
            continue
        visited.add(family_id)
        members = family_topology.members(key, ordered_keys)
        if not members:
            continue
        if key_eligible_filter is not None and any(
            not key_eligible_filter(member) for member in members
        ):
            continue
        selected.extend(members)
        if len(selected) >= target_count:
            break
    return selected
