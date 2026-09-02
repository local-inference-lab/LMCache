# SPDX-License-Identifier: Apache-2.0
"""Restart accounting behavior shared by native L2 connectors."""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    NativeConnectorL2Adapter,
)
from tests.v1.distributed.test_native_connector_l2_adapter import (
    MockNativeConnector,
    create_object_key,
)


@pytest.fixture
def restart_adapter():
    """Return the shared native-connector adapter fixture."""
    adapter = NativeConnectorL2Adapter(MockNativeConnector())
    yield adapter
    adapter.close()


class _RecordingL2Listener:
    def __init__(self):
        self.stored = []

    def on_l2_keys_stored(self, keys, sizes) -> None:
        self.stored.append((list(keys), list(sizes)))


class TestPrimeExistingKeys:
    def test_priming_accounts_deduplicated_usage(self, restart_adapter):
        key = create_object_key(1)

        restart_adapter.prime_existing_keys([key, key], [100, 999])

        assert restart_adapter.get_usage().total_bytes_used == 100
        assert restart_adapter._key_sizes == {key: 100}

    def test_first_listener_receives_startup_snapshot_once(self, restart_adapter):
        keys = [create_object_key(1), create_object_key(2)]
        restart_adapter.prime_existing_keys(keys, [100, 200])
        first = _RecordingL2Listener()
        second = _RecordingL2Listener()

        restart_adapter.register_listener(first)
        restart_adapter.register_listener(second)

        # The snapshot is age-ordered (oldest first); the replay is reversed
        # so a policy that ranks a batch's later entries as earlier eviction
        # victims seeds the oldest object as least recently used.
        assert first.stored == [(keys[::-1], [200, 100])]
        assert second.stored == []
        assert restart_adapter.get_usage().total_bytes_used == 300

    def test_startup_snapshot_seeds_lru_oldest_first(self, restart_adapter):
        # First Party
        from lmcache.v1.distributed.eviction import L2EvictionPolicy
        from lmcache.v1.distributed.eviction_policy.lru import LRUEvictionPolicy

        oldest, middle, newest = (create_object_key(i) for i in (1, 2, 3))
        restart_adapter.prime_existing_keys([oldest, middle, newest], [1, 1, 1])
        policy = LRUEvictionPolicy()

        restart_adapter.register_listener(L2EvictionPolicy(policy))
        actions = policy.get_eviction_actions(2 / 3)

        evicted = [key for action in actions for key in action.keys]
        assert evicted == [oldest, middle]

    def test_empty_snapshot_is_consumed_without_callback(self, restart_adapter):
        restart_adapter.prime_existing_keys([], [])
        listener = _RecordingL2Listener()

        restart_adapter.register_listener(listener)

        assert listener.stored == []

    def test_rejects_invalid_or_repeated_priming(self, restart_adapter):
        with pytest.raises(ValueError, match="length mismatch"):
            restart_adapter.prime_existing_keys([create_object_key(1)], [1, 2])
        with pytest.raises(ValueError, match="must be positive"):
            restart_adapter.prime_existing_keys([create_object_key(1)], [0])

        restart_adapter.prime_existing_keys([], [])
        with pytest.raises(RuntimeError, match="already primed"):
            restart_adapter.prime_existing_keys([], [])

    def test_rejects_priming_after_listener_registration(self, restart_adapter):
        restart_adapter.register_listener(_RecordingL2Listener())

        with pytest.raises(RuntimeError, match="before listeners"):
            restart_adapter.prime_existing_keys([], [])
