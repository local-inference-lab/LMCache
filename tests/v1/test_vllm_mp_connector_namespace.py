# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the deployment-wide cache namespace of the MP connector.

``LMCACHE_CACHE_NAMESPACE`` prefixes every request's ``cache_salt`` so keys
written by a server with another namespace are unreachable; ``legacy``
keeps the empty salt; ``LMCACHE_REQUIRE_CACHE_NAMESPACE=1`` makes a missing
namespace a startup error.
"""

# Standard
from types import SimpleNamespace
import hashlib

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (
    CACHE_NAMESPACE_ENV,
    CACHE_NAMESPACE_REQUIRED_ENV,
    LMCacheMPRequestTracker,
    compose_cache_salt,
    resolve_cache_namespace,
)
from lmcache.v1.distributed.api import CACHE_SALT_MAX_LEN, ObjectKey


def test_unset_namespace_is_legacy_when_not_required(monkeypatch) -> None:
    monkeypatch.delenv(CACHE_NAMESPACE_ENV, raising=False)
    monkeypatch.delenv(CACHE_NAMESPACE_REQUIRED_ENV, raising=False)
    assert resolve_cache_namespace() == ""


def test_unset_namespace_is_an_error_when_required(monkeypatch) -> None:
    monkeypatch.delenv(CACHE_NAMESPACE_ENV, raising=False)
    monkeypatch.setenv(CACHE_NAMESPACE_REQUIRED_ENV, "1")
    with pytest.raises(ValueError, match="launcher must decide"):
        resolve_cache_namespace()


def test_legacy_literal_selects_empty_salt_even_when_required(monkeypatch) -> None:
    monkeypatch.setenv(CACHE_NAMESPACE_ENV, "legacy")
    monkeypatch.setenv(CACHE_NAMESPACE_REQUIRED_ENV, "1")
    assert resolve_cache_namespace() == ""


@pytest.mark.parametrize("bad", ["iso@1", "a/b", "back\\slash"])
def test_namespace_rejects_object_key_forbidden_characters(monkeypatch, bad) -> None:
    monkeypatch.setenv(CACHE_NAMESPACE_ENV, bad)
    with pytest.raises(ValueError, match="forbidden character"):
        resolve_cache_namespace()


def test_namespace_rejects_values_that_leave_no_room_for_a_request_salt(
    monkeypatch,
) -> None:
    monkeypatch.setenv(CACHE_NAMESPACE_ENV, "n" * (CACHE_SALT_MAX_LEN - 1))
    with pytest.raises(ValueError, match="too long"):
        resolve_cache_namespace()


def test_compose_without_namespace_keeps_request_salt() -> None:
    assert compose_cache_salt("", None) == ""
    assert compose_cache_salt("", "client-7") == "client-7"


def test_compose_with_namespace_prefixes_request_salt() -> None:
    assert compose_cache_salt("prod-a1b2", None) == "prod-a1b2"
    assert compose_cache_salt("prod-a1b2", "client-7") == "prod-a1b2.client-7"


def test_compose_hashes_overlong_request_salt_and_stays_a_valid_key() -> None:
    namespace = "iso-0123456789abcdef"
    long_salt = "x" * CACHE_SALT_MAX_LEN
    composed = compose_cache_salt(namespace, long_salt)
    expected = f"{namespace}.{hashlib.sha256(long_salt.encode()).hexdigest()}"
    assert composed == expected
    # The composed salt must be accepted by ObjectKey.
    ObjectKey(chunk_hash=b"\x01" * 32, model_name="/model", kv_rank=0, cache_salt=composed)
    other = compose_cache_salt(namespace, "y" * CACHE_SALT_MAX_LEN)
    assert other != composed


def test_tracker_applies_namespace_to_request_salt() -> None:
    request = SimpleNamespace(
        request_id="req-1", cache_salt="client-7", all_token_ids=[1, 2, 3]
    )
    assert LMCacheMPRequestTracker(request).cache_salt == "client-7"
    tracker = LMCacheMPRequestTracker(request, namespace="prod-a1b2")
    assert tracker.cache_salt == "prod-a1b2.client-7"
    unsalted = SimpleNamespace(request_id="req-2", cache_salt=None, all_token_ids=[1])
    assert LMCacheMPRequestTracker(unsalted, namespace="prod-a1b2").cache_salt == (
        "prod-a1b2"
    )
