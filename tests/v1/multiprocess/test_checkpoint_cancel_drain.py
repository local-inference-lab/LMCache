# SPDX-License-Identifier: Apache-2.0
"""A restore whose storage lookup never answers misses; it is never fatal.

A pending retrieve lookup exposed no SHM slots to its worker, so no GPU copy
can touch its pages. When it is cancelled and storage still does not answer,
the worker reports a miss and keeps serving, and the server releases the
lookup once storage answers.
"""

# Standard
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
import threading
import time
import uuid

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.multiprocess.checkpoint_storage import (
    CheckpointSlots,
    checkpoint_object_keys,
)
from lmcache.v1.multiprocess.checkpoint_transfer import (
    CheckpointTransferJob,
    CheckpointTransferWorker,
    UnsafeCheckpointCopyError,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules.checkpoint import CheckpointModule
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.protocols.checkpoint import CheckpointLeaseResponse
from tests.v1.multiprocess.test_checkpoint_storage import (
    make_manifest,
    open_checkpoint_rpc,
    open_store,
    poll,
    publish_all,
)


class _Reply:
    def __init__(self, value: Any = None, error: BaseException | None = None):
        self.value = value
        self.error = error

    def result(self, timeout: float | None = None) -> Any:
        if self.error is not None:
            raise self.error
        return self.value


class _StuckStorageClient:
    """Storage never answers a lookup, not even after it is cancelled."""

    def __init__(
        self,
        cancel_error: BaseException | None = None,
        poll_error: BaseException | None = None,
    ) -> None:
        self.cancel_error = cancel_error
        self.poll_error = poll_error
        self.requests: list[RequestType] = []

    def submit_request(self, kind: RequestType, payload: list[Any]) -> _Reply:
        self.requests.append(kind)
        if kind == RequestType.CHECKPOINT_CANCEL_RETRIEVE:
            return _Reply(True, self.cancel_error)
        if kind == RequestType.CHECKPOINT_POLL_RETRIEVE and self.poll_error:
            return _Reply(error=self.poll_error)
        return _Reply(CheckpointLeaseResponse("pending", "lease"))


def _stuck_worker(client: _StuckStorageClient) -> CheckpointTransferWorker:
    return CheckpointTransferWorker(
        cast(MessageQueueClient, client),
        lambda job, lease: pytest.fail("a lookup without slots must not copy"),
        rpc_timeout=0.1,
    )


def _retrieve(rank: int = 0) -> CheckpointTransferJob:
    return CheckpointTransferJob(
        replace(make_manifest(), world_size=1), rank, "RETRIEVE", ()
    )


def test_lookup_that_never_drains_after_cancel_is_a_miss() -> None:
    client = _StuckStorageClient()
    worker = _stuck_worker(client)
    try:
        for _ in range(2):
            started = time.monotonic()
            completion = worker.submit(_retrieve())
            # The worker keeps admitting transfers after the first abandon.
            assert completion is not None
            assert completion.result(timeout=10) is False
            assert time.monotonic() - started < 5
    finally:
        worker.close()
    assert client.requests.count(RequestType.CHECKPOINT_CANCEL_RETRIEVE) == 2


@pytest.mark.parametrize(
    "error",
    [RuntimeError("cancel handler failed"), LMCacheTimeoutError("no reply")],
    ids=["error", "timeout"],
)
def test_failed_cancel_of_a_pending_lookup_is_a_miss(error: BaseException) -> None:
    client = _StuckStorageClient(cancel_error=error)
    worker = _stuck_worker(client)
    try:
        completion = worker.submit(_retrieve())
        assert completion is not None
        assert completion.result(timeout=10) is False
        again = worker.submit(_retrieve())
        assert again is not None and again.result(timeout=10) is False
    finally:
        worker.close()


def test_failed_poll_of_a_pending_lookup_is_not_fatal() -> None:
    client = _StuckStorageClient(poll_error=RuntimeError("poll handler failed"))
    worker = _stuck_worker(client)
    try:
        completion = worker.submit(_retrieve())
        assert completion is not None
        with pytest.raises(RuntimeError, match="poll handler failed") as failure:
            completion.result(timeout=10)
        assert not isinstance(failure.value, UnsafeCheckpointCopyError)
        assert RequestType.CHECKPOINT_CANCEL_RETRIEVE in client.requests
        assert worker.submit(_retrieve()) is not None
    finally:
        worker.close()


def _stall_lookups(monkeypatch: pytest.MonkeyPatch, storage: Any) -> threading.Event:
    """Keep every prefetch pending until the returned event is set."""
    answer = threading.Event()
    query = storage.query_prefetch_status_detailed
    monkeypatch.setattr(
        storage,
        "query_prefetch_status_detailed",
        lambda handle: query(handle) if answer.is_set() else None,
    )
    return answer


def test_cancelled_lookup_is_released_when_storage_answers_late(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine keeps serving and restores again once storage answers."""
    with open_checkpoint_rpc() as (client, module, _mapping, _name):
        storage = module.context.storage_manager
        entry = replace(make_manifest(), world_size=1)
        assert module.begin(entry)
        stored = module.prepare_store(entry, 0)
        assert stored.status == "ready"
        assert module.finish_store(stored.lease_id, True)
        answer = _stall_lookups(monkeypatch, storage)
        copies: list[str] = []
        worker = CheckpointTransferWorker(
            client, lambda job, lease: copies.append(lease.lease_id), rpc_timeout=0.2
        )
        job = CheckpointTransferJob(entry, 0, "RETRIEVE", ())
        try:
            stalled = worker.submit(job)
            assert stalled is not None and stalled.result(timeout=10) is False
            assert not copies
            status = module.report_status()["recurrent_checkpoints"]
            assert status["retrieve_leases"] == status["cancelled_lookups"] == 1
            answer.set()
            restored = worker.submit(job)
            assert restored is not None and restored.result(timeout=10) is True
            assert len(copies) == 1
            status = module.report_status()["recurrent_checkpoints"]
            assert status["retrieve_leases"] == status["cancelled_lookups"] == 0
            assert module.find((entry.prefix,)) == entry
            keys = [key for group in checkpoint_object_keys(entry, 0) for key in group]
            assert storage.delete_l1_keys(keys)[0] == len(keys)
        finally:
            answer.set()
            worker.close()


def test_released_cancelled_lookup_answers_its_late_poll_with_a_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with open_store() as (service, index, storage, mapping):
        entry = make_manifest()
        publish_all(service, index, mapping, entry)
        answer = _stall_lookups(monkeypatch, storage)
        cancelled = service.begin_retrieve(entry, 0)
        other = service.begin_retrieve(entry, 1)
        assert cancelled is not None and other is not None
        service.cancel_retrieve(cancelled)
        assert service.poll_retrieve(cancelled) is None
        assert service.report_status()["cancelled_lookups"] == 1
        answer.set()
        # Polling another lease releases the cancelled lookup as well.
        assert isinstance(poll(service, other), CheckpointSlots)
        assert service.report_status()["cancelled_lookups"] == 0
        assert service.poll_retrieve(cancelled) is False
        with pytest.raises(KeyError):
            service.poll_retrieve(cancelled)
        service.cancel_retrieve(cancelled)
        service.cancel_retrieve(uuid.uuid4().hex)
        service.finish_retrieve(other)
        assert service.report_status()["retrieve_leases"] == 0
        assert index.find((entry.prefix,)) == entry
        keys = [key for group in checkpoint_object_keys(entry, 0) for key in group]
        assert storage.delete_l1_keys(keys)[0] == len(keys)


def test_close_waits_only_for_leases_that_exposed_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = f"lmcache_l1_pool_checkpoint_close_{uuid.uuid4().hex}"
    with open_store(shm_name=name) as (_service, _index, storage, _mapping):
        module = CheckpointModule(
            cast(
                MPCacheServerContext,
                SimpleNamespace(
                    storage_manager=storage,
                    shm_pool_info={"shm_name": name, "pool_size": 4 * 1024 * 1024},
                ),
            )
        )
        entry = replace(make_manifest(), world_size=1)
        assert module.begin(entry)
        stored = module.prepare_store(entry, 0)
        assert module.finish_store(stored.lease_id, True)
        lookup = module.begin_retrieve(entry, 0)
        deadline = time.monotonic() + 5
        ready = module.poll_retrieve(lookup.lease_id)
        while ready.status == "pending" and time.monotonic() < deadline:
            ready = module.poll_retrieve(lookup.lease_id)
        assert ready.status == "ready"
        with pytest.raises(RuntimeError, match="must drain"):
            module.close()
        assert module.finish_retrieve(ready.lease_id)
        answer = _stall_lookups(monkeypatch, storage)
        try:
            cancelled = module.begin_retrieve(entry, 0)
            assert module.cancel_retrieve(cancelled.lease_id)
            assert module.begin_retrieve(entry, 0).status == "pending"
            status = module.report_status()["recurrent_checkpoints"]
            assert status["retrieve_lookups"] == 2
            assert status["cancelled_lookups"] == 1
            module.close()
        finally:
            answer.set()
