# SPDX-License-Identifier: Apache-2.0
"""Bounded drain tests for CheckpointTransferWorker.close().

``close()`` must stop waiting for copies that cannot drain: the executor
keeps draining in the background and the undrained case surfaces as
:class:`UnsafeCheckpointCopyError` instead of blocking worker teardown
forever.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock
import os
import threading
import time
import uuid

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.checkpoint_index import (
    CheckpointManifest,
    CheckpointPrefix,
)
from lmcache.v1.multiprocess.checkpoint_transfer import (
    CheckpointTransferJob,
    CheckpointTransferWorker,
    UnsafeCheckpointCopyError,
)


def _manifest() -> CheckpointManifest:
    return CheckpointManifest(
        uuid.uuid4().hex,
        CheckpointPrefix("weights-and-layout-and-salt", 4096, b"a" * 32, (1,)),
        1,
        b'{"schema_version":1,"payload_keys":["immutable-generation-pages"]}',
    )


def _ready_client() -> MagicMock:
    """A client stub whose every reply is a complete ready lease.

    close() exercises only the executor drain, so the metadata RPCs may
    answer instantly; the copy callback itself owns the drain behavior.
    """
    lease = SimpleNamespace(status="ready", lease_id="lease-1", slots=())
    client = MagicMock(name="client")
    client.submit_request.return_value.result.return_value = lease
    return client


def test_close_raises_when_copies_cannot_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An undraining copy must fail close() by the deadline, not block it.

    The executor is not cancelled: the blocked copy still runs to
    completion in the background and its completion future resolves once
    the caller releases it.
    """
    monkeypatch.setenv("LMCACHE_CHECKPOINT_CLOSE_TIMEOUT", "1")
    gate = threading.Event()

    def copy_pages(job: CheckpointTransferJob, lease: object) -> None:
        assert gate.wait(timeout=30)

    worker = CheckpointTransferWorker(_ready_client(), copy_pages, workers=1)
    completion = worker.submit(CheckpointTransferJob(_manifest(), 0, "STORE", ()))
    assert completion is not None
    outcome: dict[str, bool] = {}
    closer = threading.Thread(
        target=lambda: outcome.update(
            unsafe=_close_raises(worker),
        ),
        daemon=True,
    )
    closer.start()
    closer.join(timeout=10)
    assert outcome.get("unsafe"), (
        "close() neither raised UnsafeCheckpointCopyError nor returned "
        f"within the deadline (outcome={outcome}); it blocked on the "
        "undraining copy"
    )
    gate.set()
    assert completion.result(timeout=5) is True


def _close_raises(worker: CheckpointTransferWorker) -> bool:
    try:
        worker.close()
    except UnsafeCheckpointCopyError:
        return True
    return False


def test_close_returns_when_copies_drain_in_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A copy finishing inside the deadline drains and close() succeeds."""
    monkeypatch.setenv("LMCACHE_CHECKPOINT_CLOSE_TIMEOUT", "30")
    started = threading.Event()

    def copy_pages(job: CheckpointTransferJob, lease: object) -> None:
        started.set()

    worker = CheckpointTransferWorker(_ready_client(), copy_pages, workers=1)
    completion = worker.submit(CheckpointTransferJob(_manifest(), 0, "STORE", ()))
    assert completion is not None
    assert started.wait(timeout=5)
    worker.close()
    assert completion.result(timeout=5) is True


def test_drain_deadline_arms_a_force_exit_watchdog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past the hard ceiling the worker force-exits instead of hanging teardown.

    concurrent.futures' atexit hook joins the executor's non-daemon threads
    in threading._shutdown, so a copy stuck past close()'s deadline can
    block interpreter finalization forever. After the drain deadline is
    crossed, a watchdog must os._exit(1) with a banner unless teardown
    completes within the hard ceiling.
    """
    monkeypatch.setenv("LMCACHE_CHECKPOINT_CLOSE_TIMEOUT", "0.5")
    monkeypatch.setenv("LMCACHE_CHECKPOINT_FORCE_EXIT_CEILING", "0.3")
    gate = threading.Event()

    def copy_pages(job: CheckpointTransferJob, lease: object) -> None:
        assert gate.wait(timeout=30)

    exits: list[int] = []
    banners: list[bytes] = []
    monkeypatch.setattr(os, "_exit", exits.append)
    real_write = os.write

    def fake_write(fd: int, data: object) -> int:
        if fd == 2 and isinstance(data, bytes) and b"force-exit" in data:
            banners.append(data)
            return len(data)
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", fake_write)
    worker = CheckpointTransferWorker(_ready_client(), copy_pages, workers=1)
    completion = worker.submit(CheckpointTransferJob(_manifest(), 0, "STORE", ()))
    assert completion is not None
    outcome: dict[str, bool] = {}
    closer = threading.Thread(
        target=lambda: outcome.update(unsafe=_close_raises(worker)),
        daemon=True,
    )
    closer.start()
    closer.join(timeout=10)
    assert outcome.get("unsafe"), "close() did not raise by the drain deadline"
    deadline = time.monotonic() + 5
    while not exits and time.monotonic() < deadline:
        time.sleep(0.05)
    assert exits == [1], (
        "no force-exit fired within the hard ceiling after the drain "
        f"deadline (exits={exits})"
    )
    assert any(b"drain deadline" in b for b in banners)
    gate.set()
