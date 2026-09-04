# SPDX-License-Identifier: Apache-2.0
"""Source-level contracts for POSIX-FD cuMem CUDA IPC."""

# Standard
from pathlib import Path
import array
import os
import socket
import tempfile
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.cuda.cumem_ipc import (
    CuMemAllocationDescriptor,
    CuMemFDBroker,
    CuMemIPCUnsupportedError,
    ImportedCuMemMapping,
    ImportedCuMemRegistry,
    receive_fd,
    validate_broker_root,
)
from lmcache.v1.platform.cuda.ipc_wrapper import (
    CudaIPCWrapper,
    CuMemCudaIPCWrapper,
    RawCudaIPCWrapper,
)
from lmcache.v1.platform.kv_wrap import wrap_kv_caches
import lmcache.v1.platform.cuda.cumem_ipc as cumem_ipc
import lmcache.v1.platform.cuda.ipc_wrapper as ipc_wrapper
import lmcache.v1.platform.kv_wrap as kv_wrap


def _descriptor(
    *,
    allocation_id: str = "allocation",
    device_uuid: str = "device",
) -> CuMemAllocationDescriptor:
    return CuMemAllocationDescriptor(
        protocol_version=1,
        allocation_id=allocation_id,
        device_uuid=device_uuid,
        allocation_size=128,
        broker_path="/shared/lmcu.sock",
        broker_token="token",
        handle_type="posix_fd",
    )


class _RecordingImporter:
    """Record physical imports and closes without touching CUDA."""

    def __init__(self) -> None:
        self.imported: list[CuMemAllocationDescriptor] = []
        self.closed: list[ImportedCuMemMapping] = []

    def import_allocation(
        self, descriptor: CuMemAllocationDescriptor
    ) -> ImportedCuMemMapping:
        self.imported.append(descriptor)
        return ImportedCuMemMapping(0, descriptor.allocation_size, object(), 0)

    def close_allocation(self, mapping: ImportedCuMemMapping) -> None:
        self.closed.append(mapping)


def _wire_wrapper(
    descriptor: CuMemAllocationDescriptor,
    *,
    storage_nbytes: int = 64,
) -> CuMemCudaIPCWrapper:
    """Build a deserialized wrapper without exporter-side CUDA calls."""
    wrapper = CuMemCudaIPCWrapper.__new__(CuMemCudaIPCWrapper)
    wrapper.handle = descriptor
    wrapper.dtype = torch.int16
    wrapper.shape = (2, 2)
    wrapper.stride = (8, 2)
    wrapper.storage_offset = 1
    wrapper.device_uuid = descriptor.device_uuid
    wrapper.physical_storage_nbytes = storage_nbytes
    wrapper.allocation_storage_offset_bytes = 16
    wrapper._lease = None
    wrapper._mapping = None
    wrapper._tensor = None
    wrapper._closed = False
    return wrapper


def test_four_aliases_map_once_and_reregister() -> None:
    """Four views share one mapping, unregister it, then import afresh."""
    importer = _RecordingImporter()
    registry = ImportedCuMemRegistry(importer)
    descriptor = _descriptor()

    mappings = [registry.acquire(descriptor) for _ in range(4)]

    assert all(mapping is mappings[0] for mapping in mappings)
    assert len(importer.imported) == 1
    assert registry.registration_count == 1
    assert registry.alias_refcount_total == 4

    for _ in range(4):
        registry.release(descriptor)

    assert registry.registration_count == 0
    assert registry.alias_refcount_total == 0
    assert importer.closed == [mappings[0]]

    second_mapping = registry.acquire(descriptor)
    assert second_mapping is not mappings[0]
    assert len(importer.imported) == 2
    registry.release(descriptor)
    assert importer.closed == [mappings[0], second_mapping]


def test_allocation_identity_includes_device() -> None:
    """Equal allocation IDs on different devices must not alias mappings."""
    importer = _RecordingImporter()
    registry = ImportedCuMemRegistry(importer)

    first = registry.acquire(_descriptor(device_uuid="gpu-0"))
    second = registry.acquire(_descriptor(device_uuid="gpu-1"))

    assert first is not second
    assert registry.registration_count == 2


@pytest.mark.parametrize("wrapper_cls", [CudaIPCWrapper, RawCudaIPCWrapper])
def test_standard_and_isolated_factories_prefer_cumem(
    monkeypatch, wrapper_cls: type[CudaIPCWrapper] | type[RawCudaIPCWrapper]
) -> None:
    """Both CUDA factory selections avoid ordinary IPC for vLLM VMM memory."""
    tensor = torch.zeros(1)
    allocation = (100, 128, object())
    sentinel = object()
    monkeypatch.setattr(
        cumem_ipc,
        "find_cumem_allocation",
        lambda _tensor: allocation,
    )
    monkeypatch.setattr(
        ipc_wrapper,
        "CuMemCudaIPCWrapper",
        lambda wrapped, found: (
            sentinel if wrapped is tensor and found is allocation else None
        ),
    )

    assert wrapper_cls.wrap(tensor) is sentinel


def test_exact_offset_shape_stride_dtype_round_trip(monkeypatch) -> None:
    """Reconstruction preserves allocation/storage offsets and physical stride."""
    raw = torch.zeros(128, dtype=torch.uint8)

    class _Mapping:
        def as_torch_bytes(self) -> torch.Tensor:
            return raw

    descriptor = _descriptor()
    wrapper = _wire_wrapper(descriptor)

    mapping = _Mapping()
    monkeypatch.setattr(cumem_ipc, "acquire_imported_mapping", lambda _desc: mapping)
    monkeypatch.setattr(cumem_ipc, "release_imported_mapping", lambda _desc: None)

    tensor = wrapper.to_tensor()

    assert tensor.dtype == torch.int16
    assert tensor.shape == (2, 2)
    assert tensor.stride() == (8, 2)
    assert tensor.storage_offset() == 9
    tensor[1, 1] = 1234
    assert raw.view(torch.int16)[19].item() == 1234
    assert wrapper.to_tensor() is tensor


def test_partial_import_failure_releases_earlier_aliases(monkeypatch) -> None:
    """Malformed final metadata rolls back aliases acquired earlier."""
    raw = torch.zeros(128, dtype=torch.uint8)

    class _Mapping(ImportedCuMemMapping):
        def as_torch_bytes(self) -> torch.Tensor:
            return raw

    class _Importer(_RecordingImporter):
        def import_allocation(
            self, descriptor: CuMemAllocationDescriptor
        ) -> ImportedCuMemMapping:
            self.imported.append(descriptor)
            return _Mapping(0, descriptor.allocation_size, object(), 0)

    importer = _Importer()
    registry = ImportedCuMemRegistry(importer)
    descriptor = _descriptor()
    wrappers = [_wire_wrapper(descriptor) for _ in range(3)]
    wrappers.append(_wire_wrapper(descriptor, storage_nbytes=2))
    monkeypatch.setattr(cumem_ipc, "acquire_imported_mapping", registry.acquire)
    monkeypatch.setattr(cumem_ipc, "release_imported_mapping", registry.release)

    with pytest.raises(ValueError, match="physical storage extent"):
        try:
            for wrapper in wrappers:
                wrapper.to_tensor()
        finally:
            for wrapper in wrappers:
                wrapper.close()

    assert len(importer.imported) == 1
    assert len(importer.closed) == 1
    assert registry.registration_count == 0
    assert registry.alias_refcount_total == 0


class _ClosableWrapper:
    """Record partial wrapping cleanup."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_partial_wrap_failure_releases_earlier_alias(monkeypatch) -> None:
    """A later export failure closes wrappers already created in the batch."""
    first = _ClosableWrapper()
    calls = 0

    def wrap(_tensor: torch.Tensor) -> _ClosableWrapper:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise CuMemIPCUnsupportedError("not shareable")
        return first

    monkeypatch.setattr(kv_wrap, "wrap_one_kv_cache", wrap)

    with pytest.raises(CuMemIPCUnsupportedError, match="not shareable"):
        wrap_kv_caches(
            {
                "first": torch.zeros(1),
                "second": torch.zeros(1),
            }
        )

    assert first.closed


def test_broker_transfers_fd_only_under_shared_absolute_root(tmp_path: Path) -> None:
    """The broker passes an FD and rejects a path from another mount root."""
    root = validate_broker_root(tmp_path)
    broker = CuMemFDBroker(root)
    fd, path = tempfile.mkstemp(dir=root)
    os.unlink(path)
    os.write(fd, b"same-path-contract")
    lease = broker.register(("proof", 1), fd)
    try:
        received = receive_fd(
            lease.broker_path,
            lease.broker_token,
            lease.allocation_id,
            broker_root=root,
        )
        try:
            os.lseek(received, 0, os.SEEK_SET)
            assert os.read(received, 64) == b"same-path-contract"
        finally:
            os.close(received)

        other_root = root / "different-container-root"
        other_root.mkdir()
        with pytest.raises(CuMemIPCUnsupportedError, match="mount namespace"):
            receive_fd(
                lease.broker_path,
                lease.broker_token,
                lease.allocation_id,
                broker_root=other_root,
                retry_timeout=0,
            )
    finally:
        lease.close()
        broker.close()
        os.close(fd)


def test_stale_broker_socket_removed_when_filename_pid_is_live(
    tmp_path: Path,
) -> None:
    """An unreachable stale socket is removed despite a recycled live PID."""
    root = validate_broker_root(tmp_path)
    stale_path = root / f"lmcu-{os.getpid()}-deadbeef.sock"
    stale_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale_socket.bind(str(stale_path))
    stale_socket.listen()
    stale_socket.close()

    broker = CuMemFDBroker(root)
    try:
        assert not stale_path.exists()
    finally:
        broker.close()


def test_live_broker_socket_survives_peer_cleanup(tmp_path: Path) -> None:
    """A reachable same-UID broker socket is never removed as stale."""
    root = validate_broker_root(tmp_path)
    live_broker = CuMemFDBroker(root)
    try:
        peer_broker = CuMemFDBroker(root)
        try:
            assert Path(live_broker.path).exists()
        finally:
            peer_broker.close()
    finally:
        live_broker.close()


def test_broker_send_uses_fd_snapshot_across_final_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A final lease release cannot invalidate an authorized in-flight send."""
    root = validate_broker_root(tmp_path)
    broker = CuMemFDBroker(root)
    fd, path = tempfile.mkstemp(dir=root)
    os.unlink(path)
    os.write(fd, b"lease-snapshot")
    lease = broker.register(("snapshot", 1), fd)
    send_prepared = threading.Event()
    allow_send = threading.Event()
    real_array = cumem_ipc.array.array

    def pause_before_broker_send(
        typecode: str,
        initializer: list[int] | bytes | bytearray | None = None,
    ) -> array.array:
        result = (
            real_array(typecode)
            if initializer is None
            else real_array(typecode, initializer)
        )
        if threading.current_thread().name == "lmcache-cumem-fd-broker" and isinstance(
            initializer, list
        ):
            send_prepared.set()
            if not allow_send.wait(timeout=5.0):
                raise TimeoutError("broker send was not released")
        return result

    monkeypatch.setattr(cumem_ipc.array, "array", pause_before_broker_send)
    received: list[int] = []
    errors: list[BaseException] = []

    def receive() -> None:
        try:
            received.append(
                receive_fd(
                    lease.broker_path,
                    lease.broker_token,
                    lease.allocation_id,
                    broker_root=root,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    receiver = threading.Thread(target=receive)
    receiver.start()
    try:
        assert send_prepared.wait(timeout=5.0)
        lease.close()
        allow_send.set()
        receiver.join(timeout=5.0)

        assert not receiver.is_alive()
        assert errors == []
        assert len(received) == 1
        os.lseek(received[0], 0, os.SEEK_SET)
        assert os.read(received[0], 64) == b"lease-snapshot"
    finally:
        allow_send.set()
        receiver.join(timeout=5.0)
        for received_fd in received:
            os.close(received_fd)
        lease.close()
        broker.close()
        os.close(fd)


def test_broker_root_rejects_symlink(tmp_path: Path) -> None:
    """A symlink cannot stand in for the shared broker mount."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    with pytest.raises(CuMemIPCUnsupportedError, match="must not be a symlink"):
        validate_broker_root(link)
