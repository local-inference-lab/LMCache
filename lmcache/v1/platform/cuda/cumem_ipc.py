# SPDX-License-Identifier: Apache-2.0
"""Share vLLM CUDA VMM allocations through same-UID POSIX file descriptors."""

from __future__ import annotations

# Standard
from collections.abc import Hashable
from pathlib import Path
from typing import NamedTuple, Protocol, cast
import array
import ctypes
import json
import os
import secrets
import socket
import stat
import threading
import time

# Third Party
import torch


class CuMemIPCUnsupportedError(RuntimeError):
    """Indicate that a CUDA VMM allocation cannot be shared safely."""


BROKER_DIR_ENV = "LMCACHE_CUMEM_BROKER_DIR"
_SOCKET_PREFIX = "lmcu-"
_PROTOCOL_VERSION = 1


def validate_broker_root(
    directory: str | os.PathLike[str] | None = None,
    *,
    require_configured: bool = False,
) -> Path:
    """Validate the directory shared by the exporter and importer.

    Args:
        directory: Broker directory, or ``None`` to read
            :data:`BROKER_DIR_ENV`.
        require_configured: Require the environment variable instead of using
            ``/tmp`` when ``directory`` is absent.

    Returns:
        The validated absolute broker directory.

    Raises:
        CuMemIPCUnsupportedError: If the directory is absent, unsafe, not
            writable, or not owned by the current UID.
    """
    configured = os.environ.get(BROKER_DIR_ENV)
    if directory is None:
        if not configured:
            if require_configured:
                raise CuMemIPCUnsupportedError(
                    f"{BROKER_DIR_ENV} must name an existing same-UID directory "
                    "mounted at the same absolute path in the vLLM and LMCache "
                    "containers; private /tmp mount namespaces cannot carry "
                    "cuMem broker sockets"
                )
            directory = "/tmp"
        else:
            directory = configured

    root = Path(directory)
    if not root.is_absolute():
        raise CuMemIPCUnsupportedError(
            f"cuMem broker root must be absolute, got {str(root)!r}"
        )
    try:
        metadata = os.lstat(root)
    except FileNotFoundError as exc:
        raise CuMemIPCUnsupportedError(
            f"configured cuMem broker root {root} does not exist; create it and "
            f"mount it into both containers via {BROKER_DIR_ENV}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise CuMemIPCUnsupportedError(
            f"cuMem broker root {root} must not be a symlink"
        )
    if not stat.S_ISDIR(metadata.st_mode):
        raise CuMemIPCUnsupportedError(f"cuMem broker root {root} is not a directory")
    expected_uid = os.getuid()
    if metadata.st_uid != expected_uid:
        raise CuMemIPCUnsupportedError(
            f"cuMem broker root {root} must be owned by UID {expected_uid}, "
            f"not UID {metadata.st_uid}"
        )
    if not os.access(root, os.W_OK | os.X_OK):
        raise CuMemIPCUnsupportedError(
            f"cuMem broker root {root} is not writable/searchable by UID {expected_uid}"
        )
    return root


def _validate_descriptor_path(
    broker_path: str | os.PathLike[str], broker_root: Path
) -> Path:
    path = Path(broker_path)
    if not path.is_absolute() or ".." in path.parts or path.parent != broker_root:
        raise CuMemIPCUnsupportedError(
            f"cuMem broker descriptor path {path} is not directly under configured "
            f"root {broker_root}; this commonly indicates a container mount namespace "
            f"mismatch. Set {BROKER_DIR_ENV} to the same mounted path on both peers"
        )
    return path


def _validate_broker_socket(path: Path) -> None:
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode):
        raise CuMemIPCUnsupportedError(
            f"cuMem broker descriptor path {path} must not be a symlink"
        )
    if not stat.S_ISSOCK(metadata.st_mode):
        raise CuMemIPCUnsupportedError(f"cuMem broker path {path} is not a socket")
    expected_uid = os.getuid()
    if metadata.st_uid != expected_uid:
        raise CuMemIPCUnsupportedError(
            f"cuMem broker socket {path} must be owned by UID {expected_uid}, "
            f"not UID {metadata.st_uid}"
        )
    mode = stat.S_IMODE(metadata.st_mode)
    if mode != 0o600:
        raise CuMemIPCUnsupportedError(
            f"cuMem broker socket {path} must have mode 0600, not {mode:04o}"
        )


class CuMemAllocationDescriptor(NamedTuple):
    """Serializable identity and broker capability for one VMM allocation."""

    protocol_version: int
    allocation_id: str
    device_uuid: str
    allocation_size: int
    broker_path: str
    broker_token: str
    handle_type: str


def validate_tensor_view(
    *,
    allocation_size: int,
    storage_offset_bytes: int,
    storage_nbytes: int,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    itemsize: int,
) -> None:
    """Validate that a strided tensor view fits its physical VMM allocation.

    Args:
        allocation_size: Physical allocation extent in bytes.
        storage_offset_bytes: Allocation-relative start of the torch storage.
        storage_nbytes: Physical torch storage extent in bytes.
        shape: Tensor dimensions.
        stride: Tensor strides in elements.
        itemsize: Tensor element size in bytes.

    Raises:
        ValueError: If metadata is invalid or addresses bytes outside the
            storage or allocation.
    """
    if len(shape) != len(stride):
        raise ValueError("tensor shape/stride rank mismatch")
    if any(dim < 0 for dim in shape):
        raise ValueError("negative tensor dimension")
    if any(step < 0 for step in stride):
        raise ValueError("negative tensor stride is unsupported")
    if itemsize <= 0 or storage_offset_bytes < 0 or storage_nbytes < 0:
        raise ValueError("invalid tensor storage metadata")
    if storage_offset_bytes % itemsize:
        raise ValueError("storage offset is not dtype aligned")
    if storage_offset_bytes + storage_nbytes > allocation_size:
        raise ValueError("tensor storage exceeds cuMem allocation")
    required = 0
    if shape and not any(dim == 0 for dim in shape):
        required = (
            1 + sum((dim - 1) * step for dim, step in zip(shape, stride, strict=True))
        ) * itemsize
    if required > storage_nbytes:
        raise ValueError("tensor view exceeds physical storage extent")


class CuMemFDLease:
    """Keep one exporter-side broker reference alive until explicitly closed."""

    def __init__(
        self,
        broker: "CuMemFDBroker",
        allocation_id: str,
        broker_path: str,
        broker_token: str,
    ) -> None:
        self._broker = broker
        self.allocation_id = allocation_id
        self.broker_path = broker_path
        self.broker_token = broker_token
        self._closed = False

    def close(self) -> None:
        """Release this lease's reference to the brokered descriptor."""
        if not self._closed:
            self._closed = True
            self._broker.release(self.allocation_id)


class CuMemFDBroker:
    """Serve duplicate allocation FDs over a same-UID AF_UNIX socket."""

    def __init__(self, directory: str | os.PathLike[str] = "/tmp") -> None:
        self.root = validate_broker_root(directory)
        self._cleanup_stale_sockets()
        name = f"{_SOCKET_PREFIX}{os.getpid()}-{secrets.token_hex(4)}.sock"
        self.path = str(self.root / name)
        if len(self.path.encode()) >= 104:
            raise CuMemIPCUnsupportedError(
                f"cuMem broker socket path is too long: {self.path!r}; configure "
                f"a shorter shared path with {BROKER_DIR_ENV}"
            )
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        previous_umask = os.umask(0o177)
        try:
            self._socket.bind(self.path)
        finally:
            os.umask(previous_umask)
        os.chmod(self.path, 0o600, follow_symlinks=False)
        self._socket.listen()
        self._socket.settimeout(0.2)
        self._lock = threading.Lock()
        self._by_id: dict[str, tuple[str, int, Hashable, int]] = {}
        self._by_key: dict[Hashable, str] = {}
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._serve,
            name="lmcache-cumem-fd-broker",
            daemon=True,
        )
        self._thread.start()

    def _cleanup_stale_sockets(self) -> None:
        """Remove dead same-UID broker sockets left by crashed exporters."""
        for candidate in self.root.glob(f"{_SOCKET_PREFIX}*.sock"):
            try:
                metadata = os.lstat(candidate)
                if (
                    metadata.st_uid != os.getuid()
                    or not stat.S_ISSOCK(metadata.st_mode)
                    or stat.S_ISLNK(metadata.st_mode)
                ):
                    continue
                probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    probe.settimeout(0.05)
                    probe.connect(str(candidate))
                except (ConnectionRefusedError, FileNotFoundError):
                    try:
                        os.unlink(candidate)
                    except FileNotFoundError:
                        pass
                except OSError:
                    continue
                finally:
                    probe.close()
            except FileNotFoundError:
                continue

    @property
    def registration_count(self) -> int:
        """Return the number of distinct exported allocations."""
        with self._lock:
            return len(self._by_id)

    def register(self, key: Hashable, fd: int) -> CuMemFDLease:
        """Register or alias an allocation FD.

        Args:
            key: Stable exporter-local allocation identity.
            fd: Shareable allocation file descriptor; ownership remains with
                the caller.

        Returns:
            A reference-counted lease containing serializable broker metadata.
        """
        with self._lock:
            existing_id = self._by_key.get(key)
            if existing_id is not None:
                token, owned_fd, saved_key, refs = self._by_id[existing_id]
                self._by_id[existing_id] = (token, owned_fd, saved_key, refs + 1)
                return CuMemFDLease(self, existing_id, self.path, token)
            allocation_id = secrets.token_hex(16)
            token = secrets.token_hex(32)
            self._by_key[key] = allocation_id
            self._by_id[allocation_id] = (token, os.dup(fd), key, 1)
            return CuMemFDLease(self, allocation_id, self.path, token)

    def release(self, allocation_id: str) -> None:
        """Drop one alias reference and close the FD after the final release."""
        with self._lock:
            entry = self._by_id.get(allocation_id)
            if entry is None:
                return
            token, fd, key, refs = entry
            if refs > 1:
                self._by_id[allocation_id] = (token, fd, key, refs - 1)
                return
            self._by_id.pop(allocation_id)
            self._by_key.pop(key, None)
        os.close(fd)

    def _serve(self) -> None:
        while not self._closed.is_set():
            try:
                conn, _ = self._socket.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with conn:
                try:
                    peer = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
                    peer_uid = int.from_bytes(peer[4:8], byteorder="little")
                    if peer_uid != os.getuid():
                        continue
                    request = json.loads(conn.recv(4096).decode())
                    allocation_id = str(request["allocation_id"])
                    token = str(request["token"])
                    with self._lock:
                        saved = self._by_id.get(allocation_id)
                        if saved is None or not secrets.compare_digest(saved[0], token):
                            continue
                        fd = os.dup(saved[1])
                    try:
                        rights = array.array("i", [fd])
                        conn.sendmsg(
                            [b"F"],
                            [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights)],
                        )
                    finally:
                        os.close(fd)
                except (KeyError, ValueError, OSError, json.JSONDecodeError):
                    continue

    def close(self) -> None:
        """Stop the broker and release every owned descriptor."""
        if self._closed.is_set():
            return
        self._closed.set()
        self._socket.close()
        self._thread.join(timeout=1)
        with self._lock:
            entries = list(self._by_id.values())
            self._by_id.clear()
            self._by_key.clear()
        for _, fd, _, _ in entries:
            os.close(fd)
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass


def receive_fd(
    broker_path: str,
    broker_token: str,
    allocation_id: str,
    *,
    broker_root: str | os.PathLike[str] | None = None,
    retry_timeout: float = 10.0,
) -> int:
    """Receive one duplicate allocation FD from its exporting worker.

    Args:
        broker_path: Absolute AF_UNIX socket path from the descriptor.
        broker_token: Unpredictable descriptor capability token.
        allocation_id: Descriptor allocation identity.
        broker_root: Expected shared root, or ``None`` to require the
            configured root.
        retry_timeout: Seconds to retry while an exporter socket starts.

    Returns:
        A received FD owned by the caller.

    Raises:
        CuMemIPCUnsupportedError: If validation, connection, authorization, or
            descriptor transfer fails.
    """
    root = validate_broker_root(
        broker_root,
        require_configured=broker_root is None,
    )
    path = _validate_descriptor_path(broker_path, root)
    deadline = time.monotonic() + max(0.0, retry_timeout)
    while True:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            _validate_broker_socket(path)
            sock.settimeout(max(0.2, min(10.0, retry_timeout or 0.2)))
            sock.connect(str(path))
            sock.sendall(
                json.dumps(
                    {"token": broker_token, "allocation_id": allocation_id}
                ).encode()
            )
            payload, ancillary, _, _ = sock.recvmsg(
                1,
                socket.CMSG_SPACE(array.array("i").itemsize),
            )
            if payload != b"F":
                raise CuMemIPCUnsupportedError(
                    "cuMem FD broker rejected the descriptor token or allocation"
                )
            for level, kind, data in ancillary:
                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                    fds = array.array("i")
                    fds.frombytes(data[: fds.itemsize])
                    return int(fds[0])
            raise CuMemIPCUnsupportedError("cuMem FD broker returned no handle")
        except (FileNotFoundError, ConnectionRefusedError) as exc:
            if time.monotonic() >= deadline:
                raise CuMemIPCUnsupportedError(
                    f"cannot reach cuMem FD broker {path} under configured root "
                    f"{root}: {exc}. The exporter and LMCache server likely have "
                    "different mount namespaces; mount one shared directory at "
                    f"the same path and set {BROKER_DIR_ENV} on both containers"
                ) from exc
            time.sleep(0.05)
        except OSError as exc:
            raise CuMemIPCUnsupportedError(
                f"cannot receive cuMem allocation handle from {path} under "
                f"configured root {root}: {exc}"
            ) from exc
        finally:
            sock.close()


class CuMemAllocationImporter(Protocol):
    """Import and close one physical CUDA VMM allocation."""

    def import_allocation(
        self, descriptor: CuMemAllocationDescriptor
    ) -> "ImportedCuMemMapping":
        """Import ``descriptor`` into the current process."""

    def close_allocation(self, mapping: "ImportedCuMemMapping") -> None:
        """Unmap and release an imported allocation."""


class ImportedCuMemRegistry:
    """Deduplicate mappings by allocation identity and device with alias refs."""

    def __init__(self, importer: CuMemAllocationImporter) -> None:
        self._importer = importer
        self._lock = threading.Lock()
        self._entries: dict[
            tuple[str, str],
            tuple[CuMemAllocationDescriptor, ImportedCuMemMapping, int],
        ] = {}

    def acquire(self, descriptor: CuMemAllocationDescriptor) -> "ImportedCuMemMapping":
        """Acquire one alias reference to an imported allocation."""
        if (
            descriptor.protocol_version != _PROTOCOL_VERSION
            or descriptor.handle_type != "posix_fd"
        ):
            raise ValueError("unsupported cuMem IPC descriptor")
        key = (descriptor.device_uuid, descriptor.allocation_id)
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                saved, mapping, refs = existing
                if saved != descriptor:
                    raise ValueError("incompatible duplicate cuMem descriptor")
                self._entries[key] = (saved, mapping, refs + 1)
                return mapping
            mapping = self._importer.import_allocation(descriptor)
            self._entries[key] = (descriptor, mapping, 1)
            return mapping

    def release(self, descriptor: CuMemAllocationDescriptor) -> None:
        """Release one alias and close the final physical mapping."""
        key = (descriptor.device_uuid, descriptor.allocation_id)
        with self._lock:
            existing = self._entries.get(key)
            if existing is None:
                return
            saved, mapping, refs = existing
            if saved != descriptor:
                raise ValueError("incompatible cuMem descriptor release")
            if refs > 1:
                self._entries[key] = (saved, mapping, refs - 1)
                return
            # Keep the entry visible until close succeeds, so status cannot
            # claim cleanup while imported memory remains mapped.
            self._importer.close_allocation(mapping)
            self._entries.pop(key)

    def refcount(self, descriptor: CuMemAllocationDescriptor) -> int:
        """Return the alias count for one descriptor."""
        key = (descriptor.device_uuid, descriptor.allocation_id)
        with self._lock:
            entry = self._entries.get(key)
            return 0 if entry is None else entry[2]

    @property
    def registration_count(self) -> int:
        """Return the number of physical imported mappings."""
        with self._lock:
            return len(self._entries)

    @property
    def alias_refcount_total(self) -> int:
        """Return alias references summed across imported mappings."""
        with self._lock:
            return sum(refs for _, _, refs in self._entries.values())


def _check_cuda(result: object, operation: str) -> tuple[object, ...]:
    values = result if isinstance(result, tuple) else (result,)
    if int(cast(int, values[0])) != 0:
        raise CuMemIPCUnsupportedError(f"{operation} failed: {values[0]}")
    return tuple(values[1:])


class ImportedCuMemMapping:
    """Own one imported generic handle, VA reservation, and mapped address."""

    def __init__(
        self,
        ptr: int,
        size: int,
        handle: object,
        device_index: int,
    ) -> None:
        self.ptr = ptr
        self.size = size
        self.handle = handle
        self.device_index = device_index
        self._mapped = True
        self._address_reserved = True
        self._handle_open = True
        self._closed = False

    def as_torch_bytes(self) -> torch.Tensor:
        """Return a uint8 tensor spanning the complete imported allocation."""
        # Third Party
        import cupy

        with cupy.cuda.Device(self.device_index):
            memory = cupy.cuda.UnownedMemory(self.ptr, self.size, owner=self)
            pointer = cupy.cuda.MemoryPointer(memory, 0)
            raw = cupy.ndarray(self.size, dtype=cupy.uint8, memptr=pointer)
        return torch.from_dlpack(raw)


class CudaDriverAllocationImporter:
    """Import POSIX-FD CUDA VMM handles through the CUDA driver API."""

    def _device_index(self, uuid: str) -> int:
        matches = [
            index
            for index in range(torch.cuda.device_count())
            if str(torch.cuda.get_device_properties(index).uuid) == uuid
        ]
        if len(matches) != 1:
            raise CuMemIPCUnsupportedError(
                f"cuMem descriptor device UUID {uuid!r} is unavailable"
            )
        return matches[0]

    def import_allocation(
        self, descriptor: CuMemAllocationDescriptor
    ) -> ImportedCuMemMapping:
        """Import, reserve, map, and grant access to one allocation."""
        # Third Party
        from cuda.bindings import driver

        device_index = self._device_index(descriptor.device_uuid)
        fd = receive_fd(
            descriptor.broker_path,
            descriptor.broker_token,
            descriptor.allocation_id,
        )
        ptr: object | None = None
        handle: object | None = None
        mapped = False
        try:
            with torch.cuda.device(device_index):
                torch.cuda.init()
                (handle,) = _check_cuda(
                    driver.cuMemImportFromShareableHandle(
                        fd,
                        driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                    ),
                    "cuMemImportFromShareableHandle",
                )
                (ptr,) = _check_cuda(
                    driver.cuMemAddressReserve(
                        descriptor.allocation_size,
                        0,
                        0,
                        0,
                    ),
                    "cuMemAddressReserve",
                )
                _check_cuda(
                    driver.cuMemMap(
                        ptr,
                        descriptor.allocation_size,
                        0,
                        handle,
                        0,
                    ),
                    "cuMemMap",
                )
                mapped = True
                access = driver.CUmemAccessDesc()
                access.location.type = (
                    driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
                )
                access.location.id = device_index
                access.flags = (
                    driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                )
                _check_cuda(
                    driver.cuMemSetAccess(
                        ptr,
                        descriptor.allocation_size,
                        [access],
                        1,
                    ),
                    "cuMemSetAccess",
                )
            return ImportedCuMemMapping(
                int(cast(int, ptr)),
                descriptor.allocation_size,
                handle,
                device_index,
            )
        except BaseException:
            if ptr is not None:
                if mapped:
                    driver.cuMemUnmap(ptr, descriptor.allocation_size)
                driver.cuMemAddressFree(ptr, descriptor.allocation_size)
            if handle is not None:
                driver.cuMemRelease(handle)
            raise
        finally:
            os.close(fd)

    def close_allocation(self, mapping: ImportedCuMemMapping) -> None:
        """Synchronize, unmap, free VA, and release the imported handle."""
        if mapping._closed:
            return
        # Third Party
        from cuda.bindings import driver

        with torch.cuda.device(mapping.device_index):
            torch.cuda.synchronize(mapping.device_index)
            if mapping._mapped:
                _check_cuda(
                    driver.cuMemUnmap(mapping.ptr, mapping.size),
                    "cuMemUnmap",
                )
                mapping._mapped = False
            if mapping._address_reserved:
                _check_cuda(
                    driver.cuMemAddressFree(mapping.ptr, mapping.size),
                    "cuMemAddressFree",
                )
                mapping._address_reserved = False
            if mapping._handle_open:
                _check_cuda(driver.cuMemRelease(mapping.handle), "cuMemRelease")
                mapping._handle_open = False
        mapping._closed = True


_broker_lock = threading.Lock()
_broker: CuMemFDBroker | None = None
_import_registry = ImportedCuMemRegistry(CudaDriverAllocationImporter())


def get_fd_broker() -> CuMemFDBroker:
    """Return the process-local broker rooted in the configured shared mount."""
    global _broker
    with _broker_lock:
        if _broker is None:
            root = validate_broker_root(require_configured=True)
            _broker = CuMemFDBroker(root)
        return _broker


def find_cumem_allocation(
    tensor: torch.Tensor,
) -> tuple[int, int, object] | None:
    """Find the live vLLM VMM allocation containing ``tensor``.

    Args:
        tensor: CUDA tensor to locate in vLLM's cuMem allocator.

    Returns:
        ``(base_address, allocation_size, handle_holder)`` when found, or
        ``None`` for an ordinary CUDA allocation.

    Raises:
        CuMemIPCUnsupportedError: If allocator metadata ambiguously matches.
    """
    try:
        # Third Party
        from vllm.device_allocator.cumem import CuMemAllocator
    except (ImportError, AssertionError):
        return None
    allocator = CuMemAllocator.instance
    if allocator is None:
        return None
    storage = tensor.untyped_storage()
    storage_ptr = int(storage.data_ptr())
    storage_end = storage_ptr + int(storage.nbytes())
    matches: list[tuple[int, int, object]] = []
    for base, data in allocator.pointer_to_data.items():
        _device, size, pointer, handle = data.handle
        if (
            not data.is_asleep
            and int(base) == int(pointer)
            and int(base) <= storage_ptr
            and storage_end <= int(base) + int(size)
        ):
            matches.append((int(base), int(size), handle))
    if len(matches) > 1:
        raise CuMemIPCUnsupportedError("tensor matches multiple cuMem allocations")
    return matches[0] if matches else None


def generic_handle_from_holder(handle_holder: object) -> int:
    """Dereference vLLM's heap-held ``CUmemGenericAllocationHandle``.

    Args:
        handle_holder: Address of vLLM's scalar generic-handle storage.

    Returns:
        The CUDA generic allocation handle value.

    Raises:
        CuMemIPCUnsupportedError: If the allocation uses a chunked holder not
            supported by protocol version 1.
    """
    if not isinstance(handle_holder, int):
        raise CuMemIPCUnsupportedError(
            "chunked vLLM cuMem allocations are unsupported by protocol version 1"
        )
    return int(ctypes.c_uint64.from_address(handle_holder).value)


def export_cumem_allocation(
    tensor: torch.Tensor,
    allocation: tuple[int, int, object],
) -> tuple[CuMemAllocationDescriptor, CuMemFDLease, int]:
    """Export one allocation and return descriptor, lease, and storage offset.

    Args:
        tensor: Tensor whose storage lies inside ``allocation``.
        allocation: Result from :func:`find_cumem_allocation`.

    Returns:
        Descriptor, exporter lease, and allocation-relative storage offset.

    Raises:
        CuMemIPCUnsupportedError: If the generic handle cannot be exported.
    """
    # Third Party
    from cuda.bindings import driver

    base, size, handle_holder = allocation
    handle = generic_handle_from_holder(handle_holder)
    device_index = tensor.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    with torch.cuda.device(device_index):
        (fd,) = _check_cuda(
            driver.cuMemExportToShareableHandle(
                handle,
                driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                0,
            ),
            "cuMemExportToShareableHandle",
        )
    try:
        device_uuid = str(torch.cuda.get_device_properties(device_index).uuid)
        lease = get_fd_broker().register(
            (device_uuid, base, size, handle),
            int(cast(int, fd)),
        )
    finally:
        os.close(int(cast(int, fd)))
    descriptor = CuMemAllocationDescriptor(
        protocol_version=_PROTOCOL_VERSION,
        allocation_id=lease.allocation_id,
        device_uuid=device_uuid,
        allocation_size=size,
        broker_path=lease.broker_path,
        broker_token=lease.broker_token,
        handle_type="posix_fd",
    )
    storage_offset = int(tensor.untyped_storage().data_ptr()) - base
    return descriptor, lease, storage_offset


def acquire_imported_mapping(
    descriptor: CuMemAllocationDescriptor,
) -> ImportedCuMemMapping:
    """Acquire one alias reference to an imported mapping."""
    return _import_registry.acquire(descriptor)


def release_imported_mapping(descriptor: CuMemAllocationDescriptor) -> None:
    """Release one alias reference to an imported mapping."""
    _import_registry.release(descriptor)


def imported_registration_count() -> int:
    """Return the number of physical imported cuMem allocations."""
    return _import_registry.registration_count


def imported_alias_refcount_total() -> int:
    """Return the total number of imported tensor aliases."""
    return _import_registry.alias_refcount_total
