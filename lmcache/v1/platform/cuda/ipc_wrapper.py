# SPDX-License-Identifier: Apache-2.0
"""CUDA IPC wrapper implementations.

:class:`CudaIPCWrapper` shares ordinary tensors through PyTorch's storage IPC
(needs a shared ``/dev/shm``); :class:`RawCudaIPCWrapper` shares them through
driver-level CUDA IPC memory handles across isolated containers.
:class:`CuMemCudaIPCWrapper` handles vLLM VMM allocations for either factory
selection through POSIX-FD cuMem handles and a same-UID ``SCM_RIGHTS`` broker.

``device_type="cuda"`` binds to one of the two via
:attr:`~lmcache.v1.platform.cuda.CudaDeviceSpec.ipc_wrapper_cls`:
:class:`CudaIPCWrapper` by default, :class:`RawCudaIPCWrapper` when the
process-global isolated-IPC switch is on (see
``lmcache/v1/platform/isolated_ipc.py``). The multiprocess adapter
dispatches through
:func:`~lmcache.v1.platform.resolve_kv_wrapper_factory`; the TRT-LLM
adapter instantiates :class:`RawCudaIPCWrapper` directly.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, ClassVar
import threading

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.logging import init_logger
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper
from lmcache.v1.platform.cuda.utils import _cuda

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.cuda.cumem_ipc import (
        CuMemFDLease,
        ImportedCuMemMapping,
    )

logger = init_logger(__name__)

_NON_IPC_MEMORY_HINT = (
    "CUDA IPC memory handles only support cudaMalloc-style allocations. "
    "Memory created through the CUDA VMM API cannot be shared this way -- "
    "common sources are PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    "and vLLM's sleep mode (CuMemAllocator). Disable those or turn off "
    "isolated IPC for this deployment."
)

# Refcounted registry of allocations this process has mapped via
# cudaIpcOpenMemHandle, keyed by handle bytes: handle -> [mapped_ptr, opens].
# The driver returns ONE mapping per (process, allocation) no matter how
# often it is opened, and a single cudaIpcCloseMemHandle unmaps it for
# every user -- so per-layer tensors sharing one allocation must be
# counted, and the mapping unmapped only when the last wrapper closes.
# An unclosed mapping pins the exporting process's device memory even
# after the exporter dies (a dead vLLM worker's KV pool stays resident
# until the server closes or exits). Keyed by handle bytes alone: a KV
# allocation is only ever imported on the wrapper's own device.
_MAPPED_ALLOCATIONS: dict[bytes, list[int]] = {}
_MAPPINGS_LOCK = threading.Lock()


class CudaIPCWrapper(DeviceIPCWrapper):
    #: ``torch.device.type`` this wrapper handles. Kept as a class-level
    #: constant so external tooling / tests can introspect the binding.
    device_type: ClassVar[str] = "cuda"

    @classmethod
    def wrap(cls, tensor: torch.Tensor) -> "DeviceIPCWrapper":
        """Factory used by
        :func:`~lmcache.v1.platform.resolve_kv_wrapper_factory`.

        Args:
            tensor: A CUDA tensor backed by PyTorch's caching allocator or
                vLLM's cuMem allocator.

        Returns:
            A cuMem wrapper for a live vLLM VMM allocation, otherwise a
            standard PyTorch CUDA IPC wrapper.
        """
        # First Party
        from lmcache.v1.platform.cuda.cumem_ipc import find_cumem_allocation

        allocation = find_cumem_allocation(tensor)
        if allocation is not None:
            return CuMemCudaIPCWrapper(tensor, allocation)
        return cls(tensor)

    def __init__(self, tensor: torch.Tensor) -> None:
        # First Party
        from lmcache.v1.gpu_connector.kv_format.contiguity import (
            attempt_permute_to_contiguous_view,
        )

        # Permute any non-contiguous view (e.g. vLLM's NHD-over-HND) so the
        # shape/stride we encode across IPC reflects the physical layout.
        # Offset is preserved by the wrapper's storage_offset field.
        tensor = attempt_permute_to_contiguous_view(tensor)

        storage = tensor.untyped_storage()
        handle = storage._share_cuda_()

        self.handle = handle
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.storage_offset = int(tensor.storage_offset())

        device_index = tensor.device.index
        self.device_uuid = self._get_device_uuid(device_index)

    def to_tensor(self) -> torch.Tensor:
        """
        Note:
            This function may break if the accelerator is not initialized.
            We should call ``torch_dev.init()`` before using this function
            (guarded by hasattr since not all backends expose init()).
        """
        device_index = self._get_device_index_from_uuid(self.device_uuid)

        storage = torch.UntypedStorage._new_shared_cuda(  # noqa: SLF001
            device_index, *self.handle[1:]
        )

        t = torch.empty(
            (), device=f"{torch_device_type}:{device_index}", dtype=self.dtype
        )
        t.set_(storage, self.storage_offset, self.shape, self.stride)
        return t


class CuMemCudaIPCWrapper(DeviceIPCWrapper):
    """Serializable tensor view over a vLLM CUDA VMM allocation.

    The POSIX allocation FD moves out of band through a same-UID
    ``SCM_RIGHTS`` broker. Pickled metadata preserves the complete physical
    storage extent, allocation-relative offset, shape, stride, dtype, and
    device identity.
    """

    device_type: ClassVar[str] = "cuda"

    def __init__(
        self,
        tensor: torch.Tensor,
        allocation: tuple[int, int, object],
    ) -> None:
        # First Party
        from lmcache.v1.platform.cuda.cumem_ipc import (
            export_cumem_allocation,
            validate_tensor_view,
        )

        descriptor, lease, allocation_storage_offset = export_cumem_allocation(
            tensor,
            allocation,
        )
        try:
            self.handle = descriptor
            self.dtype = tensor.dtype
            self.shape = tuple(tensor.shape)
            self.stride = tuple(tensor.stride())
            self.storage_offset = int(tensor.storage_offset())
            self.device_uuid = descriptor.device_uuid
            self.physical_storage_nbytes = int(tensor.untyped_storage().nbytes())
            self.allocation_storage_offset_bytes = allocation_storage_offset
            self._lease: CuMemFDLease | None = lease
            self._mapping: ImportedCuMemMapping | None = None
            self._tensor: torch.Tensor | None = None
            self._closed = False
            validate_tensor_view(
                allocation_size=descriptor.allocation_size,
                storage_offset_bytes=allocation_storage_offset,
                storage_nbytes=self.physical_storage_nbytes,
                shape=self.shape,
                stride=self.stride,
                itemsize=tensor.element_size(),
            )
            self._validate_view_inside_storage(tensor.element_size())
        except BaseException:
            lease.close()
            raise

    def _validate_view_inside_storage(self, itemsize: int) -> None:
        if self.storage_offset < 0:
            raise ValueError("negative tensor storage offset")
        last = self.storage_offset
        if self.shape and not any(dim == 0 for dim in self.shape):
            last += sum(
                (dim - 1) * step
                for dim, step in zip(self.shape, self.stride, strict=True)
            )
        if (last + 1) * itemsize > self.physical_storage_nbytes:
            raise ValueError("tensor view exceeds physical storage extent")

    def __getstate__(self) -> dict[str, object]:
        """Serialize metadata without exporter- or importer-local resources."""
        state: dict[str, object] = self.__dict__.copy()
        state["_lease"] = None
        state["_mapping"] = None
        state["_tensor"] = None
        state["_closed"] = False
        return state

    def to_tensor(self) -> torch.Tensor:
        """Acquire the allocation mapping and reconstruct the exact view."""
        if self._closed:
            raise RuntimeError("cuMem IPC wrapper is closed")
        if self._tensor is not None:
            return self._tensor

        # First Party
        from lmcache.v1.platform.cuda.cumem_ipc import (
            acquire_imported_mapping,
            release_imported_mapping,
            validate_tensor_view,
        )

        itemsize = torch.empty((), dtype=self.dtype).element_size()
        validate_tensor_view(
            allocation_size=self.handle.allocation_size,
            storage_offset_bytes=self.allocation_storage_offset_bytes,
            storage_nbytes=self.physical_storage_nbytes,
            shape=self.shape,
            stride=self.stride,
            itemsize=itemsize,
        )
        self._validate_view_inside_storage(itemsize)
        mapping = acquire_imported_mapping(self.handle)
        try:
            raw = mapping.as_torch_bytes()
            typed = raw.view(self.dtype)
            absolute_offset = (
                self.allocation_storage_offset_bytes + self.storage_offset * itemsize
            )
            tensor = torch.as_strided(
                typed,
                self.shape,
                self.stride,
                storage_offset=absolute_offset // itemsize,
            )
        except BaseException:
            release_imported_mapping(self.handle)
            raise
        self._mapping = mapping
        self._tensor = tensor
        return tensor

    def close(self) -> None:
        """Release imported mapping and exporter lease references.

        The operation is idempotent. A mapping close failure remains retryable
        and is reported after exporter cleanup is attempted.

        Raises:
            RuntimeError: If one or more owned resources could not be released.
        """
        if self._closed:
            return
        self._tensor = None
        close_errors: list[BaseException] = []
        if self._mapping is not None:
            # First Party
            from lmcache.v1.platform.cuda.cumem_ipc import release_imported_mapping

            try:
                release_imported_mapping(self.handle)
            except BaseException as exc:
                close_errors.append(exc)
            else:
                self._mapping = None
        if self._lease is not None:
            try:
                self._lease.close()
            except BaseException as exc:
                close_errors.append(exc)
            else:
                self._lease = None
        self._closed = self._mapping is None and self._lease is None
        if close_errors:
            raise RuntimeError(
                f"CuMemCudaIPCWrapper.close failed with "
                f"{len(close_errors)} cleanup error(s)"
            ) from close_errors[0]


class RawCudaIPCWrapper(DeviceIPCWrapper):
    """IPC wrapper that shares CUDA tensors through driver-level IPC only.

    ``CudaIPCWrapper`` rides PyTorch's ``UntypedStorage._share_cuda_()``,
    which requires a shared ``/dev/shm`` between the processes (torch
    keeps its IPC reference counter there). This wrapper instead calls
    ``cudaIpcGetMemHandle`` on the tensor's allocation and reconstructs
    on the receiving side via ``cudaIpcOpenMemHandle`` plus a CuPy
    ``UnownedMemory`` → DLPack → ``torch`` round-trip. CUDA IPC *memory*
    handles rendezvous in the kernel driver, so this works across fully
    isolated containers -- no shared IPC namespace, no common /dev/shm.

    Two caller groups use it:

    - the MP registration path selects it via
      :attr:`~lmcache.v1.platform.cuda.CudaDeviceSpec.ipc_wrapper_cls`
      when the isolated-IPC switch is on
      (``lmcache/v1/platform/isolated_ipc.py``);
    - the TRT-LLM adapter instantiates it directly for its
      ``cudaMalloc``'d KV pool, which ``_share_cuda_()`` cannot wrap at
      all.

    An IPC mem handle always maps the *whole allocation* it was taken
    from, and opening it returns the allocation's *base* pointer. Torch
    caching-allocator tensors usually sit at an interior pointer, so the
    wrapper ships ``data_ptr - cuMemGetAddressRange(data_ptr).base`` and
    the consumer reads at that offset from the opened base.

    Sharing the ``DeviceIPCWrapper`` base (rather than introducing a
    parallel class with its own msgspec ext code) is load-bearing —
    msgspec does not support unions of custom ext-encoded types. With a
    common base, ``KVCache = list[DeviceIPCWrapper]`` type-checks, the
    single ext code 1 round-trips every wrapper, and pickle preserves
    the concrete subclass identity through the wire so ``to_tensor``
    dispatches correctly.
    """

    #: Same ``torch.device.type`` as ``CudaIPCWrapper``; exposed on
    #: :attr:`~lmcache.v1.platform.cuda.CudaDeviceSpec.ipc_wrapper_cls`
    #: under isolated IPC, instantiated directly by the TRT-LLM adapter.
    device_type: ClassVar[str] = "cuda"

    @classmethod
    def wrap(cls, tensor: torch.Tensor) -> "DeviceIPCWrapper":
        """Factory used by
        :func:`~lmcache.v1.platform.resolve_kv_wrapper_factory`.

        Args:
            tensor: A CUDA tensor backed by either vLLM cuMem or
                ``cudaMalloc``-style memory.

        Returns:
            A cuMem wrapper for a live vLLM VMM allocation, otherwise a raw
            CUDA IPC wrapper.
        """
        # First Party
        from lmcache.v1.platform.cuda.cumem_ipc import find_cumem_allocation

        allocation = find_cumem_allocation(tensor)
        if allocation is not None:
            return CuMemCudaIPCWrapper(tensor, allocation)
        return cls(tensor)

    def __init__(self, tensor: torch.Tensor) -> None:
        # First Party
        from lmcache.v1.gpu_connector.kv_format.contiguity import (
            attempt_permute_to_contiguous_view,
        )

        # Same layout normalization as CudaIPCWrapper: permute
        # non-contiguous views (e.g. vLLM's NHD-over-HND) into contiguous
        # ones, metadata-only. The flat-bytes reconstruction below only
        # supports contiguous tensors, so anything still non-contiguous
        # is rejected rather than silently reordered.
        tensor = attempt_permute_to_contiguous_view(tensor)
        if not tensor.is_contiguous():
            raise ValueError(
                "RawCudaIPCWrapper requires a tensor that is contiguous "
                f"(possibly after permutation); got shape={tuple(tensor.shape)} "
                f"stride={tuple(tensor.stride())}"
            )

        data_ptr = tensor.data_ptr()
        range_result = _cuda.driver.cuMemGetAddressRange(
            _cuda.driver.CUdeviceptr(data_ptr)
        )
        if range_result[0] != 0:
            raise RuntimeError(
                f"cuMemGetAddressRange failed: {range_result[0]} "
                f"(ptr=0x{data_ptr:x}). " + _NON_IPC_MEMORY_HINT
            )
        _err, alloc_base, _alloc_size = range_result

        err, ipc_handle = _cuda.runtime.cudaIpcGetMemHandle(int(alloc_base))
        if err != _cuda.runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(
                f"cudaIpcGetMemHandle failed: {err} (ptr=0x{data_ptr:x}). "
                + _NON_IPC_MEMORY_HINT
            )

        # Store only what's needed for reconstruction. The handle maps
        # the whole allocation; the offset locates the tensor within it.
        self._ipc_handle_reserved = bytes(ipc_handle.reserved)
        self._alloc_offset = data_ptr - int(
            alloc_base
        )  # offset in bytes not the same as storage offset
        self._nbytes = tensor.numel() * tensor.element_size()

        # DeviceIPCWrapper interface fields. ``handle`` is unused —
        # ``to_tensor`` is overridden to bypass it — but kept (None) so
        # the base-class equality check has a value to compare.
        # ``storage_offset`` is 0 because ``data_ptr`` (folded into
        # ``_alloc_offset``) already points at the tensor's first element.
        self.handle = None
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.storage_offset = 0

        # Opens this wrapper holds on the shared mapping registry;
        # released by close(). Travels through pickle as 0 (the producer
        # never opens), so a receiver always starts at 0.
        self._opens = 0

        device_index = tensor.device.index
        self.device_uuid = self._get_device_uuid(device_index)

    def to_tensor(self) -> torch.Tensor:
        """Reconstruct the tensor in this process via raw CUDA IPC.

        Opens the allocation's mem handle through the process-wide
        refcounted registry (one physical mapping per allocation).
        Every call takes one reference; :meth:`close` releases all
        references this wrapper holds.
        """
        # Third Party
        import cupy

        device_index = self._get_device_index_from_uuid(self.device_uuid)

        with _MAPPINGS_LOCK:
            entry = _MAPPED_ALLOCATIONS.get(self._ipc_handle_reserved)
            if entry is None:
                handle = _cuda.runtime.cudaIpcMemHandle_t()
                handle.reserved = self._ipc_handle_reserved
                with torch.cuda.device(device_index):
                    err, ptr = _cuda.runtime.cudaIpcOpenMemHandle(
                        handle, _cuda.runtime.cudaIpcMemLazyEnablePeerAccess
                    )
                if err != _cuda.runtime.cudaError_t.cudaSuccess:
                    raise RuntimeError(f"cudaIpcOpenMemHandle failed: {err}")
                entry = [int(ptr), 0]
                _MAPPED_ALLOCATIONS[self._ipc_handle_reserved] = entry
            entry[1] += 1
            self._opens += 1
            base_ptr = entry[0]

        # Wrap as a flat ``uint8`` CuPy array at the allocation offset,
        # DLPack to torch, then view as the original dtype/shape.
        # ``uint8`` avoids dtype-conversion gaps (bfloat16, fp8 have no
        # direct CuPy/NumPy equivalent without ml_dtypes).
        with cupy.cuda.Device(device_index):
            mem = cupy.cuda.UnownedMemory(
                base_ptr, self._alloc_offset + self._nbytes, owner=self
            )
            memptr = cupy.cuda.MemoryPointer(mem, self._alloc_offset)
            cp_flat = cupy.ndarray(self._nbytes, dtype=cupy.uint8, memptr=memptr)

        raw = torch.from_dlpack(cp_flat)
        return raw.view(self.dtype).reshape(self.shape)

    def close(self) -> None:
        """Release this wrapper's references on the imported mapping.

        Unmaps the allocation (``cudaIpcCloseMemHandle``) when the last
        reference across all wrappers is released, returning the
        exporter's device memory once the exporter itself has freed it.
        Idempotent; safe on wrappers that never imported. Tensors from
        :meth:`to_tensor` must no longer be dereferenced after the last
        close -- their later garbage collection is harmless (the CuPy
        memory is unowned; no device call is issued on collection).

        Unmap failures are logged, not raised: close runs on teardown
        paths (the worker reaper) where raising would abort cleanup of
        the remaining entries.
        """
        opens = getattr(self, "_opens", 0)
        if opens <= 0:
            return
        with _MAPPINGS_LOCK:
            self._opens = 0
            entry = _MAPPED_ALLOCATIONS.get(self._ipc_handle_reserved)
            if entry is None:
                return
            entry[1] -= opens
            if entry[1] > 0:
                return
            del _MAPPED_ALLOCATIONS[self._ipc_handle_reserved]
            device_index = self._get_device_index_from_uuid(self.device_uuid)
            with torch.cuda.device(device_index):
                (err,) = _cuda.runtime.cudaIpcCloseMemHandle(entry[0])
            if err != _cuda.runtime.cudaError_t.cudaSuccess:
                logger.warning(
                    "cudaIpcCloseMemHandle failed: %s (ptr=0x%x)", err, entry[0]
                )
