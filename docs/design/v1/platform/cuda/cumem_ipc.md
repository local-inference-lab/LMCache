# cuMem CUDA IPC

## Purpose

vLLM sleep-mode KV caches use CUDA virtual-memory-management allocations.
Ordinary `cudaIpcGetMemHandle` and PyTorch storage sharing do not safely export
those allocations. LMCache therefore uses a POSIX-FD-shareable cuMem path for
LMCache-driven transfers while preserving vLLM's ownership of the allocation.

## Allocation contract

The serving worker must preload
`csrc/cumem_ipc_interposer/liblmcache_cumem_shareable.so`. The interposer changes
only `CUmemAllocationProp.requestedHandleTypes` passed to `cuMemCreate`, forcing
`CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR` even when the driver selected a
FABRIC handle. Allocation, mapping, sleep/wake, and release remain owned by
vLLM and CUDA.

Build against the CUDA 13.3 toolkit:

```bash
make -C csrc/cumem_ipc_interposer CUDA_HOME=/usr/local/cuda-13.3
```

Launch each vLLM worker with the resulting library in `LD_PRELOAD`.

## Descriptor transport

`LMCACHE_CUMEM_BROKER_DIR` must point to a pre-existing directory that:

- is mounted at the same absolute path in the vLLM and LMCache containers;
- is owned by the UID shared by both processes;
- is writable/searchable by that UID; and
- is a real directory, not a symlink.

The exporter creates a mode-`0600` AF_UNIX socket directly below that root.
Pickled registration metadata carries the socket path, an allocation identity,
and a random capability token. The allocation FD itself crosses the socket with
`SCM_RIGHTS`; it is never serialized. A private `/tmp` path is rejected because
container mount namespaces can resolve the same text to different directories.

## Mapping and view contract

The sidecar imports one mapping per `(device UUID, allocation ID)`. Multiple HMA
tensor aliases increment references to that mapping and the final alias release
performs, in order:

1. stream/device synchronization;
2. `cuMemUnmap`;
3. `cuMemAddressFree`; and
4. `cuMemRelease`.

Tensor descriptors preserve allocation extent, allocation-relative storage
offset, physical storage extent, shape, stride, dtype, and device UUID. All view
metadata is validated before mapping and again before reconstruction.

Unregister drops tensor aliases and mappings, then runs allocator collection.
It deliberately does not call `cudaDeviceReset`; small CUDA contexts remain
available so the same sidecar PID can register fresh worker allocations.

## Transfer-mode fallback

In `auto` mode, an exporter failure before `REGISTER_KV_CACHE` is sent may
select engine-driven transfer. Explicit `lmcache_driven` mode fails closed.
After a registration exists, the context never changes transfer mode during a
request.
