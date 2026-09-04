// SPDX-License-Identifier: Apache-2.0
#define _GNU_SOURCE

#include <cuda.h>
#include <dlfcn.h>
#include <pthread.h>

/*
 * vLLM owns allocation, mapping, sleep/wake, and release. This interposer only
 * changes cuMemCreate's requested handle type so CUDA 13.3 allocations can be
 * exported as POSIX FDs. In particular, replace driver-selected FABRIC handles:
 * FABRIC handles cannot be transported through same-UID SCM_RIGHTS.
 */
typedef CUresult (*cuMemCreate_fn)(CUmemGenericAllocationHandle *, size_t,
                                  const CUmemAllocationProp *,
                                  unsigned long long);

static cuMemCreate_fn real_cuMemCreate;
static pthread_once_t resolve_once = PTHREAD_ONCE_INIT;

static void resolve_driver_symbol(void) {
  /*
   * CUDA may be loaded in an extension's local dependency scope, where
   * RTLD_NEXT cannot see it. Resolve against the driver DSO explicitly.
   */
  void *driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (driver != NULL) {
    real_cuMemCreate = (cuMemCreate_fn)dlsym(driver, "cuMemCreate");
  }
}

CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                     const CUmemAllocationProp *prop,
                     unsigned long long flags) {
  pthread_once(&resolve_once, resolve_driver_symbol);
  if (real_cuMemCreate == NULL) {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
  if (prop == NULL ||
      prop->requestedHandleTypes == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    return real_cuMemCreate(handle, size, prop, flags);
  }

  CUmemAllocationProp shareable = *prop;
  shareable.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  return real_cuMemCreate(handle, size, &shareable, flags);
}
