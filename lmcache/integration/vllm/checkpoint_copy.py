# SPDX-License-Identifier: Apache-2.0
"""Engine-owned raw-page DMA for complete recurrent checkpoint bundles."""

# Standard
from collections.abc import Callable
from typing import Any
import json
import threading

# Third Party
import torch

# First Party
from lmcache.v1.multiprocess.checkpoint_storage import checkpoint_page_groups
from lmcache.v1.multiprocess.checkpoint_transfer import (
    CheckpointTransferJob,
    UnsafeCheckpointCopyError,
)
from lmcache.v1.multiprocess.protocols.checkpoint import (
    CheckpointCapabilities,
    CheckpointLeaseResponse,
)
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
)
from lmcache.v1.platform import torch_dev


class CheckpointPageCopier:
    """Copy only caller-pinned pages using existing worker CUDA and SHM contexts.

    Args:
        page_pool: Block-outermost, contiguous uint8 physical GPU page pool.
        layout: Expected address-free target/draft cache and auxiliary layout.
        initialize_layout: Validates and binds the runner's hidden-state views
            before a checkpoint can be restored, including before first forward.
        transfer: Existing engine-driven SHM transfer context; caller-owned.
        capabilities: Checkpoint server format and SHM mapping identity.

    A completed call guarantees that all submitted DMA drained. GPU block pins,
    server leases and the registered SHM mapping must outlive that call.
    """

    def __init__(
        self,
        page_pool: torch.Tensor,
        layout: dict[str, Any],
        initialize_layout: Callable[[dict[str, Any]], None],
        transfer: EngineDrivenTransferContext,
        capabilities: CheckpointCapabilities,
    ) -> None:
        if (
            page_pool.dtype != torch.uint8
            or page_pool.ndim != 2
            or not page_pool.is_contiguous()
            or page_pool.device.type != "cuda"
            or layout.get("page_bytes") != page_pool.shape[1]
        ):
            raise ValueError("Checkpoint copy requires a validated CUDA raw-page pool")
        initialize_layout(layout)
        transfer.checkpoint_slot_views(
            capabilities, CheckpointLeaseResponse("ready"), ()
        )
        self._pool = page_pool
        self._layout = json.loads(json.dumps(layout))
        self._transfer = transfer
        self._capabilities = capabilities
        self._stream = torch_dev.Stream(device=page_pool.device)
        self._lock = threading.Lock()

    def __call__(
        self, job: CheckpointTransferJob, lease: CheckpointLeaseResponse
    ) -> None:
        """Validate the entire lease before DMA, then drain even a partial failure.

        Args:
            job: One rank's source/destination pages and optional producer event.
            lease: Server-pinned SHM byte slots for the entire manifest.

        Raises:
            ValueError: For incompatible layouts, page IDs or SHM descriptors;
                validation failures enqueue no copy.
            UnsafeCheckpointCopyError: If DMA cannot drain. Worker termination
                is required before reclaiming the transfer buffers.
        """
        payload = json.loads(job.manifest.payload)
        if payload.get("worker_layout") != self._layout:
            raise ValueError(
                "Checkpoint target/draft byte layout does not match this worker"
            )
        groups = checkpoint_page_groups(job.manifest)
        if len(groups) != len(job.block_ids):
            raise ValueError(
                "Checkpoint page groups differ from the destination bundle"
            )
        page_bytes = self._pool.shape[1]
        for group, ids in zip(groups, job.block_ids, strict=True):
            if group.page_bytes != page_bytes or len(group.positions) != len(ids):
                raise ValueError(
                    "Checkpoint page widths/counts differ from the manifest"
                )
            if any(
                type(block) is not int or not 0 < block < self._pool.shape[0]
                for block in ids
            ):
                raise ValueError(
                    "Checkpoint copy must not access null or out-of-range pages"
                )
        ids_flat = [block for ids in job.block_ids for block in ids]
        if len(ids_flat) != len(set(ids_flat)):
            raise ValueError("Checkpoint groups must own distinct physical pages")
        buffers = self._transfer.checkpoint_slot_views(
            self._capabilities,
            lease,
            tuple(tuple(group.page_bytes for _ in group.positions) for group in groups),
        )
        # One enqueue burst per lease prevents interleaving producer waits with
        # another task's copies on the shared stream. No model-thread barrier.
        with (
            self._lock,
            torch_dev.device(self._pool.device),
            torch_dev.stream(self._stream),
        ):
            try:
                if job.producer_event is not None:
                    self._stream.wait_event(job.producer_event)
                for ids, pages in zip(job.block_ids, buffers, strict=True):
                    for block, page in zip(ids, pages, strict=True):
                        if job.direction == "STORE":
                            page.copy_(self._pool[block], non_blocking=True)
                        else:
                            self._pool[block].copy_(page, non_blocking=True)
            finally:
                try:
                    self._stream.synchronize()
                except BaseException as exc:
                    raise UnsafeCheckpointCopyError(
                        "Checkpoint DMA could not drain"
                    ) from exc
