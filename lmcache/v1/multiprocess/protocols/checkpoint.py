# SPDX-License-Identifier: Apache-2.0
"""Metadata-only RPC for all-rank recurrent checkpoint publication.

Tensor bytes stay in the shared-memory pool. A ready lease is not a completed
GPU transfer; workers acknowledge finish only after their CUDA event drains.
"""

# Standard
from dataclasses import dataclass
from typing import Literal

# First Party
from lmcache.v1.multiprocess.checkpoint_index import (
    CheckpointManifest,
    CheckpointPrefix,
)
from lmcache.v1.multiprocess.protocols.base import HandlerType, ProtocolDefinition


@dataclass(frozen=True)
class CheckpointCapabilities:
    """Server format, SHM identity and durable-index capability."""

    format_version: int = 0
    shm_name: str = ""
    pool_size: int = 0
    durable_index: bool = False


@dataclass(frozen=True)
class CheckpointLeaseResponse:
    """Transfer admission with per-group byte offsets and lengths in SHM.

    ``pending`` requires polling; ``miss`` permits no copy. Only ``ready``
    includes pinned slots. A worker must validate slot widths against its
    manifest and keep the lease until every submitted copy has completed.
    """

    status: Literal["pending", "miss", "ready"]
    lease_id: str = ""
    slots: tuple[tuple[tuple[int, int], ...], ...] = ()


REQUEST_NAMES = [
    "CHECKPOINT_CAPABILITIES",
    "CHECKPOINT_FIND",
    "CHECKPOINT_BEGIN",
    "CHECKPOINT_ABORT",
    "CHECKPOINT_PREPARE_STORE",
    "CHECKPOINT_FINISH_STORE",
    "CHECKPOINT_BEGIN_RETRIEVE",
    "CHECKPOINT_POLL_RETRIEVE",
    "CHECKPOINT_FINISH_RETRIEVE",
    "CHECKPOINT_CANCEL_RETRIEVE",
]


def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
    """Return typed checkpoint RPC definitions without tensor payload classes."""
    blocking = HandlerType.BLOCKING
    return {
        "CHECKPOINT_CAPABILITIES": ProtocolDefinition(
            [], CheckpointCapabilities, HandlerType.SYNC
        ),
        "CHECKPOINT_FIND": ProtocolDefinition(
            [tuple[CheckpointPrefix, ...]], CheckpointManifest | None, blocking
        ),
        "CHECKPOINT_BEGIN": ProtocolDefinition([CheckpointManifest], bool, blocking),
        "CHECKPOINT_ABORT": ProtocolDefinition([str], bool, blocking),
        "CHECKPOINT_PREPARE_STORE": ProtocolDefinition(
            [CheckpointManifest, int], CheckpointLeaseResponse, blocking
        ),
        "CHECKPOINT_FINISH_STORE": ProtocolDefinition([str, bool], bool, blocking),
        "CHECKPOINT_BEGIN_RETRIEVE": ProtocolDefinition(
            [CheckpointManifest, int], CheckpointLeaseResponse, blocking
        ),
        "CHECKPOINT_POLL_RETRIEVE": ProtocolDefinition(
            [str], CheckpointLeaseResponse, blocking
        ),
        "CHECKPOINT_FINISH_RETRIEVE": ProtocolDefinition([str], bool, blocking),
        "CHECKPOINT_CANCEL_RETRIEVE": ProtocolDefinition([str], bool, blocking),
    }
