# SPDX-License-Identifier: Apache-2.0
"""Scheduler ownership and all-rank publication of external recurrent checkpoints."""

# Standard
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
import json
import uuid

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.checkpoint_identity import (
    CheckpointTokenRoots,
    checkpoint_namespace,
)
from lmcache.v1.multiprocess.checkpoint_index import CheckpointManifest
from lmcache.v1.multiprocess.checkpoint_storage import checkpoint_page_groups
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocols.base import RequestType

logger = init_logger(__name__)

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.boundary_checkpoint import BoundaryCheckpoint
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import Request


@dataclass(frozen=True)
class CheckpointEngineTask:
    """Ephemeral scheduler-to-worker transfer command, never persisted to storage."""

    task_id: str
    manifest: CheckpointManifest
    direction: Literal["STORE", "RETRIEVE"]
    block_ids: tuple[tuple[int, ...], ...]


@dataclass
class _PendingTask:
    task: CheckpointEngineTask
    checkpoint: "BoundaryCheckpoint"
    request_id: str
    begin: MessagingFuture[bool] | None = None
    acknowledgements: dict[int, bool] = field(default_factory=dict)
    sent: bool = False


@dataclass
class _Lookup:
    roots: CheckpointTokenRoots
    future: MessagingFuture[CheckpointManifest | None]
    done: bool = False
    task_id: str | None = None
    checkpoint_id: int | None = None


class CheckpointSchedulerBridge:
    """Keep imports private and store sources pinned until every rank completes.

    Args:
        manager: vLLM allocator with request-boundary checkpoint support enabled.
        client: Shared thread-safe LMCache queue client; no ownership transfer.
        identity: Immutable target/draft/source revisions and parallel geometry.
            Worker layout is added after all ranks report identical descriptors.
        world_size: Number of engine ranks contributing to one atomic generation.
        max_tasks: Admission limit for collective stores and restores.

    The scheduler must keep issuing connector-only steps while has_pending is
    true, including after a producer request finishes. Request cancellation
    never releases pages still referenced by an admitted worker copy.
    """

    def __init__(
        self,
        manager: "KVCacheManager",
        client: MessageQueueClient,
        identity: dict[str, object],
        world_size: int,
        *,
        max_tasks: int = 32,
    ) -> None:
        if manager.boundary_checkpoints is None or world_size < 1 or max_tasks < 1:
            raise ValueError(
                "Semantic transfers require a boundary allocator and ranks"
            )
        self._manager = manager
        self._cache = manager.boundary_checkpoints
        self._client = client
        self._identity = dict(identity)
        self._world_size = world_size
        self._max_tasks = max_tasks
        self._layouts: dict[int, dict[str, Any]] = {}
        self._layout: dict[str, Any] | None = None
        self._lookups: dict[str, _Lookup] = {}
        self._tasks: dict[str, _PendingTask] = {}
        self._cancelled: set[str] = set()

    @property
    def has_pending(self) -> bool:
        """Whether connector-only steps are needed to negotiate or drain copies."""
        return self._layout is None or bool(self._tasks)

    def handles(self, request: "Request") -> bool:
        """Require a text request whose complete weight identity is authenticated.

        Per-request LoRA content revisions are not part of the manifest namespace;
        such requests may use GPU-local caching but never this external directory.
        """
        return request.lora_request is None and self._cache.supports_request(request)

    def accept_layouts(self, layouts: dict[int, dict[str, Any]]) -> None:
        """Validate identical byte interpretation across all contributing ranks.

        Args:
            layouts: Address-free descriptors emitted by worker initialization.

        Raises:
            ValueError: If any rank is invalid or changes its byte layout.
        """
        for rank, layout in layouts.items():
            if not 0 <= rank < self._world_size:
                raise ValueError("Checkpoint layout came from an unknown rank")
            if self._layouts and layout != next(iter(self._layouts.values())):
                raise ValueError("Checkpoint byte layouts must match on every rank")
            self._layouts[rank] = layout
        if len(self._layouts) == self._world_size:
            self._layout = next(iter(self._layouts.values()))
            self._identity["layout"] = self._layout

    def poll_prefix(self, request: "Request") -> bool:
        """Start/poll an import and permit scheduling only after collective completion.

        Args:
            request: Waiting request whose prefix has not been admitted yet.

        Returns:
            False while layout negotiation, lookup or H2D is pending. True after
            a miss or a fully published import, which normal GPU lookup may use.
            Reusing a finished request's ID waits for its admitted copies to
            drain, so their completion cannot cancel or erase another lookup.
        """
        if request.request_id in self._cancelled:
            return False
        if not self.handles(request) or not self._manager.prefix_cache_lookup_enabled(
            request
        ):
            return True
        local = self._cache.find(request, request.num_tokens)
        if local is not None and local.num_tokens == request.num_tokens:
            return True
        if self._layout is None:
            return False
        state = self._lookups.get(request.request_id)
        if state is None:
            roots = self._roots(request)
            state = _Lookup(
                roots,
                self._client.submit_request(RequestType.CHECKPOINT_FIND, [roots.roots]),
            )
            self._lookups[request.request_id] = state
            return False
        if state.done:
            return True
        if state.task_id is not None or not state.future.query():
            return False
        manifest = state.future.result()
        if manifest is None or (
            local is not None and manifest.prefix.num_tokens <= local.num_tokens
        ):
            state.done = True
            return True
        if len(self._tasks) >= self._max_tasks:
            state.done = True
            return True
        try:
            payload, positions = self._validate_manifest(manifest, state.roots)
            checkpoint = self._manager.reserve_external_boundary_checkpoint(
                request,
                manifest.prefix.num_tokens,
                positions,
                draft_prefix_len=payload["draft_prefix_len"],
                kind=payload["kind"],
                num_ranks=self._world_size,
            )
        except (ValueError, KeyError, TypeError):
            state.done = True
            return True
        if checkpoint is None:
            state.done = True
            return True
        task = self._make_task(manifest, checkpoint, "RETRIEVE")
        self._tasks[task.task_id] = _PendingTask(task, checkpoint, request.request_id)
        state.task_id = task.task_id
        return False

    def external_tokens(self, request: "Request") -> int:
        """Attribute only this request's selected imported bundle to external cache."""
        state = self._lookups.get(request.request_id)
        checkpoint = request.boundary_checkpoint
        if (
            state is not None
            and checkpoint is not None
            and state.checkpoint_id == checkpoint.checkpoint_id
        ):
            return checkpoint.num_tokens
        return 0

    def store(self, request: "Request", checkpoint: "BoundaryCheckpoint") -> None:
        """Pin a committed checkpoint and asynchronously stage its storage generation.

        Args:
            request: Producer with exact target tokens, including committed response.
            checkpoint: Published immutable GPU bundle, not a live request block table.
        """
        if (
            not self.handles(request)
            or self._layout is None
            or len(self._tasks) >= self._max_tasks
        ):
            return
        pinned = self._cache.acquire(checkpoint.checkpoint_id)
        if pinned is None:
            return
        try:
            positions = self._manager.boundary_checkpoint_page_positions(
                checkpoint.num_tokens
            )
            page_bytes = self._layout["page_bytes"]
            groups = [
                {
                    "name": f"engine-kv-group:{i}",
                    "page_bytes": page_bytes,
                    "positions": list(pages),
                }
                for i, pages in enumerate(positions)
            ]
            groups.append(
                {
                    "name": "target-draft-auxiliary",
                    "page_bytes": page_bytes,
                    "positions": [0],
                }
            )
            manifest = CheckpointManifest(
                uuid.uuid4().hex,
                self._roots(request).prefix(checkpoint.num_tokens),
                self._world_size,
                json.dumps(
                    {
                        "schema_version": 1,
                        "worker_layout": self._layout,
                        "page_groups": groups,
                        "draft_prefix_len": checkpoint.draft_prefix_len,
                        "kind": checkpoint.kind,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode(),
            )
            task = self._make_task(manifest, checkpoint, "STORE")
            self._tasks[task.task_id] = _PendingTask(
                task,
                pinned,
                request.request_id,
                begin=self._client.submit_request(
                    RequestType.CHECKPOINT_BEGIN, [manifest]
                ),
            )
        except BaseException:
            self._cache.release(pinned)
            raise

    def take_tasks(self) -> list[CheckpointEngineTask]:
        """Return each admitted collective copy exactly once, without waiting on RPC."""
        tasks = []
        for task_id, pending in tuple(self._tasks.items()):
            if pending.sent:
                continue
            if pending.begin is not None:
                if not pending.begin.query():
                    continue
                try:
                    accepted = pending.begin.result()
                except Exception:
                    logger.exception("Recurrent checkpoint begin operation failed")
                    # No worker has received this task. Its GPU source pin can
                    # be released even if the server's begin reply was lost.
                    try:
                        self._client.submit_request(
                            RequestType.CHECKPOINT_ABORT,
                            [pending.task.manifest.generation],
                        )
                    except Exception:
                        logger.exception("Recurrent checkpoint abort submission failed")
                    accepted = False
                if not accepted:
                    self._cache.release(pending.checkpoint)
                    del self._tasks[task_id]
                    if pending.request_id in self._cancelled:
                        self.finish_request(pending.request_id)
                    continue
            pending.sent = True
            tasks.append(pending.task)
        return tasks

    def complete(self, results: dict[str, dict[int, bool]]) -> None:
        """Aggregate drained worker acknowledgements and publish or discard atomically.

        Args:
            results: Task ID to distinct rank results. True proves completed bytes;
                False proves that rank will no longer access the transfer's pages.

        Raises:
            ValueError: For duplicate or unknown rank acknowledgements.
        """
        for task_id, ranks in results.items():
            pending = self._tasks.get(task_id)
            if pending is None:
                raise ValueError("Checkpoint completion identifies an unknown task")
            for rank, success in ranks.items():
                if rank in pending.acknowledgements or not 0 <= rank < self._world_size:
                    raise ValueError("Duplicate or invalid checkpoint rank completion")
                pending.acknowledgements[rank] = success
            if len(pending.acknowledgements) != self._world_size:
                continue
            if pending.task.direction == "STORE":
                if not all(pending.acknowledgements.values()):
                    self._client.submit_request(
                        RequestType.CHECKPOINT_ABORT, [pending.task.manifest.generation]
                    )
                self._cache.release(pending.checkpoint)
            else:
                state = self._lookups.get(pending.request_id)
                if (
                    all(pending.acknowledgements.values())
                    and pending.request_id not in self._cancelled
                ):
                    for rank in range(self._world_size):
                        published = (
                            self._manager.acknowledge_external_boundary_checkpoint(
                                pending.checkpoint.checkpoint_id, rank
                            )
                        )
                    if published and state is not None:
                        state.checkpoint_id = pending.checkpoint.checkpoint_id
                else:
                    self._manager.discard_external_boundary_checkpoint(
                        pending.checkpoint.checkpoint_id
                    )
                if state is not None:
                    state.done = True
            del self._tasks[task_id]
            if pending.request_id in self._cancelled:
                self.finish_request(pending.request_id)

    def finish_request(self, request_id: str) -> None:
        """Forget lookup state, retaining copy pins until every admitted task drains."""
        self._lookups.pop(request_id, None)
        if any(task.request_id == request_id for task in self._tasks.values()):
            self._cancelled.add(request_id)
        else:
            self._cancelled.discard(request_id)

    def _roots(self, request: "Request") -> CheckpointTokenRoots:
        return CheckpointTokenRoots.build(
            checkpoint_namespace(self._identity, request.cache_salt or ""),
            request.all_token_ids,
        )

    def _validate_manifest(
        self, manifest: CheckpointManifest, roots: CheckpointTokenRoots
    ) -> tuple[dict[str, Any], tuple[tuple[int, ...], ...]]:
        if manifest.world_size != self._world_size or manifest.prefix != roots.prefix(
            manifest.prefix.num_tokens
        ):
            raise ValueError("Checkpoint rank count or token prefix does not match")
        payload = json.loads(manifest.payload)
        groups = checkpoint_page_groups(manifest)
        if (
            payload.get("worker_layout") != self._layout
            or groups[-1].name != "target-draft-auxiliary"
            or groups[-1].positions != (0,)
        ):
            raise ValueError("Checkpoint layout or auxiliary payload is incompatible")
        if any(group.page_bytes != self._layout["page_bytes"] for group in groups):
            raise ValueError(
                "Checkpoint physical page width differs from the worker pool"
            )
        if [group.name for group in groups[:-1]] != [
            f"engine-kv-group:{i}" for i in range(len(groups) - 1)
        ]:
            raise ValueError("Checkpoint engine group ordering is incompatible")
        return payload, tuple(group.positions for group in groups[:-1])

    def _make_task(
        self,
        manifest: CheckpointManifest,
        checkpoint: "BoundaryCheckpoint",
        direction: Literal["STORE", "RETRIEVE"],
    ) -> CheckpointEngineTask:
        return CheckpointEngineTask(
            uuid.uuid4().hex,
            manifest,
            direction,
            tuple(
                tuple(block for block in group if block)
                for group in checkpoint.block_ids
            )
            + (checkpoint.auxiliary_block_ids,),
        )
