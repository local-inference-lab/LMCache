# SPDX-License-Identifier: Apache-2.0
"""
Store recurrent checkpoints to L2 only after their first reuse.

A request-boundary recurrent checkpoint holds the complete recurrent state,
about 167 MB for Qwen3.8 Flash Next at TP1, and most are never restored: the
next chat turn usually cannot match the previous prompt or response endpoint
because the chat template rewrites that history. Writing every checkpoint
through to L2 therefore wears flash storage for data that is not read back.

This policy keeps new checkpoint pages in L1 and stores a page to L2 the first
time a completed restore reads it. Ordinary KV chunks keep write-through
behavior. After a restart or an L1 eviction, only checkpoints that were reused
at least once remain available; the checkpoint scheduler then falls back to
the longest one that still exists.

Select it with ``--l2-store-policy checkpoint_on_reuse``.
"""

# First Party
from lmcache.v1.distributed.api import ObjectKey, is_recurrent_checkpoint_key
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
    DefaultStorePolicy,
    register_store_policy,
)


class CheckpointReuseStorePolicy(DefaultStorePolicy):
    """
    Write ordinary keys through; store checkpoint pages on their first reuse.

    L1 deletions follow ``DefaultStorePolicy``: keys stay in L1.
    """

    def uses_reuse_admission(self) -> bool:
        """
        Report that checkpoint pages are stored only after reuse.

        Returns:
            True.
        """
        return True

    def select_store_targets(
        self,
        keys: list[ObjectKey],
        adapters: list[AdapterDescriptor],
    ) -> dict[int, list[ObjectKey]]:
        """
        Store ordinary keys to every adapter and hold back checkpoint pages.

        Args:
            keys: Keys that were just written to L1.
            adapters: Descriptors of available L2 adapters.

        Returns:
            Mapping from every adapter index to the keys that are not
            recurrent checkpoint pages.
        """
        ordinary = [key for key in keys if not is_recurrent_checkpoint_key(key)]
        return {adapter.index: list(ordinary) for adapter in adapters}

    def select_reuse_targets(
        self,
        keys: list[ObjectKey],
        adapters: list[AdapterDescriptor],
    ) -> dict[int, list[ObjectKey]]:
        """
        Store reused checkpoint pages to every adapter.

        Ordinary keys were already stored when they were written.

        Args:
            keys: Keys that a completed restore read from L1.
            adapters: Descriptors of available L2 adapters.

        Returns:
            Mapping from every adapter index to the recurrent checkpoint
            pages among ``keys``.
        """
        pages = [key for key in keys if is_recurrent_checkpoint_key(key)]
        return {adapter.index: list(pages) for adapter in adapters}


register_store_policy("checkpoint_on_reuse", CheckpointReuseStorePolicy)
