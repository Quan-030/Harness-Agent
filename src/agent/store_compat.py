# src/agent/store_compat.py
"""MongoDBStore offset-compatibility shim.

langgraph's MongoDBStore.search raises NotImplementedError for any non-zero
offset, but deepagents' StoreBackend._search_store_paginated paginates the
/persisted-skills/ routes with offset += page_size (100). Once a
single namespace holds >= 100 items, every ls/grep/glob on those routes crashes.
InMemoryStore (the previous STORE) supported offset, so this shim restores it.
"""
from __future__ import annotations

from typing import Any, Optional

from langgraph.store.base import SearchItem
from langgraph.store.mongodb import MongoDBStore


class OffsetCompatMongoDBStore(MongoDBStore):
    """MongoDBStore that emulates `offset` in search via fetch-and-slice.

    Overriding `search` is enough to cover the async path too: asearch routes
    through abatch -> batch -> self.search.
    """

    def search(  # noqa: A002 - mirrors BaseStore.search signature (`filter`)
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: Optional[str] = None,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: Optional[bool] = None,
        **kwargs: Any,
    ) -> list[SearchItem]:
        if not offset:
            return super().search(
                namespace_prefix,
                query=query,
                filter=filter,
                limit=limit,
                offset=0,
                refresh_ttl=refresh_ttl,
                **kwargs,
            )
        # MongoDBStore can't skip server-side, so fetch [0, offset+limit) and
        # slice the window. Natural order is stable for a non-mutating collection
        # across the brief pagination loop, so slices stay disjoint and complete.
        page = super().search(
            namespace_prefix,
            query=query,
            filter=filter,
            limit=offset + limit,
            offset=0,
            refresh_ttl=refresh_ttl,
            **kwargs,
        )
        return page[offset : offset + limit]
