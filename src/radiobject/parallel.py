"""Parallel execution utilities for RadiObject I/O operations."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TypeVar

import tiledb

from radiobject.ctx import get_config

T = TypeVar("T")
R = TypeVar("R")

DEFAULT_MAX_WORKERS = 4


@dataclass(frozen=True)
class WriteResult:
    """Result of a parallel volume write operation."""

    index: int
    uri: str
    obs_id: str
    success: bool
    error: Exception | None = None


def create_worker_ctx(base_ctx: tiledb.Ctx | None = None) -> tiledb.Ctx:
    """Create a thread-safe TileDB context for worker threads."""
    if base_ctx is not None:
        return tiledb.Ctx(base_ctx.config())
    return get_config().to_tiledb_ctx()


def map_on_threads(
    fn: Callable[[T], R],
    items: Iterable[T],
    max_workers: int | None = None,
) -> list[R]:
    """Execute fn on each item using thread pool, preserving order."""
    items_list = list(items)
    if not items_list:
        return []

    workers = max_workers or min(DEFAULT_MAX_WORKERS, len(items_list))
    results: list[R | None] = [None] * len(items_list)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fn, item): idx for idx, item in enumerate(items_list)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    return results  # type: ignore
