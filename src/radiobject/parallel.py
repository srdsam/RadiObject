"""Parallel execution utilities for RadiObject I/O operations."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TypeVar

import tiledb

from radiobject.ctx import radi_cfg, tdb_ctx

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class WriteResult:
    """Result of a parallel volume write operation."""

    index: int
    uri: str
    obs_id: str
    success: bool
    error: Exception | None = None


def ctx_for_threads(ctx: tiledb.Ctx | None = None) -> tiledb.Ctx:
    """Return context for thread pool workers.

    TileDB is thread-safe. Sharing a context across threads enables
    metadata caching, reducing I/O for repeated operations.
    """
    return ctx if ctx else tdb_ctx()


def ctx_for_process(base_ctx: tiledb.Ctx | None = None) -> tiledb.Ctx:
    """Create new context for a forked process.

    Forked processes (e.g., DataLoader workers) have separate memory
    and cannot share TileDB contexts with the parent process.
    """
    if base_ctx is not None:
        return tiledb.Ctx(base_ctx.config())
    return radi_cfg().to_tiledb_ctx()


def map_on_threads(
    fn: Callable[[T], R],
    items: Iterable[T],
    max_workers: int | None = None,
    progress: bool = False,
    desc: str | None = None,
) -> list[R]:
    """Execute fn on each item using thread pool, preserving order."""
    items_list = list(items)
    if not items_list:
        return []

    default_workers = radi_cfg().read.max_workers
    workers = max_workers or min(default_workers, len(items_list))
    results: list[R | None] = [None] * len(items_list)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fn, item): idx for idx, item in enumerate(items_list)}

        completed = as_completed(futures)
        if progress:
            from tqdm.auto import tqdm

            completed = tqdm(completed, total=len(futures), desc=desc, unit="vol")

        for future in completed:
            idx = futures[future]
            results[idx] = future.result()

    return results  # type: ignore
