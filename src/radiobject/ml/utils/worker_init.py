"""Worker initialization for DataLoader multiprocessing.

PyTorch DataLoader forks worker processes. libtiledb's internal state
(memory pools, open array handles, S3 connections) is not fork-safe, so
each worker must create its own isolated TileDB context. Without this,
workers segfault or return corrupted data.
"""

from __future__ import annotations

from radiobject.parallel import ctx_for_process


def worker_init_fn(worker_id: int) -> None:
    """Create an isolated TileDB context for this forked DataLoader worker."""
    _ = ctx_for_process()
