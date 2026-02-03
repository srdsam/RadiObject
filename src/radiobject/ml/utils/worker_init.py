"""Worker initialization for DataLoader multiprocessing."""

from __future__ import annotations

from radiobject.parallel import ctx_for_process


def worker_init_fn(worker_id: int) -> None:
    """Initialize TileDB context for each DataLoader worker process."""
    _ = ctx_for_process()
