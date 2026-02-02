"""Worker initialization for DataLoader multiprocessing."""

from __future__ import annotations

from radiobject.parallel import create_worker_ctx


def worker_init_fn(worker_id: int) -> None:
    """Initialize TileDB context for each DataLoader worker."""
    _ = create_worker_ctx()
