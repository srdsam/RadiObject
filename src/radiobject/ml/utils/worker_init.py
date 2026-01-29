"""Worker initialization for DataLoader multiprocessing."""

from __future__ import annotations


from radiobject.parallel import create_worker_ctx


def worker_init_fn(worker_id: int) -> None:
    """Initialize TileDB context for each DataLoader worker.

    This ensures each worker process has its own thread-safe TileDB context,
    preventing conflicts when multiple workers access TileDB arrays concurrently.

    Args:
        worker_id: Worker process ID (0-indexed).
    """
    ctx = create_worker_ctx()

    import threading

    local = threading.local()
    local.tiledb_ctx = ctx
