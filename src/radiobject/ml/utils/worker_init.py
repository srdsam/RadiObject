"""Worker initialization for DataLoader multiprocessing."""

from __future__ import annotations

from radiobject.parallel import create_worker_ctx


def worker_init_fn(worker_id: int) -> None:
    """Initialize TileDB context for each DataLoader worker.

    Pre-warms the process-level context cache in VolumeReader._get_ctx().
    The context is cached by (pid, config_hash) key, ensuring each worker
    process has its own thread-safe TileDB context.

    Args:
        worker_id: Worker process ID (0-indexed), unused but required by DataLoader.
    """
    _ = create_worker_ctx()
