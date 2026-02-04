"""Shared utilities for RadiObject."""

from __future__ import annotations

import json

import numpy as np


def affine_to_list(affine: np.ndarray) -> list[list[float]]:
    """Convert numpy affine matrix to nested list for JSON serialization."""
    return [[float(v) for v in row] for row in affine]


def affine_to_json(affine: np.ndarray) -> str:
    """Serialize 4x4 affine matrix to JSON string."""
    return json.dumps(affine_to_list(affine))


def uri_exists(uri: str) -> bool:
    """Check if a RadiObject or VolumeCollection exists at the given URI."""
    import tiledb

    from radiobject.ctx import tdb_ctx

    try:
        with tiledb.Group(uri, "r", ctx=tdb_ctx()) as grp:
            _ = grp.meta
        return True
    except Exception:
        return False


def delete_tiledb_uri(uri: str) -> None:
    """Delete a TileDB group/array at URI (works for S3 and local)."""
    import tiledb

    from radiobject.ctx import tdb_ctx

    vfs = tiledb.VFS(ctx=tdb_ctx())
    if vfs.is_dir(uri):
        vfs.remove_dir(uri)
