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
    """Check if a TileDB object (group or array) exists at the given URI."""
    import tiledb

    from radiobject.ctx import get_tiledb_ctx

    ctx = get_tiledb_ctx()
    obj_type = tiledb.object_type(uri, ctx=ctx)
    return obj_type is not None


def delete_tiledb_uri(uri: str) -> None:
    """Delete a TileDB group/array at URI (works for S3 and local)."""
    import tiledb

    from radiobject.ctx import get_tiledb_ctx

    vfs = tiledb.VFS(ctx=get_tiledb_ctx())
    if vfs.is_dir(uri):
        vfs.remove_dir(uri)
