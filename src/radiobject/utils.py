"""Shared utilities for RadiObject."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import tiledb


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


def build_obs_meta_schema(df: "pd.DataFrame") -> dict[str, "np.dtype"]:
    """Extract obs_meta schema from DataFrame, coercing object dtypes to U64."""
    import numpy as np

    schema: dict[str, np.dtype] = {}
    for col in df.columns:
        if col in ("obs_subject_id", "obs_id"):
            continue
        dtype = df[col].to_numpy().dtype
        if dtype == np.dtype("O"):
            dtype = np.dtype("U64")
        schema[col] = dtype
    return schema


def write_obs_dataframe(
    uri: str,
    df: "pd.DataFrame",
    ctx: "tiledb.Ctx | None" = None,
    columns: set[str] | None = None,
) -> None:
    """Write a DataFrame to a TileDB obs/obs_meta array.

    Args:
        uri: TileDB array URI.
        df: DataFrame with obs_subject_id and obs_id as dimension coordinates.
        ctx: TileDB context.
        columns: If provided, only write these attribute columns (for append ops).
    """
    import tiledb

    from radiobject.ctx import get_tiledb_ctx

    if len(df) == 0:
        return

    effective_ctx = ctx if ctx else get_tiledb_ctx()
    obs_subject_ids = df["obs_subject_id"].astype(str).to_numpy()
    obs_ids = df["obs_id"].astype(str).to_numpy() if "obs_id" in df.columns else obs_subject_ids
    with tiledb.open(uri, "w", ctx=effective_ctx) as arr:
        attr_data = {
            col: df[col].to_numpy()
            for col in df.columns
            if col not in ("obs_subject_id", "obs_id") and (columns is None or col in columns)
        }
        arr[obs_subject_ids, obs_ids] = attr_data
