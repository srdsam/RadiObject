"""Shared utilities for RadiObject."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tiledb

if TYPE_CHECKING:
    from radiobject.volume_collection import VolumeCollection


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


def build_obs_meta_schema(df: pd.DataFrame) -> dict[str, np.dtype]:
    """Extract obs_meta schema from DataFrame, coercing object dtypes to U64.

    Always injects the system-managed obs_ids column (JSON list of volume obs_ids per subject).
    """
    schema: dict[str, np.dtype] = {}
    for col in df.columns:
        if col == "obs_subject_id":
            continue
        dtype = df[col].to_numpy().dtype
        if dtype == np.dtype("O"):
            dtype = np.dtype("U64")
        schema[col] = dtype
    # System-managed column: JSON list of volume obs_ids per subject
    schema["obs_ids"] = np.dtype("U2048")
    return schema


def write_obs_dataframe(
    uri: str,
    df: pd.DataFrame,
    ctx: tiledb.Ctx | None = None,
    columns: set[str] | None = None,
) -> None:
    """Write a DataFrame to a TileDB obs/obs_meta array (dimension-aware).

    Discovers dimensions from the TileDB schema instead of hardcoding.

    Args:
        uri: TileDB array URI.
        df: DataFrame with dimension columns as coordinates.
        ctx: TileDB context.
        columns: If provided, only write these attribute columns (for append ops).
    """
    from radiobject.ctx import get_tiledb_ctx

    if len(df) == 0:
        return

    effective_ctx = ctx if ctx else get_tiledb_ctx()

    # Discover dimensions from schema
    with tiledb.open(uri, "r", ctx=effective_ctx) as arr:
        dim_names = [arr.schema.domain.dim(i).name for i in range(arr.schema.domain.ndim)]

    # Build coordinate arrays
    coords = tuple(df[name].astype(str).to_numpy() for name in dim_names)

    # Build attribute data
    with tiledb.open(uri, "w", ctx=effective_ctx) as arr:
        data = {}
        for i in range(arr.schema.nattr):
            attr_name = arr.schema.attr(i).name
            if columns is not None and attr_name not in columns:
                continue
            if attr_name in df.columns:
                data[attr_name] = df[attr_name].to_numpy()
        arr[coords] = data


def ensure_obs_columns(
    df: pd.DataFrame,
    *,
    require_obs_id: bool = False,
    context: str = "",
) -> pd.DataFrame:
    """Validate index columns on an obs/obs_meta DataFrame.

    Requires obs_subject_id. obs_id required only when require_obs_id=True.
    """
    prefix = f"{context}: " if context else ""
    if "obs_subject_id" not in df.columns:
        raise ValueError(f"{prefix}DataFrame must contain 'obs_subject_id'")
    if require_obs_id and "obs_id" not in df.columns:
        raise ValueError(f"{prefix}DataFrame must contain 'obs_id'")
    return df


def validate_no_column_collisions(
    user_columns: set[str],
    auto_columns: set[str],
    *,
    context: str = "",
) -> None:
    """Raise on overlap between user and auto-generated columns."""
    collisions = (user_columns - {"obs_id", "obs_subject_id"}) & auto_columns
    if collisions:
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}columns collide with auto-generated imaging metadata: "
            f"{sorted(collisions)}"
        )


def merge_obs_ids(
    obs_meta: pd.DataFrame,
    collections: dict[str, VolumeCollection] | Iterable[VolumeCollection],
) -> pd.DataFrame:
    """Merge system-managed obs_ids into obs_meta from collections."""
    obs_ids_map = build_obs_ids_mapping(collections)
    obs_meta = obs_meta.merge(obs_ids_map, on="obs_subject_id", how="left")
    obs_meta["obs_ids"] = obs_meta["obs_ids"].fillna("[]")
    return obs_meta


def create_and_write_obs_meta(
    uri: str,
    obs_meta: pd.DataFrame,
    ctx: tiledb.Ctx | None = None,
) -> None:
    """Create obs_meta TileDB array and write data (single-dim, obs_subject_id PK)."""
    from radiobject.ctx import get_tiledb_ctx
    from radiobject.dataframe import Dataframe

    effective_ctx = ctx if ctx else get_tiledb_ctx()
    obs_meta_schema = build_obs_meta_schema(obs_meta)
    obs_meta_uri = f"{uri}/obs_meta"
    Dataframe.create(
        obs_meta_uri,
        schema=obs_meta_schema,
        ctx=ctx,
        index_columns=("obs_subject_id",),
    )
    write_obs_dataframe(obs_meta_uri, obs_meta, ctx=effective_ctx)


def _aggregate_obs_ids(pairs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate (obs_subject_id, obs_id) pairs into JSON obs_ids per subject."""
    if pairs.empty:
        return pd.DataFrame(columns=["obs_subject_id", "obs_ids"])
    return (
        pairs.groupby("obs_subject_id")["obs_id"]
        .apply(lambda x: json.dumps(sorted(x.tolist())))
        .reset_index()
        .rename(columns={"obs_id": "obs_ids"})
    )


def build_obs_ids_mapping(
    collections: dict[str, VolumeCollection] | Iterable[VolumeCollection],
) -> pd.DataFrame:
    """Build obs_subject_id -> obs_ids (JSON list) mapping from collections."""
    vcs = collections.values() if isinstance(collections, dict) else collections
    frames = []
    for vc in vcs:
        obs_df = vc.obs.read()
        if "obs_subject_id" in obs_df.columns and "obs_id" in obs_df.columns:
            frames.append(obs_df[["obs_subject_id", "obs_id"]])
    pairs = pd.concat(frames) if frames else pd.DataFrame(columns=["obs_subject_id", "obs_id"])
    return _aggregate_obs_ids(pairs)
