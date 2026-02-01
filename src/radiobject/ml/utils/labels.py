"""Label loading utilities for ML datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from radiobject._types import LabelSource

if TYPE_CHECKING:
    from radiobject.ml.reader import VolumeReader


def load_labels(
    reader: VolumeReader,
    labels: LabelSource,
    obs_df: pd.DataFrame | None = None,
) -> dict[int, Any] | None:
    """Load labels from various sources, indexed by volume position.

    Args:
        reader: VolumeReader for the primary collection (used to get obs_ids).
        labels: Label source - see LabelSource type for options.
        obs_df: Pre-loaded obs DataFrame from the collection. Required when
            labels is a column name string.

    Returns:
        Dict mapping volume index to label value, or None if labels is None.

    Raises:
        ValueError: If label column not found or DataFrame missing required columns.
    """
    if labels is None:
        return None

    n_volumes = len(reader)
    result: dict[int, Any] = {}

    if isinstance(labels, str):
        # Column name in obs DataFrame
        if obs_df is None:
            raise ValueError(
                "obs_df required when labels is a column name. "
                "Pass collection.obs.read() as obs_df."
            )

        if labels not in obs_df.columns:
            raise ValueError(f"Label column '{labels}' not found in obs DataFrame")

        # Build lookup by obs_id
        if "obs_id" in obs_df.columns:
            label_lookup = dict(zip(obs_df["obs_id"], obs_df[labels]))
        elif obs_df.index.name == "obs_id":
            label_lookup = obs_df[labels].to_dict()
        else:
            raise ValueError("obs DataFrame must have 'obs_id' column or index")

        for idx in range(n_volumes):
            obs_id = reader.get_obs_id(idx)
            if obs_id in label_lookup:
                result[idx] = label_lookup[obs_id]

    elif isinstance(labels, pd.DataFrame):
        # DataFrame with obs_id mapping
        if "obs_id" in labels.columns:
            # obs_id as column - use first non-obs_id column as label
            label_cols = [c for c in labels.columns if c != "obs_id"]
            if not label_cols:
                raise ValueError("Labels DataFrame must have at least one label column")
            label_col = label_cols[0]
            label_lookup = dict(zip(labels["obs_id"], labels[label_col]))
        elif labels.index.name == "obs_id" or labels.index.dtype == object:
            # obs_id as index
            label_col = labels.columns[0]
            label_lookup = labels[label_col].to_dict()
        else:
            raise ValueError("Labels DataFrame must have 'obs_id' as column or index")

        for idx in range(n_volumes):
            obs_id = reader.get_obs_id(idx)
            if obs_id in label_lookup:
                result[idx] = label_lookup[obs_id]

    elif isinstance(labels, dict):
        # Direct mapping from obs_id to label
        for idx in range(n_volumes):
            obs_id = reader.get_obs_id(idx)
            if obs_id in labels:
                result[idx] = labels[obs_id]

    elif callable(labels):
        # Function that takes obs_id and returns label
        for idx in range(n_volumes):
            obs_id = reader.get_obs_id(idx)
            result[idx] = labels(obs_id)

    else:
        raise TypeError(
            f"labels must be str, DataFrame, dict, callable, or None, got {type(labels)}"
        )

    return result if result else None
