"""Shared type aliases for the radiobject package."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

# Scalar values storable in TileDB obs attributes
AttrValue = int | float | bool | str

# Transform result: either just a volume, or (volume, obs_updates_dict)
# When obs_updates is provided, those key-value pairs are merged into the obs row on write.
TransformResult = Union[
    npt.NDArray[np.floating],
    tuple[npt.NDArray[np.floating], dict[str, AttrValue]],
]

# Volume transform function: (volume, obs_row) -> volume OR (volume, obs_updates)
TransformFn = Callable[[npt.NDArray[np.floating], pd.Series], TransformResult]

# Batch transform function: list[(volume, obs_row)] -> list[volume | (volume, obs_updates)]
BatchTransformFn = Callable[
    [list[tuple[npt.NDArray[np.floating], pd.Series]]],
    list[TransformResult],
]

# Flexible label specification for ML datasets
LabelSource = str | pd.DataFrame | dict[str, Any] | Callable[[str], Any] | None


def normalize_transform_result(result: Any) -> tuple[Any, dict[str, AttrValue]]:
    """Normalize a transform result to (value, obs_updates).

    Handles both plain values and (value, obs_updates_dict) tuples.
    """
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result[0], result[1]
    return result, {}
