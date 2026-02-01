"""Shared type aliases for the radiobject package."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

# Volume transform function: (X, Y, Z) -> (X', Y', Z')
TransformFn = Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]

# Scalar values storable in TileDB obs attributes
AttrValue = int | float | bool | str

# Flexible label specification for ML datasets
LabelSource = str | pd.DataFrame | dict[str, Any] | Callable[[str], Any] | None
