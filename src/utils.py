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
