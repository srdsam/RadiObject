"""Configuration models for ML training pipeline."""

from collections.abc import Callable
from enum import Enum
from typing import Any, Self

from pydantic import BaseModel, field_validator, model_validator

from radiobject.ctx import SliceOrientation


class LoadingMode(str, Enum):
    """Volume loading strategy.

    FULL_VOLUME: Load entire 3D volumes. Requires uniform shapes. Best for
        small volumes or whole-volume models.
    PATCH: Extract random 3D sub-arrays via TileDB slice reads. Supports
        heterogeneous shapes. Primary mode for large-volume training.
    SLICE_2D: Extract 2D slices along a configurable orientation axis.
        Requires uniform shapes.
    """

    FULL_VOLUME = "full_volume"
    PATCH = "patch"
    SLICE_2D = "slice_2d"


class DatasetConfig(BaseModel):
    """Configuration for ML datasets."""

    loading_mode: LoadingMode = LoadingMode.FULL_VOLUME
    patch_size: tuple[int, int, int] | None = None
    patches_per_volume: int = 1
    slice_orientation: SliceOrientation = SliceOrientation.AXIAL
    modalities: list[str] | None = None
    label_column: str | None = None
    value_filter: str | None = None

    @model_validator(mode="after")
    def validate_patch_config(self) -> Self:
        """Validate patch configuration consistency."""
        if self.loading_mode == LoadingMode.PATCH and self.patch_size is None:
            raise ValueError("patch_size required when loading_mode is PATCH")
        return self

    @field_validator("patches_per_volume")
    @classmethod
    def validate_patches_per_volume(cls, v: int) -> int:
        """Ensure patches_per_volume is positive."""
        if v < 1:
            raise ValueError("patches_per_volume must be >= 1")
        return v

    model_config = {"frozen": True}


# Dimension index for each slice orientation: AXIAL→Z(2), SAGITTAL→X(0), CORONAL→Y(1)
SLICE_DIM: dict[SliceOrientation, int] = {
    SliceOrientation.AXIAL: 2,
    SliceOrientation.SAGITTAL: 0,
    SliceOrientation.CORONAL: 1,
}

# Volume method for each slice orientation
SLICE_METHODS: dict[SliceOrientation, Callable[..., Any]] = {
    SliceOrientation.AXIAL: lambda vol, idx: vol.axial(idx),
    SliceOrientation.SAGITTAL: lambda vol, idx: vol.sagittal(idx),
    SliceOrientation.CORONAL: lambda vol, idx: vol.coronal(idx),
}


def slice_dim_index(orientation: SliceOrientation) -> int:
    """Return the volume shape dimension index for the given slice orientation."""
    return SLICE_DIM[orientation]
