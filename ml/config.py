"""Configuration models for ML training pipeline."""

from enum import Enum
from typing import Self

from pydantic import BaseModel, field_validator, model_validator


class LoadingMode(str, Enum):
    """Volume loading strategy."""

    FULL_VOLUME = "full_volume"
    PATCH = "patch"
    SLICE_2D = "slice_2d"


class CacheStrategy(str, Enum):
    """Caching strategy for dataset samples."""

    NONE = "none"
    IN_MEMORY = "in_memory"
    DISK = "disk"


class DatasetConfig(BaseModel):
    """Configuration for RadiObjectDataset."""

    loading_mode: LoadingMode = LoadingMode.FULL_VOLUME
    cache_strategy: CacheStrategy = CacheStrategy.NONE
    patch_size: tuple[int, int, int] | None = None
    patches_per_volume: int = 1
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
