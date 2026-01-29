"""Unit tests for ML datasets."""

from __future__ import annotations

import pytest
import torch

from ml.cache import InMemoryCache
from ml.config import DatasetConfig, LoadingMode
from ml.datasets.volume_dataset import RadiObjectDataset


class TestRadiObjectDataset:
    """Tests for RadiObjectDataset."""

    def test_dataset_length(self, ml_dataset: RadiObjectDataset) -> None:
        """Test dataset reports correct length."""
        assert len(ml_dataset) == 3

    def test_getitem_returns_dict(self, ml_dataset: RadiObjectDataset) -> None:
        """Test __getitem__ returns dict with expected keys."""
        sample = ml_dataset[0]
        assert isinstance(sample, dict)
        assert "image" in sample
        assert "idx" in sample

    def test_image_shape(self, ml_dataset: RadiObjectDataset) -> None:
        """Test image tensor has correct shape [C, X, Y, Z]."""
        sample = ml_dataset[0]
        image = sample["image"]
        assert image.shape == (2, 240, 240, 155)

    def test_image_dtype(self, ml_dataset: RadiObjectDataset) -> None:
        """Test image tensor has float dtype."""
        sample = ml_dataset[0]
        assert sample["image"].dtype == torch.float32

    def test_modalities_property(self, ml_dataset: RadiObjectDataset) -> None:
        """Test modalities property returns expected list."""
        assert ml_dataset.modalities == ["flair", "T1w"]

    def test_volume_shape_property(self, ml_dataset: RadiObjectDataset) -> None:
        """Test volume_shape property."""
        assert ml_dataset.volume_shape == (240, 240, 155)


class TestPatchDataset:
    """Tests for patch extraction mode."""

    def test_patch_dataset_length(self, ml_dataset_patch: RadiObjectDataset) -> None:
        """Test patch dataset length accounts for patches_per_volume."""
        assert len(ml_dataset_patch) == 3 * 2

    def test_patch_shape(self, ml_dataset_patch: RadiObjectDataset) -> None:
        """Test extracted patch has correct shape."""
        sample = ml_dataset_patch[0]
        assert sample["image"].shape == (1, 64, 64, 64)

    def test_patch_includes_metadata(self, ml_dataset_patch: RadiObjectDataset) -> None:
        """Test patch sample includes position metadata."""
        sample = ml_dataset_patch[0]
        assert "patch_idx" in sample
        assert "patch_start" in sample

    def test_patch_within_bounds(self, ml_dataset_patch: RadiObjectDataset) -> None:
        """Test patch start position is within volume bounds."""
        sample = ml_dataset_patch[0]
        start = sample["patch_start"]
        vol_shape = ml_dataset_patch.volume_shape
        patch_size = (64, 64, 64)
        for i in range(3):
            assert start[i] >= 0
            assert start[i] + patch_size[i] <= vol_shape[i]


class TestCachedDataset:
    """Tests for caching behavior."""

    def test_cache_hits_increase(self, ml_dataset_cached: RadiObjectDataset) -> None:
        """Test cache records hits on repeated access."""
        _ = ml_dataset_cached[0]
        _ = ml_dataset_cached[0]
        assert ml_dataset_cached.cache.hits >= 1

    def test_cache_is_in_memory(self, ml_dataset_cached: RadiObjectDataset) -> None:
        """Test correct cache type is used."""
        assert isinstance(ml_dataset_cached.cache, InMemoryCache)


class TestSliceDataset:
    """Tests for 2D slice extraction mode."""

    def test_slice_dataset_creation(self, populated_radi_object_module) -> None:
        """Test 2D slice dataset can be created."""
        config = DatasetConfig(
            loading_mode=LoadingMode.SLICE_2D,
            modalities=["flair"],
        )
        dataset = RadiObjectDataset(populated_radi_object_module, config)
        assert len(dataset) == 3 * 155

    def test_slice_shape(self, populated_radi_object_module) -> None:
        """Test 2D slice has correct shape."""
        config = DatasetConfig(
            loading_mode=LoadingMode.SLICE_2D,
            modalities=["flair"],
        )
        dataset = RadiObjectDataset(populated_radi_object_module, config)
        sample = dataset[0]
        assert sample["image"].shape == (1, 240, 240)


class TestDatasetParameterized:
    """Tests parameterized by storage backend."""

    def test_dataset_works_with_backend(self, ml_dataset_param: RadiObjectDataset) -> None:
        """Test dataset works with both local and S3 backends."""
        assert len(ml_dataset_param) == 3
        sample = ml_dataset_param[0]
        assert "image" in sample
        assert sample["image"].shape[0] == 2


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_patch_requires_size(self) -> None:
        """Test PATCH mode requires patch_size."""
        with pytest.raises(ValueError, match="patch_size required"):
            DatasetConfig(loading_mode=LoadingMode.PATCH)

    def test_patches_per_volume_positive(self) -> None:
        """Test patches_per_volume must be positive."""
        with pytest.raises(ValueError, match="patches_per_volume must be >= 1"):
            DatasetConfig(
                loading_mode=LoadingMode.PATCH,
                patch_size=(64, 64, 64),
                patches_per_volume=0,
            )
