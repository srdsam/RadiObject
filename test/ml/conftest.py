"""Pytest fixtures for ML tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from radiobject.ml.config import CacheStrategy, DatasetConfig, LoadingMode
from radiobject.ml.datasets.volume_dataset import RadiObjectDataset

if TYPE_CHECKING:
    from radiobject.radi_object import RadiObject


@pytest.fixture(scope="module")
def ml_dataset(populated_radi_object_module: "RadiObject") -> RadiObjectDataset:
    """RadiObjectDataset backed by real BraTS volumes."""
    config = DatasetConfig(
        loading_mode=LoadingMode.FULL_VOLUME,
        modalities=["flair", "T1w"],
        label_column=None,
    )
    return RadiObjectDataset(populated_radi_object_module, config)


@pytest.fixture(scope="module")
def ml_dataset_patch(populated_radi_object_module: "RadiObject") -> RadiObjectDataset:
    """RadiObjectDataset configured for patch extraction."""
    config = DatasetConfig(
        loading_mode=LoadingMode.PATCH,
        patch_size=(64, 64, 64),
        patches_per_volume=2,
        modalities=["flair"],
    )
    return RadiObjectDataset(populated_radi_object_module, config)


@pytest.fixture(scope="module")
def ml_dataset_cached(populated_radi_object_module: "RadiObject") -> RadiObjectDataset:
    """RadiObjectDataset with in-memory caching."""
    config = DatasetConfig(
        loading_mode=LoadingMode.FULL_VOLUME,
        cache_strategy=CacheStrategy.IN_MEMORY,
        modalities=["flair"],
    )
    return RadiObjectDataset(populated_radi_object_module, config)


@pytest.fixture
def ml_dataset_param(s3_populated_radi_object: "RadiObject") -> RadiObjectDataset:
    """RadiObjectDataset on S3 backend.

    Note: This fixture only runs on S3 since local tests are covered by
    ml_dataset fixture. If S3 is unavailable, the test is skipped.
    """
    config = DatasetConfig(
        loading_mode=LoadingMode.FULL_VOLUME,
        modalities=["flair", "T1w"],
    )
    return RadiObjectDataset(s3_populated_radi_object, config)
