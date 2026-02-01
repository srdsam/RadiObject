"""Pytest fixtures for ML tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from radiobject.ml.config import DatasetConfig, LoadingMode
from radiobject.ml.datasets import VolumeCollectionDataset

if TYPE_CHECKING:
    from radiobject.radi_object import RadiObject


@pytest.fixture(scope="module")
def ml_dataset(populated_radi_object_module: "RadiObject") -> VolumeCollectionDataset:
    """VolumeCollectionDataset backed by real BraTS volumes."""
    config = DatasetConfig(loading_mode=LoadingMode.FULL_VOLUME)
    # Use first two collections for multi-modal testing
    collections = [
        populated_radi_object_module.collection("flair"),
        populated_radi_object_module.collection("T1w"),
    ]
    return VolumeCollectionDataset(collections, config=config)


@pytest.fixture(scope="module")
def ml_dataset_patch(populated_radi_object_module: "RadiObject") -> VolumeCollectionDataset:
    """VolumeCollectionDataset configured for patch extraction."""
    config = DatasetConfig(
        loading_mode=LoadingMode.PATCH,
        patch_size=(64, 64, 64),
        patches_per_volume=2,
    )
    collection = populated_radi_object_module.collection("flair")
    return VolumeCollectionDataset(collection, config=config)


@pytest.fixture
def ml_dataset_param(s3_populated_radi_object: "RadiObject") -> VolumeCollectionDataset:
    """VolumeCollectionDataset on S3 backend.

    Note: This fixture only runs on S3 since local tests are covered by
    ml_dataset fixture. If S3 is unavailable, the test is skipped.
    """
    config = DatasetConfig(loading_mode=LoadingMode.FULL_VOLUME)
    collections = [
        s3_populated_radi_object.collection("flair"),
        s3_populated_radi_object.collection("T1w"),
    ]
    return VolumeCollectionDataset(collections, config=config)
