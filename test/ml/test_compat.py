"""Tests for MONAI/TorchIO compatibility."""

from __future__ import annotations

import pytest
import torch

from radiobject.ml.compat import Compose

try:
    import torchio as tio

    HAS_TORCHIO = True
except ImportError:
    HAS_TORCHIO = False

try:
    import monai.transforms  # noqa: F401

    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False


def test_fallback_compose():
    """Compose works without MONAI/TorchIO."""

    def add_one(data: dict) -> dict:
        data["value"] = data["value"] + 1
        return data

    transform = Compose([add_one, add_one])
    result = transform({"value": 0})
    assert result["value"] == 2


def test_compose_repr():
    """Compose has a useful repr."""

    class MyTransform:
        def __call__(self, data):
            return data

    transform = Compose([MyTransform(), MyTransform()])
    assert "Compose" in repr(transform)


@pytest.mark.skipif(not HAS_TORCHIO, reason="TorchIO not installed")
def test_subjects_dataset(populated_radi_object_module):
    """VolumeCollectionSubjectsDataset returns tio.Subject."""
    from radiobject.ml.compat import VolumeCollectionSubjectsDataset

    collection = populated_radi_object_module.collection("flair")
    dataset = VolumeCollectionSubjectsDataset(collection)

    assert len(dataset) > 0
    subject = dataset[0]
    assert isinstance(subject, tio.Subject)
    assert "flair" in subject


@pytest.mark.skipif(not HAS_TORCHIO, reason="TorchIO not installed")
def test_subjects_dataset_multimodal(populated_radi_object_module):
    """VolumeCollectionSubjectsDataset handles multiple collections."""
    from radiobject.ml.compat import VolumeCollectionSubjectsDataset

    collections = [
        populated_radi_object_module.collection("flair"),
        populated_radi_object_module.collection("T1w"),
    ]
    dataset = VolumeCollectionSubjectsDataset(collections)

    subject = dataset[0]
    assert "flair" in subject
    assert "T1w" in subject


@pytest.mark.skipif(not HAS_TORCHIO, reason="TorchIO not installed")
def test_subjects_dataset_with_transform(populated_radi_object_module):
    """VolumeCollectionSubjectsDataset applies TorchIO transforms."""
    from radiobject.ml.compat import VolumeCollectionSubjectsDataset

    collection = populated_radi_object_module.collection("flair")
    transform = tio.ZNormalization()
    dataset = VolumeCollectionSubjectsDataset(collection, transform=transform)

    subject = dataset[0]
    image_data = subject["flair"].data
    assert torch.abs(image_data.mean()) < 1.0


@pytest.mark.skipif(not HAS_MONAI, reason="MONAI not installed")
def test_monai_transforms_work_directly(populated_radi_object_module):
    """MONAI dict transforms work with VolumeCollectionDataset output."""
    from monai.transforms import RandFlipd

    from radiobject.ml.config import DatasetConfig, LoadingMode
    from radiobject.ml.datasets import VolumeCollectionDataset

    config = DatasetConfig(loading_mode=LoadingMode.FULL_VOLUME)
    collection = populated_radi_object_module.collection("flair")
    dataset = VolumeCollectionDataset(collection, config=config)

    transform = RandFlipd(keys="image", prob=1.0)
    sample = dataset[0]
    original_shape = sample["image"].shape

    transformed = transform(sample)

    assert "image" in transformed
    assert transformed["image"].shape == original_shape


@pytest.mark.skipif(not HAS_MONAI, reason="MONAI not installed")
def test_monai_normalize_intensity(populated_radi_object_module):
    """MONAI NormalizeIntensityd works with VolumeCollectionDataset."""
    from monai.transforms import NormalizeIntensityd

    from radiobject.ml.config import DatasetConfig, LoadingMode
    from radiobject.ml.datasets import VolumeCollectionDataset

    config = DatasetConfig(loading_mode=LoadingMode.FULL_VOLUME)
    collection = populated_radi_object_module.collection("flair")
    dataset = VolumeCollectionDataset(collection, config=config)

    transform = NormalizeIntensityd(keys="image", channel_wise=True)
    sample = dataset[0]
    transformed = transform(sample)

    assert torch.abs(transformed["image"].float().mean()) < 1.0
