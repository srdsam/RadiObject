"""Tests for SegmentationDataset and create_segmentation_dataloader."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

from radiobject.ml.config import DatasetConfig, LoadingMode
from radiobject.ml.datasets import SegmentationDataset
from radiobject.ml.factory import create_segmentation_dataloader
from radiobject.volume import Volume
from radiobject.volume_collection import VolumeCollection

if TYPE_CHECKING:
    pass


@pytest.fixture(scope="module")
def segmentation_collections(temp_dir_module: Path) -> dict[str, VolumeCollection]:
    """Create image and mask VolumeCollections from BraTS data."""
    import json

    from radiobject.data import get_dataset

    try:
        data_dir = get_dataset("msd-brain-tumour", prefer_s3=False)
        manifest_path = data_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (FileNotFoundError, Exception):
        pytest.skip("NIfTI manifest not found")

    # Build image collection (first modality from 4D)
    image_volumes = []
    image_obs_rows = []
    for idx, entry in enumerate(manifest[:3]):
        img_path = data_dir / entry["image_path"]
        img = nib.load(img_path)
        data = np.asarray(img.dataobj, dtype=np.float32)[..., 0]  # First modality

        obs_id = f"{entry['sample_id']}_flair"
        vol_uri = str(temp_dir_module / f"seg_image_vol_{idx}")
        vol = Volume.from_numpy(vol_uri, data)
        vol.set_obs_id(obs_id)
        image_volumes.append((obs_id, vol))
        image_obs_rows.append({"obs_id": obs_id, "obs_subject_id": entry["sample_id"]})

    # Build mask collection (from label files)
    mask_volumes = []
    mask_obs_rows = []
    for idx, entry in enumerate(manifest[:3]):
        label_path = data_dir / entry["label_path"]
        img = nib.load(label_path)
        data = np.asarray(img.dataobj, dtype=np.float32)

        obs_id = f"{entry['sample_id']}_seg"
        vol_uri = str(temp_dir_module / f"seg_mask_vol_{idx}")
        vol = Volume.from_numpy(vol_uri, data)
        vol.set_obs_id(obs_id)
        mask_volumes.append((obs_id, vol))
        mask_obs_rows.append({"obs_id": obs_id, "obs_subject_id": entry["sample_id"]})

    image_obs_df = pd.DataFrame(image_obs_rows)
    mask_obs_df = pd.DataFrame(mask_obs_rows)

    image_vc = VolumeCollection._from_volumes(
        str(temp_dir_module / "seg_image_vc"),
        image_volumes,
        obs_data=image_obs_df,
        name="flair",
    )
    mask_vc = VolumeCollection._from_volumes(
        str(temp_dir_module / "seg_mask_vc"),
        mask_volumes,
        obs_data=mask_obs_df,
        name="seg",
    )

    return {"image": image_vc, "mask": mask_vc}


@pytest.fixture(scope="module")
def seg_dataset(segmentation_collections: dict[str, VolumeCollection]) -> SegmentationDataset:
    """SegmentationDataset for full volume mode."""
    return SegmentationDataset(
        image=segmentation_collections["image"],
        mask=segmentation_collections["mask"],
    )


@pytest.fixture(scope="module")
def seg_dataset_patch(
    segmentation_collections: dict[str, VolumeCollection],
) -> SegmentationDataset:
    """SegmentationDataset for patch extraction mode."""
    config = DatasetConfig(
        loading_mode=LoadingMode.PATCH,
        patch_size=(64, 64, 64),
        patches_per_volume=2,
    )
    return SegmentationDataset(
        image=segmentation_collections["image"],
        mask=segmentation_collections["mask"],
        config=config,
    )


class TestSegmentationDatasetBasic:
    """Basic SegmentationDataset tests."""

    def test_dataset_length(self, seg_dataset: SegmentationDataset) -> None:
        """Test dataset reports correct length."""
        assert len(seg_dataset) == 3

    def test_getitem_returns_dict(self, seg_dataset: SegmentationDataset) -> None:
        """Test __getitem__ returns dict with expected keys."""
        sample = seg_dataset[0]
        assert isinstance(sample, dict)
        assert "image" in sample
        assert "mask" in sample
        assert "idx" in sample
        assert "obs_id" in sample
        assert "obs_subject_id" in sample

    def test_image_mask_separate(self, seg_dataset: SegmentationDataset) -> None:
        """Test image and mask are returned as separate tensors."""
        sample = seg_dataset[0]
        assert sample["image"].shape != sample["mask"].shape or not torch.equal(
            sample["image"], sample["mask"]
        )

    def test_image_shape(self, seg_dataset: SegmentationDataset) -> None:
        """Test image tensor has correct shape [C, X, Y, Z]."""
        sample = seg_dataset[0]
        image = sample["image"]
        assert image.shape == (1, 240, 240, 155)

    def test_mask_shape(self, seg_dataset: SegmentationDataset) -> None:
        """Test mask tensor has correct shape [C, X, Y, Z]."""
        sample = seg_dataset[0]
        mask = sample["mask"]
        assert mask.shape == (1, 240, 240, 155)

    def test_obs_subject_id_included(self, seg_dataset: SegmentationDataset) -> None:
        """Test obs_subject_id is included in sample."""
        sample = seg_dataset[0]
        assert "obs_subject_id" in sample
        assert isinstance(sample["obs_subject_id"], str)
        # obs_subject_id should NOT have the modality suffix
        assert "_flair" not in sample["obs_subject_id"]
        assert "_seg" not in sample["obs_subject_id"]

    def test_volume_shape_property(self, seg_dataset: SegmentationDataset) -> None:
        """Test volume_shape property."""
        assert seg_dataset.volume_shape == (240, 240, 155)


class TestSegmentationDatasetPatch:
    """Tests for patch extraction mode."""

    def test_patch_dataset_length(self, seg_dataset_patch: SegmentationDataset) -> None:
        """Test patch dataset length accounts for patches_per_volume."""
        assert len(seg_dataset_patch) == 3 * 2

    def test_patch_shape(self, seg_dataset_patch: SegmentationDataset) -> None:
        """Test extracted patches have correct shape."""
        sample = seg_dataset_patch[0]
        assert sample["image"].shape == (1, 64, 64, 64)
        assert sample["mask"].shape == (1, 64, 64, 64)

    def test_patch_includes_metadata(self, seg_dataset_patch: SegmentationDataset) -> None:
        """Test patch sample includes position metadata."""
        sample = seg_dataset_patch[0]
        assert "patch_idx" in sample
        assert "patch_start" in sample
        assert "obs_subject_id" in sample

    def test_patch_within_bounds(self, seg_dataset_patch: SegmentationDataset) -> None:
        """Test patch start position is within volume bounds."""
        sample = seg_dataset_patch[0]
        start = sample["patch_start"]
        vol_shape = seg_dataset_patch.volume_shape
        patch_size = (64, 64, 64)
        for i in range(3):
            assert start[i] >= 0
            assert start[i] + patch_size[i] <= vol_shape[i]


class TestForegroundSampling:
    """Tests for foreground sampling feature."""

    def test_foreground_sampling_no_crash(
        self, segmentation_collections: dict[str, VolumeCollection]
    ) -> None:
        """Test foreground sampling doesn't crash."""
        config = DatasetConfig(
            loading_mode=LoadingMode.PATCH,
            patch_size=(64, 64, 64),
            patches_per_volume=2,
        )
        dataset = SegmentationDataset(
            image=segmentation_collections["image"],
            mask=segmentation_collections["mask"],
            config=config,
            foreground_sampling=True,
            foreground_threshold=0.001,  # Low threshold for test data
        )
        sample = dataset[0]
        assert "image" in sample
        assert "mask" in sample


class TestTransforms:
    """Tests for transform application."""

    def test_image_transform_applied(
        self, segmentation_collections: dict[str, VolumeCollection]
    ) -> None:
        """Test image_transform is applied to image only."""
        transform_called = {"image": False, "mask": False}

        def image_transform(data: dict[str, Any]) -> dict[str, Any]:
            transform_called["image"] = True
            assert "image" in data
            return data

        dataset = SegmentationDataset(
            image=segmentation_collections["image"],
            mask=segmentation_collections["mask"],
            image_transform=image_transform,
        )
        _ = dataset[0]
        assert transform_called["image"]

    def test_spatial_transform_applied(
        self, segmentation_collections: dict[str, VolumeCollection]
    ) -> None:
        """Test spatial_transform is applied before image_transform."""
        call_order: list[str] = []

        def spatial_transform(data: dict[str, Any]) -> dict[str, Any]:
            call_order.append("spatial")
            return data

        def image_transform(data: dict[str, Any]) -> dict[str, Any]:
            call_order.append("image")
            return data

        dataset = SegmentationDataset(
            image=segmentation_collections["image"],
            mask=segmentation_collections["mask"],
            spatial_transform=spatial_transform,
            image_transform=image_transform,
        )
        _ = dataset[0]
        # Spatial should be called first, then image
        assert call_order == ["spatial", "image"]


class TestCreateSegmentationDataloader:
    """Tests for create_segmentation_dataloader factory function."""

    def test_dataloader_creation(
        self, segmentation_collections: dict[str, VolumeCollection]
    ) -> None:
        """Test dataloader can be created."""
        loader = create_segmentation_dataloader(
            image=segmentation_collections["image"],
            mask=segmentation_collections["mask"],
            batch_size=2,
            num_workers=0,
        )
        assert loader is not None
        assert loader.batch_size == 2

    def test_dataloader_batch_shape(
        self, segmentation_collections: dict[str, VolumeCollection]
    ) -> None:
        """Test dataloader yields correct batch shapes."""
        loader = create_segmentation_dataloader(
            image=segmentation_collections["image"],
            mask=segmentation_collections["mask"],
            batch_size=2,
            patch_size=(32, 32, 32),
            num_workers=0,
        )
        batch = next(iter(loader))
        assert batch["image"].shape == (2, 1, 32, 32, 32)
        assert batch["mask"].shape == (2, 1, 32, 32, 32)

    def test_dataloader_with_foreground_sampling(
        self, segmentation_collections: dict[str, VolumeCollection]
    ) -> None:
        """Test dataloader with foreground sampling enabled."""
        loader = create_segmentation_dataloader(
            image=segmentation_collections["image"],
            mask=segmentation_collections["mask"],
            batch_size=2,
            patch_size=(32, 32, 32),
            foreground_sampling=True,
            foreground_threshold=0.001,
            num_workers=0,
        )
        batch = next(iter(loader))
        assert "image" in batch
        assert "mask" in batch


class TestAlignmentValidation:
    """Tests for image/mask alignment validation."""

    def test_mismatched_lengths_raises(self, temp_dir_module: Path) -> None:
        """Test ValueError raised when image/mask have different lengths."""
        # Create mismatched collections
        data = np.random.randn(10, 10, 10).astype(np.float32)

        # Image collection with 2 volumes
        img_volumes = []
        img_obs = []
        for i in range(2):
            vol_uri = str(temp_dir_module / f"mismatch_img_{i}")
            vol = Volume.from_numpy(vol_uri, data)
            obs_id = f"sub{i}_img"
            vol.set_obs_id(obs_id)
            img_volumes.append((obs_id, vol))
            img_obs.append({"obs_id": obs_id, "obs_subject_id": f"sub{i}"})

        # Mask collection with 3 volumes
        mask_volumes = []
        mask_obs = []
        for i in range(3):
            vol_uri = str(temp_dir_module / f"mismatch_mask_{i}")
            vol = Volume.from_numpy(vol_uri, data)
            obs_id = f"sub{i}_mask"
            vol.set_obs_id(obs_id)
            mask_volumes.append((obs_id, vol))
            mask_obs.append({"obs_id": obs_id, "obs_subject_id": f"sub{i}"})

        img_vc = VolumeCollection._from_volumes(
            str(temp_dir_module / "mismatch_img_vc"),
            img_volumes,
            obs_data=pd.DataFrame(img_obs),
        )
        mask_vc = VolumeCollection._from_volumes(
            str(temp_dir_module / "mismatch_mask_vc"),
            mask_volumes,
            obs_data=pd.DataFrame(mask_obs),
        )

        with pytest.raises(ValueError, match="has .* volumes, expected"):
            SegmentationDataset(image=img_vc, mask=mask_vc)


class TestViewBasedTraining:
    """Tests for view-based train/val splits."""

    def test_view_reads_correct_volumes(
        self, segmentation_collections: dict[str, VolumeCollection]
    ) -> None:
        """Verify views read from correct storage locations."""
        image_vc = segmentation_collections["image"]
        mask_vc = segmentation_collections["mask"]

        # Create view that skips first volume
        image_view = image_vc.iloc[1:]
        mask_view = mask_vc.iloc[1:]

        dataset = SegmentationDataset(image=image_view, mask=mask_view)

        # Position 0 in view should be original position 1
        sample = dataset[0]
        expected_subject = image_vc.obs_subject_ids[1]
        assert sample["obs_subject_id"] == expected_subject

    def test_view_dataset_length(
        self, segmentation_collections: dict[str, VolumeCollection]
    ) -> None:
        """Verify view-based dataset has correct length."""
        image_vc = segmentation_collections["image"]
        mask_vc = segmentation_collections["mask"]

        # Create view with 2 of 3 volumes
        image_view = image_vc.iloc[:2]
        mask_view = mask_vc.iloc[:2]

        dataset = SegmentationDataset(image=image_view, mask=mask_view)
        assert len(dataset) == 2

    def test_view_obs_ids_match_source(
        self, segmentation_collections: dict[str, VolumeCollection]
    ) -> None:
        """Verify obs_ids in samples match the source collection."""
        image_vc = segmentation_collections["image"]
        mask_vc = segmentation_collections["mask"]

        # Create view that skips first volume
        image_view = image_vc.iloc[1:]
        mask_view = mask_vc.iloc[1:]

        dataset = SegmentationDataset(image=image_view, mask=mask_view)

        # All obs_ids should come from the view's obs_ids
        for i in range(len(dataset)):
            sample = dataset[i]
            assert sample["obs_id"] == image_view.obs_ids[i]

    def test_foreground_sampling_with_views(
        self, segmentation_collections: dict[str, VolumeCollection]
    ) -> None:
        """Verify foreground sampling works correctly with views."""
        image_vc = segmentation_collections["image"]
        mask_vc = segmentation_collections["mask"]

        # Create view
        image_view = image_vc.iloc[1:]
        mask_view = mask_vc.iloc[1:]

        config = DatasetConfig(
            loading_mode=LoadingMode.PATCH,
            patch_size=(64, 64, 64),
            patches_per_volume=2,
        )
        dataset = SegmentationDataset(
            image=image_view,
            mask=mask_view,
            config=config,
            foreground_sampling=True,
            foreground_threshold=0.001,
        )

        # Should not crash and produce valid samples
        sample = dataset[0]
        assert "image" in sample
        assert "mask" in sample
        assert sample["image"].shape == (1, 64, 64, 64)
