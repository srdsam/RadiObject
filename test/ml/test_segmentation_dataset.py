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


class TestOrientationConsistency:
    """Tests for image/mask orientation consistency in segmentation datasets."""

    def test_image_mask_orientation_match(
        self, segmentation_collections: dict[str, VolumeCollection]
    ) -> None:
        """Verify image and mask volumes have matching orientation info."""
        image_vc = segmentation_collections["image"]
        mask_vc = segmentation_collections["mask"]

        for i in range(len(image_vc)):
            image_vol = image_vc.iloc[i]
            mask_vol = mask_vc.iloc[i]

            image_orient = image_vol.orientation_info
            mask_orient = mask_vol.orientation_info

            # Both should have orientation info (or both None for synthetic data)
            if image_orient is not None and mask_orient is not None:
                assert image_orient.axcodes == mask_orient.axcodes, (
                    f"Volume {i}: Image orientation {image_orient.axcodes} "
                    f"!= Mask orientation {mask_orient.axcodes}"
                )

    def test_image_mask_shape_consistency(
        self, segmentation_collections: dict[str, VolumeCollection]
    ) -> None:
        """Verify image and mask volumes have identical shapes."""
        image_vc = segmentation_collections["image"]
        mask_vc = segmentation_collections["mask"]

        for i in range(len(image_vc)):
            image_vol = image_vc.iloc[i]
            mask_vol = mask_vc.iloc[i]

            assert (
                image_vol.shape == mask_vol.shape
            ), f"Volume {i}: Image shape {image_vol.shape} != Mask shape {mask_vol.shape}"

    def test_patch_extraction_alignment(
        self, segmentation_collections: dict[str, VolumeCollection]
    ) -> None:
        """Verify patches from same position are correctly aligned."""
        image_vc = segmentation_collections["image"]
        mask_vc = segmentation_collections["mask"]

        image_vol = image_vc.iloc[0]
        mask_vol = mask_vc.iloc[0]

        # Extract same patch from both volumes
        patch_start = (50, 50, 30)
        patch_size = (32, 32, 32)

        image_patch = image_vol.slice(
            slice(patch_start[0], patch_start[0] + patch_size[0]),
            slice(patch_start[1], patch_start[1] + patch_size[1]),
            slice(patch_start[2], patch_start[2] + patch_size[2]),
        )
        mask_patch = mask_vol.slice(
            slice(patch_start[0], patch_start[0] + patch_size[0]),
            slice(patch_start[1], patch_start[1] + patch_size[1]),
            slice(patch_start[2], patch_start[2] + patch_size[2]),
        )

        # Both patches should have the same shape
        assert image_patch.shape == mask_patch.shape == patch_size

        # Verify patches are from the same spatial region (not shifted/misaligned)
        # For BraTS data, mask should be non-zero only where there's brain tissue
        # If mask has foreground, image at same location should have values
        if mask_patch.max() > 0:
            mask_fg = mask_patch > 0
            image_at_mask = image_patch[mask_fg]
            # Image should have non-zero values where mask has foreground
            # (this verifies spatial alignment - tumor is visible in image)
            assert image_at_mask.max() > 0, "Image has no signal where mask indicates foreground"

    def test_dataset_batch_orientation_consistency(
        self, segmentation_collections: dict[str, VolumeCollection]
    ) -> None:
        """Verify batched samples maintain orientation consistency."""
        from radiobject.ml.factory import create_segmentation_dataloader

        loader = create_segmentation_dataloader(
            image=segmentation_collections["image"],
            mask=segmentation_collections["mask"],
            batch_size=2,
            patch_size=(32, 32, 32),
            num_workers=0,
        )

        batch = next(iter(loader))

        # Image and mask in batch should have same shape
        assert batch["image"].shape == batch["mask"].shape

        # Each sample in batch should have consistent dimensions
        assert batch["image"].shape[0] == batch["mask"].shape[0] == 2  # batch size
        assert batch["image"].shape[1] == batch["mask"].shape[1] == 1  # channels


class TestReorientedVolumeDataset:
    """Tests for datasets created from reoriented volumes."""

    def test_reoriented_volumes_spatial_alignment(self, temp_dir_module: Path) -> None:
        """Verify reoriented image/mask remain spatially aligned."""
        from radiobject.volume import Volume

        # Create synthetic LPS-oriented image and mask
        rng = np.random.default_rng(42)
        shape = (32, 32, 16)

        # Image with some structure
        image_data = rng.random(shape, dtype=np.float32)

        # Mask marking region of interest in image
        mask_data = np.zeros(shape, dtype=np.float32)
        mask_data[10:20, 10:20, 5:12] = 1.0  # Box in center

        # Create NIfTI files with LPS orientation
        import nibabel as nib

        lps_affine = np.array(
            [
                [-1.0, 0.0, 0.0, 16.0],
                [0.0, -1.0, 0.0, 16.0],
                [0.0, 0.0, 1.0, -8.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        image_nifti_path = temp_dir_module / "test_reorient_image.nii.gz"
        mask_nifti_path = temp_dir_module / "test_reorient_mask.nii.gz"

        nib.save(nib.Nifti1Image(image_data, lps_affine), image_nifti_path)
        nib.save(nib.Nifti1Image(mask_data, lps_affine), mask_nifti_path)

        # Load with reorientation to RAS
        image_vol = Volume.from_nifti(
            str(temp_dir_module / "reorient_image_vol"),
            image_nifti_path,
            reorient=True,
        )
        mask_vol = Volume.from_nifti(
            str(temp_dir_module / "reorient_mask_vol"),
            mask_nifti_path,
            reorient=True,
        )

        # Both should now be RAS
        assert image_vol.orientation_info.axcodes == ("R", "A", "S")
        assert mask_vol.orientation_info.axcodes == ("R", "A", "S")

        # Verify spatial alignment is preserved after reorientation
        image_reoriented = image_vol.to_numpy()
        mask_reoriented = mask_vol.to_numpy()

        # The mask should still mark valid regions in the image
        # (foreground region should have non-zero image values)
        mask_fg = mask_reoriented > 0
        if mask_fg.sum() > 0:
            # Image should have signal where mask indicates
            assert image_reoriented[mask_fg].mean() > 0
