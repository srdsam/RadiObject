"""Tests for cross-collection alignment and shape-changing transforms (resampling)."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import zoom

from radiobject.volume_collection import VolumeCollection


def _sphere_mask(
    shape: tuple[int, int, int], center: tuple[int, int, int], radius: int
) -> np.ndarray:
    """Binary mask for a sphere at center with given radius."""
    grid = np.mgrid[: shape[0], : shape[1], : shape[2]]
    dist = np.sqrt(sum((g - c) ** 2 for g, c in zip(grid, center)))
    return (dist <= radius).astype(np.float32)


@pytest.fixture
def ct_seg_nifti_dirs(temp_dir: Path) -> tuple[Path, Path, tuple[float, float, float]]:
    """Create 3 synthetic CT+seg NIfTI pairs with anisotropic spacing.

    CT: tissue sphere (HU=40) in air background (HU=-1024), shape (40,40,20).
    Seg: smaller tumor sphere (label=1) at same center.
    Spacing: 0.78 x 0.78 x 2.5 mm (anisotropic, like real MSD Lung).
    """
    ct_dir = temp_dir / "imagesTr"
    seg_dir = temp_dir / "labelsTr"
    ct_dir.mkdir()
    seg_dir.mkdir()

    shape = (40, 40, 20)
    spacing = (0.78, 0.78, 2.5)
    center = (20, 20, 10)

    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])

    for i in range(3):
        # CT: air background + tissue sphere
        ct_data = np.full(shape, -1024.0, dtype=np.float32)
        tissue_mask = _sphere_mask(shape, center, radius=12)
        ct_data[tissue_mask > 0] = 40.0

        # Seg: tumor sphere (subset of tissue)
        seg_data = _sphere_mask(shape, center, radius=5)

        for data, out_dir, prefix in [
            (ct_data, ct_dir, "lung"),
            (seg_data, seg_dir, "lung"),
        ]:
            img = nib.Nifti1Image(data, affine)
            img.header.set_qform(affine, code=1)
            img.header.set_sform(affine, code=1)
            nib.save(img, out_dir / f"{prefix}_{i:03d}.nii.gz")

    return ct_dir, seg_dir, spacing


class TestResample:
    """Tests for resampling via map() + write()."""

    def test_same_zoom_produces_matching_shapes(
        self, temp_dir: Path, ct_seg_nifti_dirs: tuple[Path, Path, tuple[float, float, float]]
    ):
        """map(zoom) on independent CT/seg collections produces identical output shapes."""
        ct_dir, seg_dir, _ = ct_seg_nifti_dirs

        ct_niftis = [(p, f"sub-{i:02d}") for i, p in enumerate(sorted(ct_dir.glob("*.nii.gz")))]
        seg_niftis = [(p, f"sub-{i:02d}") for i, p in enumerate(sorted(seg_dir.glob("*.nii.gz")))]

        ct_vc = VolumeCollection.from_niftis(
            str(temp_dir / "ct_raw"), ct_niftis, validate_dimensions=False
        )
        seg_vc = VolumeCollection.from_niftis(
            str(temp_dir / "seg_raw"), seg_niftis, validate_dimensions=False
        )

        ct_resampled = ct_vc.map(lambda v, obs: zoom(v, 0.5, order=1)).write(
            str(temp_dir / "ct_out")
        )
        seg_resampled = seg_vc.map(lambda v, obs: zoom(v, 0.5, order=0)).write(
            str(temp_dir / "seg_out")
        )

        for ct_vol, seg_vol in zip(ct_resampled, seg_resampled):
            assert ct_vol.shape == seg_vol.shape

    def test_tumor_overlaps_tissue_after_zoom(
        self, temp_dir: Path, ct_seg_nifti_dirs: tuple[Path, Path, tuple[float, float, float]]
    ):
        """After resampling, tumor voxels in seg correspond to tissue (not air) in CT."""
        ct_dir, seg_dir, _ = ct_seg_nifti_dirs

        ct_niftis = [(p, f"sub-{i:02d}") for i, p in enumerate(sorted(ct_dir.glob("*.nii.gz")))]
        seg_niftis = [(p, f"sub-{i:02d}") for i, p in enumerate(sorted(seg_dir.glob("*.nii.gz")))]

        ct_vc = VolumeCollection.from_niftis(
            str(temp_dir / "ct_raw"), ct_niftis, validate_dimensions=False
        )
        seg_vc = VolumeCollection.from_niftis(
            str(temp_dir / "seg_raw"), seg_niftis, validate_dimensions=False
        )

        ct_resampled = ct_vc.map(lambda v, obs: zoom(v, 0.5, order=1)).write(
            str(temp_dir / "ct_out")
        )
        seg_resampled = seg_vc.map(lambda v, obs: zoom(v, 0.5, order=0)).write(
            str(temp_dir / "seg_out")
        )

        ct_vol = ct_resampled.iloc[0]
        seg_vol = seg_resampled.iloc[0]

        ct_data = ct_vol.to_numpy()
        seg_data = seg_vol.to_numpy()
        tumor_mask = seg_data > 0

        assert tumor_mask.any(), "No tumor voxels found in resampled seg"
        ct_at_tumor = ct_data[tumor_mask]
        assert (
            ct_at_tumor.mean() > -900
        ), f"CT at tumor is air (mean={ct_at_tumor.mean():.0f} HU) -- misaligned"

    def test_zoom_half_halves_dimensions(self, temp_dir: Path):
        """zoom(0.5) on (32,32,16) produces (16,16,8)."""
        shape = (32, 32, 16)
        affine = np.eye(4)
        data = np.random.rand(*shape).astype(np.float32)
        img = nib.Nifti1Image(data, affine)
        path = temp_dir / "test.nii.gz"
        nib.save(img, path)

        vc = VolumeCollection.from_niftis(
            str(temp_dir / "vc"), [(path, "sub-01")], validate_dimensions=False
        )
        resampled = vc.map(lambda v, obs: zoom(v, 0.5, order=1)).write(str(temp_dir / "vc_out"))

        vol = resampled.iloc[0]
        assert vol.shape == (16, 16, 8)

    def test_anisotropic_zoom_factors(
        self, temp_dir: Path, ct_seg_nifti_dirs: tuple[Path, Path, tuple[float, float, float]]
    ):
        """Per-axis zoom factors produce correct dimensions for anisotropic spacing."""
        ct_dir, _, spacing = ct_seg_nifti_dirs
        target_spacing = 2.0
        zoom_factors = tuple(s / target_spacing for s in spacing)

        ct_niftis = [(p, f"sub-{i:02d}") for i, p in enumerate(sorted(ct_dir.glob("*.nii.gz")))]
        ct_vc = VolumeCollection.from_niftis(
            str(temp_dir / "ct_raw"), ct_niftis, validate_dimensions=False
        )

        resampled = ct_vc.map(lambda v, obs: zoom(v, zoom_factors, order=1)).write(
            str(temp_dir / "ct_out")
        )

        vol = resampled.iloc[0]
        raw_shape = (40, 40, 20)
        expected_shape = tuple(round(d * z) for d, z in zip(raw_shape, zoom_factors))
        assert vol.shape == expected_shape, f"Expected {expected_shape}, got {vol.shape}"

    def test_seg_labels_overlap_ct_tissue(
        self, temp_dir: Path, ct_seg_nifti_dirs: tuple[Path, Path, tuple[float, float, float]]
    ):
        """Raw ingested data has CT tissue at seg label locations (pre-resample sanity check)."""
        ct_dir, seg_dir, _ = ct_seg_nifti_dirs

        ct_niftis = [(p, f"sub-{i:02d}") for i, p in enumerate(sorted(ct_dir.glob("*.nii.gz")))]
        seg_niftis = [(p, f"sub-{i:02d}") for i, p in enumerate(sorted(seg_dir.glob("*.nii.gz")))]

        ct_vc = VolumeCollection.from_niftis(
            str(temp_dir / "ct_raw"), ct_niftis, validate_dimensions=False
        )
        seg_vc = VolumeCollection.from_niftis(
            str(temp_dir / "seg_raw"), seg_niftis, validate_dimensions=False
        )

        ct_data = ct_vc.iloc[0].to_numpy()
        seg_data = seg_vc.iloc[0].to_numpy()

        assert ct_data.shape == seg_data.shape
        tumor_mask = seg_data > 0
        assert tumor_mask.any()
        ct_at_tumor = ct_data[tumor_mask]
        assert ct_at_tumor.mean() > -900, f"CT at tumor is air (mean={ct_at_tumor.mean():.0f} HU)"
