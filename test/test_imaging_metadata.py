"""Tests for radiobject.imaging_metadata module.

Note: infer_series_type, extract_nifti_metadata, and NiftiMetadata are covered
in test_from_images.py. This file tests DicomMetadata and spatial unit parsing.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from radiobject.imaging_metadata import (
    DicomMetadata,
    NiftiMetadata,
    _get_spatial_units,
    extract_dicom_metadata,
    extract_nifti_metadata,
)


class TestSpatialUnitParsing:
    """Tests for _get_spatial_units internal function."""

    def test_millimeters(self):
        """xyzt_units=2 indicates millimeters."""
        assert _get_spatial_units(2) == "mm"

    def test_meters(self):
        """xyzt_units=1 indicates meters."""
        assert _get_spatial_units(1) == "m"

    def test_micrometers(self):
        """xyzt_units=3 indicates micrometers."""
        assert _get_spatial_units(3) == "um"

    def test_unknown(self):
        """xyzt_units=0 indicates unknown."""
        assert _get_spatial_units(0) == "unknown"

    def test_high_bits_masked(self):
        """Temporal bits (high nibble) are masked out."""
        # 0x12 = temporal_code=1, spatial_code=2 (mm)
        assert _get_spatial_units(0x12) == "mm"


class TestNiftiMetadataModel:
    """Tests for NiftiMetadata Pydantic model."""

    def test_to_obs_dict_serializes_tuples(self):
        """to_obs_dict converts tuples to strings for TileDB storage."""
        meta = NiftiMetadata(
            voxel_spacing=(1.0, 1.0, 2.0),
            dimensions=(64, 64, 32),
            datatype=16,
            bitpix=32,
            scl_slope=1.0,
            scl_inter=0.0,
            xyzt_units=2,
            spatial_units="mm",
            qform_code=1,
            sform_code=1,
            axcodes="RAS",
            affine_json="[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]",
            orientation_source="nifti_sform",
            source_path="/path/to/file.nii.gz",
        )

        obs_dict = meta.to_obs_dict("vol_001", "sub-01", "T1w")

        assert obs_dict["obs_id"] == "vol_001"
        assert obs_dict["obs_subject_id"] == "sub-01"
        assert obs_dict["series_type"] == "T1w"
        assert obs_dict["voxel_spacing"] == "(1.0, 1.0, 2.0)"
        assert obs_dict["dimensions"] == "(64, 64, 32)"

    def test_model_is_frozen(self):
        """NiftiMetadata is immutable after creation."""
        meta = NiftiMetadata(
            voxel_spacing=(1.0, 1.0, 1.0),
            dimensions=(32, 32, 16),
            datatype=16,
            bitpix=32,
            scl_slope=1.0,
            scl_inter=0.0,
            xyzt_units=2,
            spatial_units="mm",
            qform_code=1,
            sform_code=1,
            axcodes="RAS",
            affine_json="[]",
            orientation_source="nifti_sform",
            source_path="/test.nii",
        )

        with pytest.raises(Exception):
            meta.voxel_spacing = (2.0, 2.0, 2.0)


class TestDicomMetadataModel:
    """Tests for DicomMetadata Pydantic model."""

    def test_to_obs_dict_serializes_tuples(self):
        """to_obs_dict converts tuples to strings for TileDB storage."""
        meta = DicomMetadata(
            voxel_spacing=(0.5, 0.5, 1.0),
            dimensions=(512, 512, 150),
            modality="CT",
            series_description="Chest CT",
            kvp=120.0,
            exposure=250.0,
            repetition_time=None,
            echo_time=None,
            magnetic_field_strength=None,
            axcodes="RAS",
            affine_json="[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]",
            orientation_source="dicom_iop",
            source_path="/path/to/dicom",
        )

        obs_dict = meta.to_obs_dict("scan_001", "patient-01")

        assert obs_dict["obs_id"] == "scan_001"
        assert obs_dict["obs_subject_id"] == "patient-01"
        assert obs_dict["voxel_spacing"] == "(0.5, 0.5, 1.0)"
        assert obs_dict["dimensions"] == "(512, 512, 150)"
        assert obs_dict["modality"] == "CT"

    def test_optional_acquisition_params(self):
        """DICOM metadata allows optional acquisition parameters."""
        meta = DicomMetadata(
            voxel_spacing=(1.0, 1.0, 3.0),
            dimensions=(256, 256, 50),
            modality="MR",
            series_description="Brain T1",
            kvp=None,
            exposure=None,
            repetition_time=2000.0,
            echo_time=25.0,
            magnetic_field_strength=3.0,
            axcodes="LAS",
            affine_json="[]",
            orientation_source="dicom_iop",
            source_path="/test",
        )

        assert meta.kvp is None
        assert meta.repetition_time == 2000.0
        assert meta.magnetic_field_strength == 3.0


class TestExtractDicomMetadata:
    """Tests for extract_dicom_metadata function."""

    def test_directory_not_found_raises(self, temp_dir: Path):
        """Non-existent directory raises FileNotFoundError."""
        nonexistent = temp_dir / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            extract_dicom_metadata(nonexistent)

    def test_empty_directory_raises(self, temp_dir: Path):
        """Empty directory raises ValueError."""
        empty_dir = temp_dir / "empty_dicom"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No DICOM files"):
            extract_dicom_metadata(empty_dir)

    def test_with_real_dicom(self, sample_dicom_series: Path):
        """Extract metadata from real DICOM series (NSCLC Radiomics)."""
        meta = extract_dicom_metadata(sample_dicom_series)

        assert meta.modality == "CT"
        assert len(meta.voxel_spacing) == 3
        assert len(meta.dimensions) == 3
        assert meta.dimensions[2] > 1  # Multiple slices
        assert len(meta.axcodes) == 3


class TestNiftiMetadata4D:
    """Tests for 4D NIfTI metadata handling."""

    def test_extract_nifti_metadata_4d(self, temp_dir: Path):
        """4D NIfTI metadata captures time dimension."""
        shape_4d = (64, 64, 32, 200)
        affine = np.eye(4)
        data = np.zeros(shape_4d, dtype=np.float32)
        img = nib.Nifti1Image(data, affine)
        img.header.set_sform(affine, code=1)
        path = temp_dir / "sub-01_bold.nii.gz"
        nib.save(img, path)

        meta = extract_nifti_metadata(path)

        assert meta.dimensions == (64, 64, 32, 200)
        assert len(meta.dimensions) == 4

    def test_extract_nifti_metadata_3d_no_time(self, temp_dir: Path):
        """3D NIfTI stays 3D (no spurious time dimension)."""
        shape_3d = (64, 64, 32)
        affine = np.eye(4)
        data = np.zeros(shape_3d, dtype=np.float32)
        img = nib.Nifti1Image(data, affine)
        img.header.set_sform(affine, code=1)
        path = temp_dir / "sub-01_T1w.nii.gz"
        nib.save(img, path)

        meta = extract_nifti_metadata(path)

        assert meta.dimensions == (64, 64, 32)
        assert len(meta.dimensions) == 3

    def test_spatial_dimensions_property(self):
        """spatial_dimensions returns first 3 dims for both 3D and 4D."""
        base = dict(
            voxel_spacing=(1.0, 1.0, 1.0),
            datatype=16,
            bitpix=32,
            scl_slope=1.0,
            scl_inter=0.0,
            xyzt_units=2,
            spatial_units="mm",
            qform_code=1,
            sform_code=1,
            axcodes="RAS",
            affine_json="[]",
            orientation_source="nifti_sform",
            source_path="/test.nii",
        )
        meta_3d = NiftiMetadata(dimensions=(64, 64, 32), **base)
        meta_4d = NiftiMetadata(dimensions=(64, 64, 32, 200), **base)

        assert meta_3d.spatial_dimensions == (64, 64, 32)
        assert meta_4d.spatial_dimensions == (64, 64, 32)

    def test_is_4d_property(self):
        """is_4d correctly distinguishes 3D from 4D."""
        base = dict(
            voxel_spacing=(1.0, 1.0, 1.0),
            datatype=16,
            bitpix=32,
            scl_slope=1.0,
            scl_inter=0.0,
            xyzt_units=2,
            spatial_units="mm",
            qform_code=1,
            sform_code=1,
            axcodes="RAS",
            affine_json="[]",
            orientation_source="nifti_sform",
            source_path="/test.nii",
        )
        meta_3d = NiftiMetadata(dimensions=(64, 64, 32), **base)
        meta_4d = NiftiMetadata(dimensions=(64, 64, 32, 200), **base)

        assert meta_3d.is_4d is False
        assert meta_4d.is_4d is True

    def test_to_obs_dict_4d_dimensions(self):
        """4D dimensions serialize correctly in to_obs_dict."""
        meta = NiftiMetadata(
            voxel_spacing=(1.0, 1.0, 2.0),
            dimensions=(64, 64, 32, 200),
            datatype=16,
            bitpix=32,
            scl_slope=1.0,
            scl_inter=0.0,
            xyzt_units=2,
            spatial_units="mm",
            qform_code=1,
            sform_code=1,
            axcodes="RAS",
            affine_json="[]",
            orientation_source="nifti_sform",
            source_path="/test.nii",
        )

        obs_dict = meta.to_obs_dict("vol_001", "sub-01", "bold")

        assert obs_dict["dimensions"] == "(64, 64, 32, 200)"
