"""Imaging metadata extraction for NIfTI and DICOM files."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
from pydantic import BaseModel, Field

from radiobject.utils import affine_to_json

# BIDS-aligned series type identifiers
KNOWN_SERIES_TYPES: frozenset[str] = frozenset(
    {
        # Anatomical MRI
        "T1w",
        "T2w",
        "T1rho",
        "T1map",
        "T2map",
        "T2star",
        "FLAIR",
        "FLASH",
        "PD",
        "PDmap",
        "PDT2",
        "inplaneT1",
        "inplaneT2",
        "angio",
        "T1gd",
        # Functional MRI
        "bold",
        "cbv",
        "phase",
        # Diffusion MRI
        "dwi",
        # CT variants
        "CT",
        "HRCT",
        "CTA",
        "CTPA",
        # Field maps
        "phasediff",
        "magnitude",
        "fieldmap",
        "epi",
    }
)

# Filename pattern to series type mapping (ordered from most specific to least specific)
_FILENAME_PATTERNS: tuple[tuple[str, str], ...] = (
    # Contrast-enhanced T1 (check before T1)
    ("T1GD", "T1gd"),
    ("T1CE", "T1gd"),
    ("T1C", "T1gd"),
    # Standard patterns
    ("T1W", "T1w"),
    ("T1", "T1w"),
    ("T2W", "T2w"),
    ("T2", "T2w"),
    ("FLAIR", "FLAIR"),
    ("DWI", "dwi"),
    ("DTI", "dwi"),
    ("BOLD", "bold"),
    ("FUNC", "bold"),
    # CT patterns (MSD datasets, common naming)
    ("LUNG_", "CT"),
    ("LIVER_", "CT"),
    ("COLON_", "CT"),
    ("PANCREAS_", "CT"),
    ("SPLEEN_", "CT"),
    ("HEPATIC", "CT"),
    ("_CT_", "CT"),
    ("_CT.", "CT"),
)

# Spatial unit mapping from NIfTI xyzt_units
_SPATIAL_UNIT_MAP: dict[int, str] = {
    0: "unknown",
    1: "m",
    2: "mm",
    3: "um",
}


def infer_series_type(path: Path, header: nib.Nifti1Header | None = None) -> str:
    """Infer series type from filename patterns and header.

    Priority:
    1. BIDS-style suffix: sub-01_ses-01_T1w.nii.gz -> "T1w"
    2. Common patterns: T1_MPRAGE.nii.gz -> "T1w"
    3. Header description field
    4. Fallback: "unknown"
    """
    filename = path.stem
    if filename.endswith(".nii"):
        filename = filename[:-4]

    # Check for BIDS suffix (last underscore-separated part)
    parts = filename.split("_")
    if parts:
        suffix = parts[-1]
        for known in KNOWN_SERIES_TYPES:
            if known.lower() == suffix.lower():
                return known

    # Check common patterns in full filename (ordered from most specific to least)
    filename_upper = filename.upper()
    for pattern, series_type in _FILENAME_PATTERNS:
        if pattern in filename_upper:
            return series_type

    # Check header description if available
    if header is not None:
        descrip_raw = header.get("descrip", b"")
        if isinstance(descrip_raw, bytes):
            descrip = descrip_raw.decode("utf-8", errors="ignore")
        else:
            descrip = str(descrip_raw)
        descrip_lower = descrip.lower()
        for known in KNOWN_SERIES_TYPES:
            if known.lower() in descrip_lower:
                return known

    return "unknown"


def _get_spatial_units(xyzt_units: int) -> str:
    """Extract spatial units from NIfTI xyzt_units field."""
    spatial_code = xyzt_units & 0x07  # Lower 3 bits
    return _SPATIAL_UNIT_MAP.get(spatial_code, "unknown")


class NiftiMetadata(BaseModel):
    """NIfTI header metadata for obs DataFrame."""

    # Voxel spacing (from pixdim[1:4]) as (x, y, z) tuple
    voxel_spacing: tuple[float, float, float]

    # Original dimensions as (x, y, z) or (x, y, z, t) tuple
    dimensions: tuple[int, ...]

    # Data type
    datatype: int
    bitpix: int

    # Scaling
    scl_slope: float
    scl_inter: float

    # Units
    xyzt_units: int
    spatial_units: str  # "mm", "um", "m", or "unknown"

    # Coordinate system codes
    qform_code: int
    sform_code: int

    # Orientation
    axcodes: str  # e.g., "RAS"
    affine_json: str  # 4x4 matrix as JSON
    orientation_source: Literal["nifti_sform", "nifti_qform", "identity"]

    # Provenance
    source_path: str

    model_config = {"frozen": True}

    @property
    def spatial_dimensions(self) -> tuple[int, int, int]:
        """Spatial dims (X, Y, Z) regardless of 3D/4D."""
        return (self.dimensions[0], self.dimensions[1], self.dimensions[2])

    @property
    def is_4d(self) -> bool:
        """Whether this volume has a temporal dimension."""
        return len(self.dimensions) == 4

    def to_obs_dict(self, obs_id: str, obs_subject_id: str, series_type: str) -> dict:
        """Convert to dictionary for obs DataFrame row."""
        data = self.model_dump()
        # Serialize tuples as strings for TileDB storage
        data["voxel_spacing"] = str(data["voxel_spacing"])
        data["dimensions"] = str(data["dimensions"])
        data.update(obs_id=obs_id, obs_subject_id=obs_subject_id, series_type=series_type)
        return data


class DicomMetadata(BaseModel):
    """DICOM header metadata for obs DataFrame."""

    # Voxel spacing as (x, y, z) tuple - z is slice_thickness
    voxel_spacing: tuple[float, float, float]

    # Dimensions as (rows, columns, n_slices) tuple
    dimensions: tuple[int, int, int]

    # Patient/Study info (anonymized identifiers only)
    modality: str  # CT, MR, PT, etc.
    series_description: str

    # Acquisition parameters (None if not applicable)
    kvp: float | None = Field(default=None)  # CT tube voltage
    exposure: float | None = Field(default=None)  # CT exposure (mAs)
    repetition_time: float | None = Field(default=None)  # MRI TR
    echo_time: float | None = Field(default=None)  # MRI TE
    magnetic_field_strength: float | None = Field(default=None)  # MRI field strength

    # Orientation
    axcodes: str
    affine_json: str
    orientation_source: Literal["dicom_iop", "identity"]

    # Provenance
    source_path: str

    model_config = {"frozen": True}

    def to_obs_dict(self, obs_id: str, obs_subject_id: str) -> dict:
        """Convert to dictionary for obs DataFrame row."""
        data = self.model_dump()
        # Serialize tuples as strings for TileDB storage
        data["voxel_spacing"] = str(data["voxel_spacing"])
        data["dimensions"] = str(data["dimensions"])
        data.update(obs_id=obs_id, obs_subject_id=obs_subject_id)
        return data


def extract_nifti_metadata(nifti_path: str | Path) -> NiftiMetadata:
    """Extract comprehensive metadata from NIfTI header."""
    path = Path(nifti_path)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    img = nib.load(path)
    header = img.header

    # Extract dimensions
    dim = header.get("dim")
    n_dims = int(dim[0]) if len(dim) > 0 else 0
    dim_x = int(dim[1]) if len(dim) > 1 else 0
    dim_y = int(dim[2]) if len(dim) > 2 else 0
    dim_z = int(dim[3]) if len(dim) > 3 else 0
    dim_t = int(dim[4]) if n_dims >= 4 and len(dim) > 4 and int(dim[4]) > 1 else None

    # Extract voxel spacing
    pixdim = header.get("pixdim")
    voxel_spacing_x = float(pixdim[1]) if len(pixdim) > 1 else 1.0
    voxel_spacing_y = float(pixdim[2]) if len(pixdim) > 2 else 1.0
    voxel_spacing_z = float(pixdim[3]) if len(pixdim) > 3 else 1.0

    # Data type info
    datatype = int(header.get("datatype", 0))
    bitpix = int(header.get("bitpix", 0))

    # Scaling
    scl_slope = float(header.get("scl_slope", 1.0))
    scl_inter = float(header.get("scl_inter", 0.0))
    # Handle NaN slope (nibabel returns nan for 0)
    if np.isnan(scl_slope):
        scl_slope = 1.0
    if np.isnan(scl_inter):
        scl_inter = 0.0

    # Units
    xyzt_units = int(header.get("xyzt_units", 0))
    spatial_units = _get_spatial_units(xyzt_units)

    # Coordinate system codes
    sform_code = int(header.get("sform_code", 0))
    qform_code = int(header.get("qform_code", 0))

    # Determine orientation source and get affine
    if sform_code > 0:
        affine = img.get_sform()
        orientation_source: Literal["nifti_sform", "nifti_qform", "identity"] = "nifti_sform"
    elif qform_code > 0:
        affine = img.get_qform()
        orientation_source = "nifti_qform"
    else:
        affine = img.affine
        orientation_source = "identity"

    # Get axis codes
    ornt = nib.orientations.io_orientation(affine)
    axcodes = "".join(nib.orientations.ornt2axcodes(ornt))

    return NiftiMetadata(
        voxel_spacing=(voxel_spacing_x, voxel_spacing_y, voxel_spacing_z),
        dimensions=(dim_x, dim_y, dim_z, dim_t) if dim_t is not None else (dim_x, dim_y, dim_z),
        datatype=datatype,
        bitpix=bitpix,
        scl_slope=scl_slope,
        scl_inter=scl_inter,
        xyzt_units=xyzt_units,
        spatial_units=spatial_units,
        qform_code=qform_code,
        sform_code=sform_code,
        axcodes=axcodes,
        affine_json=affine_to_json(affine),
        orientation_source=orientation_source,
        source_path=str(path.absolute()),
    )


def extract_dicom_metadata(dicom_dir: str | Path) -> DicomMetadata:
    """Extract comprehensive metadata from DICOM series."""
    import pydicom

    path = Path(dicom_dir)
    if not path.exists():
        raise FileNotFoundError(f"DICOM directory not found: {path}")

    # Find DICOM files
    dicom_files = sorted(path.glob("*.dcm"))
    if not dicom_files:
        dicom_files = sorted(
            f for f in path.iterdir() if f.is_file() and not f.name.startswith(".")
        )

    if not dicom_files:
        raise ValueError(f"No DICOM files found in {path}")

    # Read first DICOM for most metadata
    ds = pydicom.dcmread(dicom_files[0])

    # Pixel spacing
    pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
    pixel_spacing_x = float(pixel_spacing[0])
    pixel_spacing_y = float(pixel_spacing[1])
    slice_thickness = float(getattr(ds, "SliceThickness", 1.0))

    # Dimensions
    rows = int(getattr(ds, "Rows", 0))
    columns = int(getattr(ds, "Columns", 0))
    n_slices = len(dicom_files)

    # Modality and description
    modality = str(getattr(ds, "Modality", "unknown"))
    series_description = str(getattr(ds, "SeriesDescription", ""))

    # Acquisition parameters (modality-specific)
    kvp = float(ds.KVP) if hasattr(ds, "KVP") else None
    exposure = float(ds.Exposure) if hasattr(ds, "Exposure") else None
    repetition_time = float(ds.RepetitionTime) if hasattr(ds, "RepetitionTime") else None
    echo_time = float(ds.EchoTime) if hasattr(ds, "EchoTime") else None
    magnetic_field_strength = (
        float(ds.MagneticFieldStrength) if hasattr(ds, "MagneticFieldStrength") else None
    )

    # Orientation
    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = getattr(ds, "ImagePositionPatient", None)

    if iop is not None:
        # Build affine from DICOM tags
        row_cosines = np.array([float(iop[0]), float(iop[1]), float(iop[2])])
        col_cosines = np.array([float(iop[3]), float(iop[4]), float(iop[5])])
        slice_cosines = np.cross(row_cosines, col_cosines)

        voxel_spacing = [pixel_spacing_y, pixel_spacing_x, slice_thickness]

        affine = np.eye(4)
        affine[:3, 0] = row_cosines * voxel_spacing[0]
        affine[:3, 1] = col_cosines * voxel_spacing[1]
        affine[:3, 2] = slice_cosines * voxel_spacing[2]

        if ipp is not None:
            affine[:3, 3] = [float(ipp[0]), float(ipp[1]), float(ipp[2])]

        # Convert DICOM LPS to RAS
        lps_to_ras = np.diag([-1, -1, 1, 1])
        affine_ras = lps_to_ras @ affine

        ornt = nib.orientations.io_orientation(affine_ras)
        axcodes = "".join(nib.orientations.ornt2axcodes(ornt))
        orientation_source: Literal["dicom_iop", "identity"] = "dicom_iop"
    else:
        affine_ras = np.eye(4)
        axcodes = "RAS"
        orientation_source = "identity"

    return DicomMetadata(
        voxel_spacing=(pixel_spacing_x, pixel_spacing_y, slice_thickness),
        dimensions=(rows, columns, n_slices),
        modality=modality,
        series_description=series_description,
        kvp=kvp,
        exposure=exposure,
        repetition_time=repetition_time,
        echo_time=echo_time,
        magnetic_field_strength=magnetic_field_strength,
        axcodes=axcodes,
        affine_json=affine_to_json(affine_ras),
        orientation_source=orientation_source,
        source_path=str(path.absolute()),
    )
