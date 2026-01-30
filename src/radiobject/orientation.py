"""Orientation detection and reorientation for radiology volumes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Protocol

import nibabel as nib
import numpy as np
from pydantic import BaseModel, Field

from radiobject.utils import affine_to_list


class OrientationInfo(BaseModel):
    """Anatomical orientation information for a volume."""

    axcodes: tuple[str, str, str] = Field(
        description="Axis codes indicating anatomical direction, e.g., ('R', 'A', 'S')"
    )
    affine: list[list[float]] = Field(description="4x4 affine transformation matrix")
    is_canonical: bool = Field(description="True if orientation is RAS canonical")
    confidence: Literal["header", "inferred", "unknown"] = Field(
        description="Source confidence: header-based, ML-inferred, or unknown"
    )
    source: Literal["nifti_sform", "nifti_qform", "dicom_iop", "identity"] = Field(
        description="Where the orientation was derived from"
    )

    model_config = {"frozen": True}


class OrientationInferenceModel(Protocol):
    """Protocol for ML-based orientation inference (future implementation)."""

    def infer(self, data: np.ndarray) -> OrientationInfo:
        """Infer orientation from volume data when headers are unavailable."""
        ...


def _is_identity_affine(affine: np.ndarray, tol: float = 1e-6) -> bool:
    """Check if affine is effectively an identity matrix."""
    return np.allclose(affine, np.eye(4), atol=tol)


def _is_degenerate_affine(affine: np.ndarray) -> bool:
    """Check if affine has degenerate determinant (not invertible or extreme scaling)."""
    det = np.linalg.det(affine[:3, :3])
    return not (1e-6 < abs(det) < 1e6)


def _are_axes_orthogonal(affine: np.ndarray, tol: float = 1e-4) -> bool:
    """Check if rotation/scaling axes are orthogonal within tolerance."""
    rotation_matrix = affine[:3, :3]
    gram = rotation_matrix.T @ rotation_matrix
    # For orthogonal axes, off-diagonal elements should be near zero
    off_diag = gram - np.diag(np.diag(gram))
    return np.allclose(off_diag, 0, atol=tol * np.max(np.abs(np.diag(gram))))


def is_orientation_valid(info: OrientationInfo) -> bool:
    """Check if orientation information represents a valid, usable transform."""
    affine = np.array(info.affine)

    if _is_identity_affine(affine):
        return False

    if _is_degenerate_affine(affine):
        return False

    if not _are_axes_orthogonal(affine):
        return False

    return True


def detect_nifti_orientation(img: nib.Nifti1Image) -> OrientationInfo:
    """Extract orientation information from a NIfTI image header."""
    header = img.header

    # Determine affine source: sform has priority if valid
    sform_code = int(header.get("sform_code", 0))
    qform_code = int(header.get("qform_code", 0))

    if sform_code > 0:
        affine = img.get_sform()
        source: Literal["nifti_sform", "nifti_qform", "dicom_iop", "identity"] = "nifti_sform"
    elif qform_code > 0:
        affine = img.get_qform()
        source = "nifti_qform"
    else:
        affine = img.affine
        source = "identity"

    # Check if affine is effectively identity (common default when missing)
    if _is_identity_affine(affine):
        source = "identity"

    # Get axis codes using nibabel's orientation utilities
    ornt = nib.orientations.io_orientation(affine)
    axcodes = tuple(nib.orientations.ornt2axcodes(ornt))

    # Check if already RAS canonical
    is_canonical = axcodes == ("R", "A", "S")

    # Confidence based on source (identity means unreliable header)
    confidence: Literal["header", "inferred", "unknown"] = (
        "header" if source != "identity" else "unknown"
    )

    return OrientationInfo(
        axcodes=axcodes,
        affine=affine_to_list(affine),
        is_canonical=is_canonical,
        confidence=confidence,
        source=source,
    )


def detect_dicom_orientation(series_path: Path) -> OrientationInfo:
    """Extract orientation from a DICOM series using Image Orientation Patient tag."""
    import pydicom

    # Find all DICOM files in the directory
    dicom_files = sorted(series_path.glob("*.dcm"))
    if not dicom_files:
        # Try without extension filtering
        dicom_files = sorted(
            f for f in series_path.iterdir() if f.is_file() and not f.name.startswith(".")
        )

    if not dicom_files:
        return OrientationInfo(
            axcodes=("R", "A", "S"),
            affine=affine_to_list(np.eye(4)),
            is_canonical=True,
            confidence="unknown",
            source="identity",
        )

    # Read first DICOM for orientation info
    ds = pydicom.dcmread(dicom_files[0])

    # Get Image Orientation Patient (direction cosines)
    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = getattr(ds, "ImagePositionPatient", None)
    pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
    slice_thickness = getattr(ds, "SliceThickness", 1.0)

    if iop is None:
        return OrientationInfo(
            axcodes=("R", "A", "S"),
            affine=affine_to_list(np.eye(4)),
            is_canonical=True,
            confidence="unknown",
            source="identity",
        )

    # Build affine from DICOM tags
    row_cosines = np.array([float(iop[0]), float(iop[1]), float(iop[2])])
    col_cosines = np.array([float(iop[3]), float(iop[4]), float(iop[5])])

    # Slice direction is cross product of row and column
    slice_cosines = np.cross(row_cosines, col_cosines)

    # Build rotation matrix
    voxel_spacing = [float(pixel_spacing[1]), float(pixel_spacing[0]), float(slice_thickness)]

    affine = np.eye(4)
    affine[:3, 0] = row_cosines * voxel_spacing[0]
    affine[:3, 1] = col_cosines * voxel_spacing[1]
    affine[:3, 2] = slice_cosines * voxel_spacing[2]

    if ipp is not None:
        affine[:3, 3] = [float(ipp[0]), float(ipp[1]), float(ipp[2])]

    # Convert DICOM LPS to nibabel-style for consistent axis code computation
    # DICOM uses LPS, nibabel uses RAS
    lps_to_ras = np.diag([-1, -1, 1, 1])
    affine_ras = lps_to_ras @ affine

    # Get axis codes
    ornt = nib.orientations.io_orientation(affine_ras)
    axcodes = tuple(nib.orientations.ornt2axcodes(ornt))

    is_canonical = axcodes == ("R", "A", "S")

    return OrientationInfo(
        axcodes=axcodes,
        affine=affine_to_list(affine_ras),
        is_canonical=is_canonical,
        confidence="header",
        source="dicom_iop",
    )


def reorient_to_canonical(
    data: np.ndarray,
    affine: np.ndarray,
    target: Literal["RAS", "LAS", "LPS"] = "RAS",
) -> tuple[np.ndarray, np.ndarray]:
    """Reorient volume data to canonical orientation.

    Uses nibabel's as_closest_canonical() internally for RAS, with axis flipping
    for other target orientations.
    """
    # Create temporary NIfTI image to use nibabel's reorientation
    img = nib.Nifti1Image(data, affine)

    if target == "RAS":
        canonical = nib.as_closest_canonical(img)
        return np.asarray(canonical.dataobj), canonical.affine

    # For other targets, first go to RAS then flip axes
    canonical = nib.as_closest_canonical(img)
    reoriented_data = np.asarray(canonical.dataobj)
    reoriented_affine = canonical.affine.copy()

    if target == "LAS":
        # Flip X axis (R->L)
        reoriented_data = np.flip(reoriented_data, axis=0)
        reoriented_affine[0, :] = -reoriented_affine[0, :]
        reoriented_affine[0, 3] = -reoriented_affine[0, 3]
    elif target == "LPS":
        # Flip X and Y axes (R->L, A->P)
        reoriented_data = np.flip(np.flip(reoriented_data, axis=0), axis=1)
        reoriented_affine[0, :] = -reoriented_affine[0, :]
        reoriented_affine[1, :] = -reoriented_affine[1, :]
        reoriented_affine[0, 3] = -reoriented_affine[0, 3]
        reoriented_affine[1, 3] = -reoriented_affine[1, 3]

    return reoriented_data, reoriented_affine


def orientation_info_to_metadata(
    info: OrientationInfo, original_affine: np.ndarray | None = None
) -> dict[str, str]:
    """Convert OrientationInfo to TileDB metadata key-value pairs."""
    metadata = {
        "orientation_axcodes": "".join(info.axcodes),
        "orientation_affine": json.dumps(info.affine),
        "orientation_source": info.source,
        "orientation_confidence": info.confidence,
    }

    if original_affine is not None:
        metadata["original_affine"] = json.dumps(affine_to_list(original_affine))

    return metadata


def metadata_to_orientation_info(metadata: dict) -> OrientationInfo | None:
    """Reconstruct OrientationInfo from TileDB metadata."""
    axcodes_str = metadata.get("orientation_axcodes")
    affine_json = metadata.get("orientation_affine")

    if not axcodes_str or not affine_json:
        return None

    axcodes = tuple(axcodes_str)
    affine = json.loads(affine_json)
    source = metadata.get("orientation_source", "identity")
    confidence = metadata.get("orientation_confidence", "unknown")

    return OrientationInfo(
        axcodes=axcodes,
        affine=affine,
        is_canonical=axcodes == ("R", "A", "S"),
        confidence=confidence,
        source=source,
    )
