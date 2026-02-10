"""Image discovery utilities for bulk ingestion (NIfTI and DICOM)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from glob import glob
from pathlib import Path


class ImageFormat(Enum):
    """Supported medical image formats."""

    NIFTI = "nifti"
    DICOM = "dicom"


@dataclass(frozen=True)
class NiftiSource:
    """Source NIfTI file with optional paired label."""

    image_path: Path
    subject_id: str
    label_path: Path | None = None

    @property
    def has_label(self) -> bool:
        return self.label_path is not None


def _default_subject_id(p: Path) -> str:
    """Extract subject ID from path (stem without .nii extension)."""
    name = p.stem
    if name.endswith(".nii"):
        name = name[:-4]
    return name


def _is_nifti_path(p: str | Path) -> bool:
    """Check if a path looks like a NIfTI file."""
    s = str(p).lower()
    return s.endswith(".nii") or s.endswith(".nii.gz")


def detect_format(
    source: str | Path | Sequence[tuple[str | Path, str]],
) -> ImageFormat:
    """Auto-detect image format from source.

    Detection logic:
    - Pre-resolved tuples: inspect first tuple's path
    - Glob pattern: check for .nii in pattern
    - Directory: scan contents for .nii/.dcm files
    - File path: check extension
    """
    # Pre-resolved sequence of tuples
    if isinstance(source, (list, tuple)) and source and isinstance(source[0], tuple):
        first_path = Path(source[0][0])
        if _is_nifti_path(first_path):
            return ImageFormat.NIFTI
        if first_path.is_dir():
            return ImageFormat.DICOM
        raise ValueError(
            f"Cannot detect format from pre-resolved path: {first_path}. "
            "Use format_hint to specify explicitly."
        )

    source_str = str(source)

    # Glob pattern
    if any(c in source_str for c in "*?["):
        if ".nii" in source_str.lower():
            return ImageFormat.NIFTI
        raise ValueError(
            f"Cannot detect format from glob pattern: {source_str}. "
            "Use format_hint to specify explicitly."
        )

    # File or directory path
    path = Path(source_str)

    if _is_nifti_path(path):
        return ImageFormat.NIFTI

    if path.is_dir():
        # Check for NIfTI files
        nifti_files = list(path.glob("*.nii.gz")) + list(path.glob("*.nii"))
        if nifti_files:
            return ImageFormat.NIFTI

        # Check subdirectories for DICOM files
        for subdir in path.iterdir():
            if subdir.is_dir():
                dcm_files = list(subdir.glob("*.dcm"))
                if dcm_files:
                    return ImageFormat.DICOM
                # DICOM files sometimes have no extension — check for any files
                non_hidden = [
                    f for f in subdir.iterdir() if f.is_file() and not f.name.startswith(".")
                ]
                if non_hidden:
                    return ImageFormat.DICOM

        raise ValueError(
            f"Cannot detect format from directory: {path}. "
            "No .nii/.nii.gz files found, and no DICOM subdirectories detected. "
            "Use format_hint to specify explicitly."
        )

    raise ValueError(
        f"Cannot detect format from path: {path}. " "Use format_hint to specify explicitly."
    )


def discover_nifti_pairs(
    image_dir: str | Path,
    label_dir: str | Path | None = None,
    pattern: str = "*.nii.gz",
    subject_id_fn: Callable[[Path], str] | None = None,
) -> list[NiftiSource]:
    """Discover NIfTI files and optionally pair with labels."""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if subject_id_fn is None:
        subject_id_fn = _default_subject_id

    image_files = sorted(image_dir.glob(pattern))
    if not image_files:
        alt_pattern = pattern.replace(".gz", "")
        image_files = sorted(image_dir.glob(alt_pattern))

    if not image_files:
        raise ValueError(f"No NIfTI files found in {image_dir} with pattern {pattern}")

    label_lookup: dict[str, Path] = {}
    if label_dir is not None:
        label_dir = Path(label_dir)
        if not label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {label_dir}")

        for label_file in label_dir.glob(pattern):
            label_id = subject_id_fn(label_file)
            label_lookup[label_id] = label_file

        alt_pattern = pattern.replace(".gz", "")
        for label_file in label_dir.glob(alt_pattern):
            label_id = subject_id_fn(label_file)
            if label_id not in label_lookup:
                label_lookup[label_id] = label_file

    sources = []
    for image_path in image_files:
        subject_id = subject_id_fn(image_path)
        label_path = label_lookup.get(subject_id)
        sources.append(
            NiftiSource(
                image_path=image_path,
                subject_id=subject_id,
                label_path=label_path,
            )
        )

    return sources


def resolve_nifti_source(
    source: str | Path | Sequence[tuple[str | Path, str]],
    subject_id_fn: Callable[[Path], str] | None = None,
) -> list[tuple[Path, str]]:
    """Resolve various NIfTI source formats to (path, subject_id) tuples."""
    if isinstance(source, (list, tuple)) and source and isinstance(source[0], tuple):
        return [(Path(p), sid) for p, sid in source]

    source_str = str(source)

    if subject_id_fn is None:
        subject_id_fn = _default_subject_id

    if any(c in source_str for c in "*?["):
        matched = sorted(glob(source_str, recursive=True))
        if not matched:
            raise ValueError(f"No files matched pattern: {source}")
        return [(Path(f), subject_id_fn(Path(f))) for f in matched]

    sources = discover_nifti_pairs(source)
    return [(s.image_path, s.subject_id) for s in sources]


def resolve_dicom_source(
    source: str | Path | Sequence[tuple[str | Path, str]],
    subject_id_fn: Callable[[Path], str] | None = None,
) -> list[tuple[Path, str]]:
    """Resolve various DICOM source formats to (directory_path, subject_id) tuples.

    Supports:
    - Pre-resolved list: [(dicom_dir, subject_id), ...]
    - Directory path: each immediate subdirectory treated as one DICOM series
    """
    if isinstance(source, (list, tuple)) and source and isinstance(source[0], tuple):
        return [(Path(p), sid) for p, sid in source]

    source_str = str(source)

    if any(c in source_str for c in "*?["):
        raise ValueError(
            f"Glob patterns are not supported for DICOM sources: {source_str}. "
            "Provide a directory or pre-resolved list of (path, subject_id) tuples."
        )

    if subject_id_fn is None:

        def subject_id_fn(p: Path) -> str:
            return p.name

    path = Path(source_str)
    if not path.is_dir():
        raise FileNotFoundError(f"DICOM source directory not found: {path}")

    subdirs = sorted(d for d in path.iterdir() if d.is_dir())
    if not subdirs:
        raise ValueError(
            f"No subdirectories found in DICOM source: {path}. "
            "Each subdirectory should contain one DICOM series."
        )

    return [(d, subject_id_fn(d)) for d in subdirs]


def resolve_image_source(
    source: str | Path | Sequence[tuple[str | Path, str]],
    format_hint: ImageFormat | None = None,
) -> tuple[list[tuple[Path, str]], ImageFormat]:
    """Resolve an image source, auto-detecting format if needed.

    Returns (resolved_items, detected_format).
    """
    fmt = format_hint if format_hint is not None else detect_format(source)

    if fmt == ImageFormat.NIFTI:
        return resolve_nifti_source(source), fmt
    else:
        return resolve_dicom_source(source), fmt
