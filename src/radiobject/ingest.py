"""NIfTI discovery utilities for bulk ingestion."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from glob import glob
from pathlib import Path


@dataclass(frozen=True)
class NiftiSource:
    """Source NIfTI file with optional paired label."""

    image_path: Path
    subject_id: str
    label_path: Path | None = None

    @property
    def has_label(self) -> bool:
        return self.label_path is not None


def discover_nifti_pairs(
    image_dir: str | Path,
    label_dir: str | Path | None = None,
    pattern: str = "*.nii.gz",
    subject_id_fn: Callable[[Path], str] | None = None,
) -> list[NiftiSource]:
    """Discover NIfTI files and optionally pair with labels.

    Args:
        image_dir: Directory containing image NIfTIs
        label_dir: Optional directory containing label NIfTIs (matched by filename)
        pattern: Glob pattern for finding NIfTI files
        subject_id_fn: Function to extract subject ID from path.
                      Default: stem without .nii extension

    Returns:
        List of NiftiSource objects
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if subject_id_fn is None:

        def subject_id_fn(p: Path) -> str:
            name = p.stem
            if name.endswith(".nii"):
                name = name[:-4]
            return name

    # Find all image files
    image_files = sorted(image_dir.glob(pattern))
    if not image_files:
        # Try non-gzipped pattern
        alt_pattern = pattern.replace(".gz", "")
        image_files = sorted(image_dir.glob(alt_pattern))

    if not image_files:
        raise ValueError(f"No NIfTI files found in {image_dir} with pattern {pattern}")

    # Build label lookup if label_dir provided
    label_lookup: dict[str, Path] = {}
    if label_dir is not None:
        label_dir = Path(label_dir)
        if not label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {label_dir}")

        for label_file in label_dir.glob(pattern):
            label_id = subject_id_fn(label_file)
            label_lookup[label_id] = label_file

        # Also try non-gzipped
        alt_pattern = pattern.replace(".gz", "")
        for label_file in label_dir.glob(alt_pattern):
            label_id = subject_id_fn(label_file)
            if label_id not in label_lookup:
                label_lookup[label_id] = label_file

    # Create NiftiSource objects
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
    """Resolve various NIfTI source formats to (path, subject_id) tuples.

    Supports:
    - Glob pattern: "./imagesTr/*.nii.gz"
    - Directory path: "./imagesTr"
    - Pre-resolved list: [(path, subject_id), ...]
    """
    # Already resolved - return as-is
    if isinstance(source, (list, tuple)) and source and isinstance(source[0], tuple):
        return [(Path(p), sid) for p, sid in source]

    source_str = str(source)

    if subject_id_fn is None:

        def subject_id_fn(p: Path) -> str:
            name = p.stem
            if name.endswith(".nii"):
                name = name[:-4]
            return name

    # Glob pattern
    if any(c in source_str for c in "*?["):
        matched = sorted(glob(source_str, recursive=True))
        if not matched:
            raise ValueError(f"No files matched pattern: {source}")
        return [(Path(f), subject_id_fn(Path(f))) for f in matched]

    # Directory path
    sources = discover_nifti_pairs(source)
    return [(s.image_path, s.subject_id) for s in sources]
