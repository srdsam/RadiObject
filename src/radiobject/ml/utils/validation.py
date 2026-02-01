"""Validation utilities for ML datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from radiobject.ml.reader import VolumeReader


def validate_collection_alignment(readers: dict[str, VolumeReader]) -> None:
    """Validate all readers have matching subjects by obs_subject_id.

    For multi-modal training, volumes from different collections must correspond
    to the same subjects. This validates alignment using the obs_subject_id field
    directly (no string parsing).

    Args:
        readers: Dict mapping collection names to VolumeReader instances.

    Raises:
        ValueError: If collections have different volume counts or mismatched subjects.
    """
    if len(readers) < 2:
        return

    names = list(readers.keys())
    first_name = names[0]
    first_reader = readers[first_name]
    n_volumes = len(first_reader)

    first_subjects = {first_reader.get_obs_subject_id(idx) for idx in range(n_volumes)}

    for name in names[1:]:
        reader = readers[name]
        if len(reader) != n_volumes:
            raise ValueError(f"Collection '{name}' has {len(reader)} volumes, expected {n_volumes}")

        mod_subjects = {reader.get_obs_subject_id(idx) for idx in range(len(reader))}

        if mod_subjects != first_subjects:
            missing = first_subjects - mod_subjects
            extra = mod_subjects - first_subjects
            raise ValueError(
                f"Subject mismatch for collection '{name}': "
                f"missing={list(missing)[:3]}, extra={list(extra)[:3]}"
            )


def validate_uniform_shapes(readers: dict[str, VolumeReader]) -> tuple[int, int, int]:
    """Validate all readers have uniform shapes and return the common shape.

    Args:
        readers: Dict mapping collection names to VolumeReader instances.

    Returns:
        Common volume shape (X, Y, Z).

    Raises:
        ValueError: If any collection has non-uniform shapes or shapes don't match.
    """
    shape: tuple[int, int, int] | None = None

    for name, reader in readers.items():
        if not reader.is_uniform:
            raise ValueError(
                f"Collection '{name}' has heterogeneous shapes. "
                f"Resample to uniform dimensions before ML training."
            )

        reader_shape = reader.shape
        if reader_shape is None:
            raise ValueError(f"Collection '{name}' has no shape metadata.")

        if shape is None:
            shape = reader_shape
        elif reader_shape != shape:
            raise ValueError(
                f"Shape mismatch: collection '{name}' has shape {reader_shape}, "
                f"expected {shape}"
            )

    if shape is None:
        raise ValueError("No collections provided")

    return shape
