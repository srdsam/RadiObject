"""Validation utilities for ML datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from radiobject.volume_collection import VolumeCollection


def validate_collection_alignment(collections: dict[str, VolumeCollection]) -> None:
    """Validate all collections have matching subjects by obs_subject_id.

    For multi-modal training, volumes from different collections must correspond
    to the same subjects. This validates alignment using the obs_subject_id field
    directly (no string parsing).

    Args:
        collections: Dict mapping collection names to VolumeCollection instances.

    Raises:
        ValueError: If collections have different volume counts or mismatched subjects.
    """
    if len(collections) < 2:
        return

    names = list(collections.keys())
    first_name = names[0]
    first_coll = collections[first_name]
    n_volumes = len(first_coll)

    first_subjects = set(first_coll.obs_subject_ids)

    for name in names[1:]:
        coll = collections[name]
        if len(coll) != n_volumes:
            raise ValueError(f"Collection '{name}' has {len(coll)} volumes, expected {n_volumes}")

        mod_subjects = set(coll.obs_subject_ids)

        if mod_subjects != first_subjects:
            missing = first_subjects - mod_subjects
            extra = mod_subjects - first_subjects
            raise ValueError(
                f"Subject mismatch for collection '{name}': "
                f"missing={list(missing)[:3]}, extra={list(extra)[:3]}"
            )


def validate_uniform_shapes(collections: dict[str, VolumeCollection]) -> tuple[int, int, int]:
    """Validate all collections have uniform shapes and return the common shape.

    Args:
        collections: Dict mapping collection names to VolumeCollection instances.

    Returns:
        Common volume shape (X, Y, Z).

    Raises:
        ValueError: If any collection has non-uniform shapes or shapes don't match.
    """
    shape: tuple[int, int, int] | None = None

    for name, coll in collections.items():
        if not coll.is_uniform:
            raise ValueError(
                f"Collection '{name}' has heterogeneous shapes. "
                f"Resample to uniform dimensions before ML training."
            )

        coll_shape = coll.shape
        if coll_shape is None:
            raise ValueError(f"Collection '{name}' has no shape metadata.")

        if shape is None:
            shape = coll_shape
        elif coll_shape != shape:
            raise ValueError(
                f"Shape mismatch: collection '{name}' has shape {coll_shape}, " f"expected {shape}"
            )

    if shape is None:
        raise ValueError("No collections provided")

    return shape
