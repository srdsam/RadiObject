"""Multi-modal dataset for loading aligned volumes from multiple VolumeCollections."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from radiobject.ml.utils.validation import validate_collection_alignment

if TYPE_CHECKING:
    from radiobject.radi_object import RadiObject
    from radiobject.volume_collection import VolumeCollection


class MultiModalDataset(Dataset):
    """Dataset for loading aligned volumes from multiple VolumeCollections."""

    def __init__(
        self,
        radi_object: RadiObject,
        modalities: list[str],
        label_column: str | None = None,
        value_filter: str | None = None,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        if not modalities:
            raise ValueError("At least one modality required")

        self._modalities = modalities
        self._transform = transform
        self._radi_object = radi_object

        self._collections: dict[str, VolumeCollection] = {
            mod: radi_object.collection(mod) for mod in modalities
        }

        first_coll = self._collections[modalities[0]]
        self._n_volumes = len(first_coll)
        self._volume_shape = first_coll.shape

        self._validate_alignment()

        self._labels: dict[int, Any] | None = None
        if label_column:
            self._load_labels(radi_object, label_column, value_filter)

    def _validate_alignment(self) -> None:
        """Validate that all modalities have matching subjects."""
        validate_collection_alignment(self._collections)

    def _load_labels(
        self,
        radi_object: RadiObject,
        label_column: str,
        value_filter: str | None,
    ) -> None:
        """Load labels from obs_meta."""
        obs_meta = radi_object.obs_meta.read(value_filter=value_filter)
        if label_column not in obs_meta.columns:
            raise ValueError(f"Label column '{label_column}' not found")

        first_coll = self._collections[self._modalities[0]]
        obs_subject_ids = first_coll.obs_subject_ids
        self._labels = {}
        for idx in range(self._n_volumes):
            subject_id = obs_subject_ids[idx]
            match = obs_meta[obs_meta["obs_subject_id"] == subject_id]
            if len(match) > 0:
                self._labels[idx] = match[label_column].iloc[0]

    def __len__(self) -> int:
        return self._n_volumes

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        volumes = [self._collections[mod].iloc[idx].to_numpy() for mod in self._modalities]

        stacked = np.stack(volumes, axis=0)

        first_coll = self._collections[self._modalities[0]]
        obs_id = first_coll.obs_ids[idx]

        result: dict[str, Any] = {
            "image": torch.from_numpy(stacked),
            "idx": idx,
            "obs_id": obs_id,
        }

        if self._labels is not None and idx in self._labels:
            label = self._labels[idx]
            if isinstance(label, (int, float, np.integer, np.floating)):
                result["label"] = torch.tensor(label)
            else:
                result["label"] = label

        if self._transform is not None:
            result = self._transform(result)

        return result

    @property
    def modalities(self) -> list[str]:
        """List of modalities."""
        return self._modalities

    @property
    def volume_shape(self) -> tuple[int, int, int]:
        """Volume dimensions."""
        return self._volume_shape
