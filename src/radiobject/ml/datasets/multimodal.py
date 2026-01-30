"""Multi-modal dataset for loading aligned volumes from multiple VolumeCollections."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from radiobject.ml.reader import VolumeReader

if TYPE_CHECKING:
    from radiobject.radi_object import RadiObject


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

        self._readers = {
            mod: VolumeReader(radi_object.collection(mod), ctx=radi_object._ctx)
            for mod in modalities
        }

        first_reader = self._readers[modalities[0]]
        self._n_volumes = len(first_reader)
        self._volume_shape = first_reader.shape

        self._validate_alignment()

        self._labels: dict[int, int | float] | None = None
        if label_column:
            self._load_labels(radi_object, label_column, value_filter)

    def _validate_alignment(self) -> None:
        """Validate that all modalities have matching subjects."""
        first_mod = self._modalities[0]
        first_reader = self._readers[first_mod]

        first_subjects = set()
        for idx in range(len(first_reader)):
            obs_id = first_reader.get_obs_id(idx)
            parts = obs_id.rsplit("_", 1)
            subject = parts[0] if len(parts) > 1 else obs_id
            first_subjects.add(subject)

        for mod in self._modalities[1:]:
            reader = self._readers[mod]
            if len(reader) != self._n_volumes:
                raise ValueError(
                    f"Modality '{mod}' has {len(reader)} volumes, expected {self._n_volumes}"
                )

            mod_subjects = set()
            for idx in range(len(reader)):
                obs_id = reader.get_obs_id(idx)
                parts = obs_id.rsplit("_", 1)
                subject = parts[0] if len(parts) > 1 else obs_id
                mod_subjects.add(subject)

            if mod_subjects != first_subjects:
                missing = first_subjects - mod_subjects
                extra = mod_subjects - first_subjects
                raise ValueError(
                    f"Subject mismatch for modality '{mod}': "
                    f"missing={list(missing)[:3]}, extra={list(extra)[:3]}"
                )

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

        first_reader = self._readers[self._modalities[0]]
        self._labels = {}
        for idx in range(self._n_volumes):
            obs_id = first_reader.get_obs_id(idx)
            parts = obs_id.rsplit("_", 1)
            subject_id = parts[0] if len(parts) > 1 else obs_id
            match = obs_meta[obs_meta["obs_subject_id"] == subject_id]
            if len(match) > 0:
                self._labels[idx] = match[label_column].iloc[0]

    def __len__(self) -> int:
        return self._n_volumes

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        volumes = [self._readers[mod].read_full(idx) for mod in self._modalities]

        stacked = np.stack(volumes, axis=0)

        first_reader = self._readers[self._modalities[0]]
        obs_id = first_reader.get_obs_id(idx)

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
