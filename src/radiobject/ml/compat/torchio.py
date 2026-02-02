"""TorchIO integration for RadiObject."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import torch
from torch.utils.data import Dataset

from radiobject._types import LabelSource
from radiobject.ml.utils.labels import load_labels

if TYPE_CHECKING:
    from radiobject.volume_collection import VolumeCollection

try:
    import torchio as tio

    HAS_TORCHIO = True
except ImportError:
    HAS_TORCHIO = False
    tio = None


def _require_torchio() -> None:
    """Raise ImportError if TorchIO not installed."""
    if not HAS_TORCHIO:
        raise ImportError("TorchIO required. Install with: pip install radiobject[torchio]")


class VolumeCollectionSubjectsDataset(Dataset):
    """TorchIO-compatible dataset yielding Subject objects from VolumeCollection(s)."""

    def __init__(
        self,
        collections: VolumeCollection | Sequence[VolumeCollection],
        labels: LabelSource = None,
        transform: Any | None = None,
    ):
        """Initialize TorchIO-compatible dataset.

        Args:
            collections: Single VolumeCollection or sequence of collections.
                Each collection becomes a separate image in the Subject.
            labels: Label source. Can be:
                - str: Column name in collection's obs DataFrame
                - pd.DataFrame: With obs_id as column/index and label values
                - dict[str, Any]: Mapping from obs_id to label
                - Callable[[str], Any]: Function taking obs_id, returning label
                - None: No labels
            transform: TorchIO transform (e.g., tio.Compose) applied to each Subject.
        """
        _require_torchio()

        # Normalize to list
        if not isinstance(collections, Sequence):
            collections = [collections]
        if not collections:
            raise ValueError("At least one collection required")

        self._collections = list(collections)
        self._collection_names = [c.name or f"collection_{i}" for i, c in enumerate(collections)]
        self._transform = transform

        first_coll = self._collections[0]
        self._n_subjects = len(first_coll)

        # Load labels from first collection's obs
        self._labels: dict[int, Any] | None = None
        if labels is not None:
            obs_df = first_coll.obs.read() if isinstance(labels, str) else None
            self._labels = load_labels(first_coll, labels, obs_df)

    def __len__(self) -> int:
        return self._n_subjects

    def __getitem__(self, idx: int) -> "tio.Subject":
        """Return TorchIO Subject with images for all collections."""
        subject_dict: dict[str, Any] = {}

        for name, coll in zip(self._collection_names, self._collections):
            data = coll.iloc[idx].to_numpy()
            tensor = torch.from_numpy(data).unsqueeze(0).float()
            subject_dict[name] = tio.ScalarImage(tensor=tensor)

        if self._labels is not None and idx in self._labels:
            subject_dict["label"] = self._labels[idx]

        subject = tio.Subject(subject_dict)

        if self._transform:
            subject = self._transform(subject)

        return subject

    @property
    def collection_names(self) -> list[str]:
        """Names of collections in each Subject."""
        return self._collection_names
