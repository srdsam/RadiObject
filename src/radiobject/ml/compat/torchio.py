"""TorchIO integration for RadiObject."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import torch
from torch.utils.data import Dataset

from radiobject.ml.reader import VolumeReader
from radiobject.ml.utils.labels import LabelSource, load_labels

if TYPE_CHECKING:
    from radiobject.volume_collection import VolumeCollection

try:
    import torchio as tio

    HAS_TORCHIO = True
except ImportError:
    HAS_TORCHIO = False
    tio = None


def _require_torchio() -> None:
    if not HAS_TORCHIO:
        raise ImportError("TorchIO required. Install with: pip install radiobject[torchio]")


class VolumeCollectionSubjectsDataset(Dataset):
    """TorchIO-compatible dataset yielding Subject objects from VolumeCollection(s).

    Use this when you need TorchIO's Queue for efficient patch-based training,
    or when using TorchIO's specialized transforms.

    Example:
        # Single collection
        dataset = VolumeCollectionSubjectsDataset(radi.CT, labels="has_tumor")

        # Multi-modal
        dataset = VolumeCollectionSubjectsDataset(
            [radi.T1w, radi.FLAIR],
            labels=labels_df,
        )

        # With TorchIO transforms
        transform = tio.Compose([
            tio.ZNormalization(),
            tio.RandomFlip(axes=('LR',)),
        ])
        dataset = VolumeCollectionSubjectsDataset(radi.CT, labels="grade", transform=transform)

        # With TorchIO Queue for patch training
        sampler = tio.data.UniformSampler(patch_size=64)
        queue = tio.Queue(dataset, max_length=100, samples_per_volume=10, sampler=sampler)
        loader = DataLoader(queue, batch_size=16, num_workers=0)
    """

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

        # Create readers
        self._readers: dict[str, VolumeReader] = {}
        for name, coll in zip(self._collection_names, self._collections):
            self._readers[name] = VolumeReader(coll, ctx=coll._ctx)

        first_reader = self._readers[self._collection_names[0]]
        self._n_subjects = len(first_reader)

        # Load labels from first collection's obs
        self._labels: dict[int, Any] | None = None
        if labels is not None:
            first_coll = self._collections[0]
            obs_df = first_coll.obs.read() if isinstance(labels, str) else None
            self._labels = load_labels(first_reader, labels, obs_df)

    def __len__(self) -> int:
        return self._n_subjects

    def __getitem__(self, idx: int) -> "tio.Subject":
        """Return TorchIO Subject with images for all collections."""
        subject_dict: dict[str, Any] = {}

        for name in self._collection_names:
            data = self._readers[name].read_full(idx)
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
