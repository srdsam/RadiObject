"""TorchIO integration for RadiObject."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset

from radiobject.ml.reader import VolumeReader

if TYPE_CHECKING:
    from radiobject.radi_object import RadiObject

try:
    import torchio as tio

    HAS_TORCHIO = True
except ImportError:
    HAS_TORCHIO = False
    tio = None


def _require_torchio() -> None:
    if not HAS_TORCHIO:
        raise ImportError("TorchIO required. Install with: pip install radiobject[torchio]")


class RadiObjectSubjectsDataset(Dataset):
    """TorchIO-compatible dataset yielding Subject objects.

    Use this when you need TorchIO's Queue for efficient patch-based training,
    or when using TorchIO's specialized transforms.

    Example:
        dataset = RadiObjectSubjectsDataset(radi, modalities=["T1w", "FLAIR"])

        # With TorchIO transforms
        transform = tio.Compose([
            tio.ZNormalization(),
            tio.RandomFlip(axes=('LR',)),
        ])
        dataset = RadiObjectSubjectsDataset(radi, modalities=["T1w"], transform=transform)

        # With TorchIO Queue for patch training
        sampler = tio.data.UniformSampler(patch_size=64)
        queue = tio.Queue(dataset, max_length=100, samples_per_volume=10, sampler=sampler)
        loader = DataLoader(queue, batch_size=16, num_workers=0)
    """

    def __init__(
        self,
        radi_object: "RadiObject",
        modalities: list[str],
        label_column: str | None = None,
        transform: Any | None = None,
    ):
        _require_torchio()

        self._modalities = modalities
        self._transform = transform

        self._readers = {
            mod: VolumeReader(radi_object.collection(mod), ctx=radi_object._ctx)
            for mod in modalities
        }

        first_reader = self._readers[modalities[0]]
        self._n_subjects = len(first_reader)

        self._labels: dict[int, int | float] | None = None
        if label_column:
            obs_meta = radi_object.obs_meta.read()
            self._labels = {}
            for idx in range(self._n_subjects):
                obs_id = first_reader.get_obs_id(idx)
                parts = obs_id.rsplit("_", 1)
                subject_id = parts[0] if len(parts) > 1 else obs_id
                match = obs_meta[obs_meta["obs_subject_id"] == subject_id]
                if len(match) > 0:
                    self._labels[idx] = match[label_column].iloc[0]

    def __len__(self) -> int:
        return self._n_subjects

    def __getitem__(self, idx: int) -> "tio.Subject":
        """Return TorchIO Subject with images for all modalities."""
        subject_dict: dict[str, Any] = {}

        for mod in self._modalities:
            data = self._readers[mod].read_full(idx)
            tensor = torch.from_numpy(data).unsqueeze(0).float()
            subject_dict[mod] = tio.ScalarImage(tensor=tensor)

        if self._labels is not None and idx in self._labels:
            subject_dict["label"] = self._labels[idx]

        subject = tio.Subject(subject_dict)

        if self._transform:
            subject = self._transform(subject)

        return subject

    @property
    def modalities(self) -> list[str]:
        return self._modalities
