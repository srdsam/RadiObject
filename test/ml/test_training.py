"""Integration tests for training loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ml.config import DatasetConfig, LoadingMode
from ml.datasets.multimodal import MultiModalDataset
from ml.datasets.volume_dataset import RadiObjectDataset
from ml.factory import create_training_dataloader
from ml.transforms import IntensityNormalize

if TYPE_CHECKING:
    from src.radi_object import RadiObject


class SimpleCNN(nn.Module):
    """Minimal 3D CNN for testing."""

    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class TestTrainingIntegration:
    """Integration tests for training workflow."""

    def test_single_forward_pass(self, ml_dataset: RadiObjectDataset) -> None:
        """Test forward pass through model with real data."""
        sample = ml_dataset[0]
        image = sample["image"].unsqueeze(0)

        model = SimpleCNN(in_channels=2)
        output = model(image)

        assert output.shape == (1, 2)

    def test_gradient_flow(self, ml_dataset: RadiObjectDataset) -> None:
        """Test gradients flow through model."""
        sample = ml_dataset[0]
        image = sample["image"].unsqueeze(0)
        target = torch.tensor([0])

        model = SimpleCNN(in_channels=2)
        criterion = nn.CrossEntropyLoss()

        output = model(image)
        loss = criterion(output, target)
        loss.backward()

        for param in model.parameters():
            if param.grad is not None:
                assert not torch.all(param.grad == 0)

    def test_training_epoch(self, populated_radi_object_module: "RadiObject") -> None:
        """Test complete training epoch with DataLoader."""
        config = DatasetConfig(
            loading_mode=LoadingMode.FULL_VOLUME,
            modalities=["flair"],
        )
        dataset = RadiObjectDataset(populated_radi_object_module, config)

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=True, drop_last=False
        )

        model = SimpleCNN(in_channels=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        model.train()
        epoch_loss = 0.0
        batch_count = 0
        for batch in loader:
            images = batch["image"]
            targets = torch.zeros(images.shape[0], dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        assert batch_count > 0
        assert epoch_loss > 0

    def test_multimodal_stacking(self, populated_radi_object_module: "RadiObject") -> None:
        """Test multimodal data stacks correctly along channel dimension."""
        dataset = MultiModalDataset(
            populated_radi_object_module,
            modalities=["flair", "T1w"],
        )

        sample = dataset[0]
        assert sample["image"].shape[0] == 2

        model = SimpleCNN(in_channels=2)
        output = model(sample["image"].unsqueeze(0))
        assert output.shape == (1, 2)


class TestFactoryDataloader:
    """Tests for factory-created dataloaders."""

    def test_create_training_dataloader(self, populated_radi_object_module: "RadiObject") -> None:
        """Test factory creates valid dataloader."""
        loader = create_training_dataloader(
            populated_radi_object_module,
            modalities=["flair"],
            batch_size=1,
            num_workers=0,
        )

        batch = next(iter(loader))
        assert "image" in batch
        assert batch["image"].shape[0] == 1

    def test_dataloader_with_patch(self, populated_radi_object_module: "RadiObject") -> None:
        """Test dataloader with patch extraction."""
        loader = create_training_dataloader(
            populated_radi_object_module,
            modalities=["flair"],
            patch_size=(32, 32, 32),
            batch_size=2,
            num_workers=0,
        )

        batch = next(iter(loader))
        assert batch["image"].shape == (2, 1, 32, 32, 32)


class TestTransformIntegration:
    """Tests for transform integration."""

    def test_transform_applied(self, populated_radi_object_module: "RadiObject") -> None:
        """Test transforms are applied to samples."""
        transform = IntensityNormalize()

        config = DatasetConfig(
            loading_mode=LoadingMode.FULL_VOLUME,
            modalities=["flair"],
        )
        dataset = RadiObjectDataset(
            populated_radi_object_module, config, transform=transform
        )

        sample = dataset[0]
        image = sample["image"]
        mean = image.mean().item()
        std = image.std().item()

        assert abs(mean) < 0.1
        assert abs(std - 1.0) < 0.1


class TestParameterizedTraining:
    """Training tests parameterized by storage backend."""

    def test_training_with_backend(self, ml_dataset_param: RadiObjectDataset) -> None:
        """Test training works with both local and S3 backends."""
        sample = ml_dataset_param[0]
        image = sample["image"].unsqueeze(0)

        model = SimpleCNN(in_channels=2)
        output = model(image)

        assert output.shape == (1, 2)
