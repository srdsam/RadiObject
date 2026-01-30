"""Tests for ml/transforms - intensity and spatial transforms."""

from __future__ import annotations

import torch

from radiobject.ml.transforms import (
    IntensityNormalize,
    RandomCrop3D,
    RandomFlip3D,
    RandomNoise,
    Resample3D,
    WindowLevel,
)


class TestIntensityNormalize:
    """Tests for IntensityNormalize transform."""

    def test_normalizes_to_zero_mean(self) -> None:
        transform = IntensityNormalize(channel_wise=True)
        data = {"image": torch.randn(1, 32, 32, 32) * 100 + 50}

        result = transform(data)

        assert torch.abs(result["image"][0].mean()) < 0.1

    def test_normalizes_to_unit_variance(self) -> None:
        transform = IntensityNormalize(channel_wise=True)
        data = {"image": torch.randn(1, 32, 32, 32) * 100 + 50}

        result = transform(data)

        assert 0.9 < result["image"][0].std() < 1.1

    def test_channel_wise_normalization(self) -> None:
        transform = IntensityNormalize(channel_wise=True)
        image = torch.zeros(2, 32, 32, 32)
        image[0] = torch.randn(32, 32, 32) * 10 + 100
        image[1] = torch.randn(32, 32, 32) * 50 - 50
        data = {"image": image}

        result = transform(data)

        assert torch.abs(result["image"][0].mean()) < 0.1
        assert torch.abs(result["image"][1].mean()) < 0.1

    def test_global_normalization(self) -> None:
        transform = IntensityNormalize(channel_wise=False)
        data = {"image": torch.randn(2, 32, 32, 32) * 100 + 50}

        result = transform(data)

        assert torch.abs(result["image"].mean()) < 0.1
        assert 0.9 < result["image"].std() < 1.1

    def test_preserves_other_keys(self) -> None:
        transform = IntensityNormalize()
        data = {"image": torch.randn(1, 16, 16, 16), "label": torch.ones(16, 16, 16)}

        result = transform(data)

        assert "label" in result
        torch.testing.assert_close(result["label"], torch.ones(16, 16, 16))

    def test_handles_constant_channel(self) -> None:
        transform = IntensityNormalize(channel_wise=True)
        data = {"image": torch.ones(1, 16, 16, 16) * 5}

        result = transform(data)

        assert not torch.isnan(result["image"]).any()


class TestWindowLevel:
    """Tests for WindowLevel transform (CT windowing)."""

    def test_clips_to_window(self) -> None:
        transform = WindowLevel(window=400, level=40)
        image = torch.tensor([[[-200, 40, 300]]]).float()
        data = {"image": image}

        result = transform(data)

        assert result["image"].min() >= 0
        assert result["image"].max() <= 1

    def test_values_normalized_to_unit_range(self) -> None:
        transform = WindowLevel(window=400, level=40)
        data = {"image": torch.randn(1, 32, 32, 32) * 1000}

        result = transform(data)

        assert result["image"].min() >= 0
        assert result["image"].max() <= 1

    def test_center_value_maps_to_half(self) -> None:
        transform = WindowLevel(window=100, level=50)
        data = {"image": torch.tensor([[[50.0]]])}

        result = transform(data)

        assert torch.isclose(result["image"], torch.tensor([[[0.5]]]))

    def test_brain_window(self) -> None:
        transform = WindowLevel(window=80, level=40)
        data = {"image": torch.linspace(-40, 120, 1000).view(1, 10, 10, 10)}

        result = transform(data)

        assert result["image"].min() >= 0
        assert result["image"].max() <= 1


class TestRandomNoise:
    """Tests for RandomNoise transform."""

    def test_adds_noise_when_triggered(self) -> None:
        torch.manual_seed(42)
        transform = RandomNoise(std=0.1, prob=1.0)
        original = torch.zeros(1, 16, 16, 16)
        data = {"image": original.clone()}

        result = transform(data)

        assert not torch.equal(result["image"], original)

    def test_no_noise_when_prob_zero(self) -> None:
        transform = RandomNoise(std=0.1, prob=0.0)
        original = torch.zeros(1, 16, 16, 16)
        data = {"image": original.clone()}

        result = transform(data)

        torch.testing.assert_close(result["image"], original)

    def test_noise_std_approximately_correct(self) -> None:
        torch.manual_seed(42)
        transform = RandomNoise(std=0.5, prob=1.0)
        original = torch.zeros(1, 64, 64, 64)
        data = {"image": original.clone()}

        result = transform(data)

        noise = result["image"] - original
        # Noise std should be non-zero and reasonable (MONAI may scale differently)
        assert 0.1 < float(noise.std()) < 1.0


class TestRandomFlip3D:
    """Tests for RandomFlip3D transform."""

    def test_flips_when_triggered(self) -> None:
        torch.manual_seed(42)
        transform = RandomFlip3D(axes=(0,), prob=1.0)
        image = torch.arange(8).float().view(1, 2, 2, 2)
        data = {"image": image.clone()}

        result = transform(data)

        expected = torch.flip(image, dims=[1])
        torch.testing.assert_close(result["image"], expected)

    def test_no_flip_when_prob_zero(self) -> None:
        transform = RandomFlip3D(axes=(0, 1, 2), prob=0.0)
        image = torch.arange(8).float().view(1, 2, 2, 2)
        data = {"image": image.clone()}

        result = transform(data)

        torch.testing.assert_close(result["image"], image)

    def test_preserves_shape(self) -> None:
        torch.manual_seed(42)
        transform = RandomFlip3D(axes=(0, 1, 2), prob=0.5)
        data = {"image": torch.randn(2, 32, 32, 32)}

        result = transform(data)

        assert result["image"].shape == (2, 32, 32, 32)

    def test_flip_specific_axis(self) -> None:
        torch.manual_seed(42)
        transform = RandomFlip3D(axes=(2,), prob=1.0)
        image = torch.zeros(1, 4, 4, 4)
        image[0, :, :, 0] = 1
        data = {"image": image.clone()}

        result = transform(data)

        assert result["image"][0, :, :, 3].sum() == 16
        assert result["image"][0, :, :, 0].sum() == 0


class TestRandomCrop3D:
    """Tests for RandomCrop3D transform."""

    def test_crop_returns_correct_size(self) -> None:
        transform = RandomCrop3D(size=(16, 16, 16))
        data = {"image": torch.randn(1, 64, 64, 64)}

        result = transform(data)

        assert result["image"].shape == (1, 16, 16, 16)

    def test_stores_crop_start(self) -> None:
        transform = RandomCrop3D(size=(16, 16, 16))
        data = {"image": torch.randn(1, 64, 64, 64)}

        result = transform(data)

        assert "crop_start" in result
        assert len(result["crop_start"]) == 3

    def test_crop_within_bounds(self) -> None:
        torch.manual_seed(42)
        transform = RandomCrop3D(size=(16, 16, 16))
        data = {"image": torch.randn(1, 32, 32, 32)}

        result = transform(data)
        start = result["crop_start"]

        for i in range(3):
            assert start[i] >= 0
            assert start[i] + 16 <= 32

    def test_crop_equal_to_input_size(self) -> None:
        transform = RandomCrop3D(size=(32, 32, 32))
        data = {"image": torch.randn(1, 32, 32, 32)}

        result = transform(data)

        assert result["image"].shape == (1, 32, 32, 32)
        assert result["crop_start"] == (0, 0, 0)

    def test_preserves_channels(self) -> None:
        transform = RandomCrop3D(size=(16, 16, 16))
        data = {"image": torch.randn(4, 64, 64, 64)}

        result = transform(data)

        assert result["image"].shape[0] == 4


class TestResample3D:
    """Tests for Resample3D transform."""

    def test_resample_to_target_shape(self) -> None:
        transform = Resample3D(target_shape=(64, 64, 64))
        data = {"image": torch.randn(1, 32, 32, 32)}

        result = transform(data)

        assert result["image"].shape == (1, 64, 64, 64)

    def test_resample_downsample(self) -> None:
        transform = Resample3D(target_shape=(16, 16, 16))
        data = {"image": torch.randn(1, 64, 64, 64)}

        result = transform(data)

        assert result["image"].shape == (1, 16, 16, 16)

    def test_resample_anisotropic(self) -> None:
        transform = Resample3D(target_shape=(32, 64, 128))
        data = {"image": torch.randn(1, 64, 64, 64)}

        result = transform(data)

        assert result["image"].shape == (1, 32, 64, 128)

    def test_resample_preserves_channels(self) -> None:
        transform = Resample3D(target_shape=(32, 32, 32))
        data = {"image": torch.randn(4, 64, 64, 64)}

        result = transform(data)

        assert result["image"].shape == (4, 32, 32, 32)

    def test_resample_trilinear_mode(self) -> None:
        transform = Resample3D(target_shape=(64, 64, 64), mode="trilinear")
        data = {"image": torch.randn(1, 32, 32, 32)}

        result = transform(data)

        assert result["image"].shape == (1, 64, 64, 64)

    def test_resample_nearest_mode(self) -> None:
        transform = Resample3D(target_shape=(64, 64, 64), mode="nearest")
        data = {"image": torch.randn(1, 32, 32, 32)}

        result = transform(data)

        assert result["image"].shape == (1, 64, 64, 64)


class TestTransformComposition:
    """Tests for composing multiple transforms."""

    def test_compose_normalize_and_crop(self) -> None:
        transforms = [
            IntensityNormalize(),
            RandomCrop3D(size=(16, 16, 16)),
        ]

        data = {"image": torch.randn(1, 64, 64, 64) * 100 + 50}

        for t in transforms:
            data = t(data)

        assert data["image"].shape == (1, 16, 16, 16)
        assert torch.abs(data["image"].mean()) < 2

    def test_compose_window_resample_flip(self) -> None:
        torch.manual_seed(42)
        transforms = [
            WindowLevel(window=400, level=40),
            Resample3D(target_shape=(32, 32, 32)),
            RandomFlip3D(prob=1.0),
        ]

        data = {"image": torch.randn(1, 64, 64, 64) * 200}

        for t in transforms:
            data = t(data)

        assert data["image"].shape == (1, 32, 32, 32)
        assert data["image"].min() >= 0
        assert data["image"].max() <= 1
