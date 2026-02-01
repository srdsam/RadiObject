"""Tests for radiobject.utils module."""

from __future__ import annotations

import json

import numpy as np
import pytest

from radiobject.utils import affine_to_json, affine_to_list


class TestAffineConversion:
    """Tests for affine matrix conversion utilities."""

    def test_affine_to_list_identity(self):
        """Identity matrix converts to nested list correctly."""
        affine = np.eye(4)
        result = affine_to_list(affine)

        assert len(result) == 4
        assert all(len(row) == 4 for row in result)
        assert result[0][0] == 1.0
        assert result[1][1] == 1.0

    def test_affine_to_list_custom_values(self):
        """Custom affine matrix values are preserved."""
        affine = np.array(
            [
                [1.0, 0.0, 0.0, 10.5],
                [0.0, 2.0, 0.0, -5.2],
                [0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        result = affine_to_list(affine)

        assert result[0][3] == pytest.approx(10.5)
        assert result[1][1] == pytest.approx(2.0)
        assert result[1][3] == pytest.approx(-5.2)

    def test_affine_to_json_valid_json(self):
        """affine_to_json produces valid JSON string."""
        affine = np.eye(4)
        json_str = affine_to_json(affine)

        parsed = json.loads(json_str)
        assert len(parsed) == 4

    def test_affine_to_json_roundtrip(self):
        """JSON roundtrip preserves affine values."""
        affine = np.array(
            [
                [0.5, 0.1, 0.0, 100.0],
                [0.0, 0.5, 0.2, -50.0],
                [0.0, 0.0, 1.5, 25.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        json_str = affine_to_json(affine)
        parsed = json.loads(json_str)
        reconstructed = np.array(parsed)

        np.testing.assert_array_almost_equal(affine, reconstructed)
