"""Tests for parallel execution utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from radiobject.parallel import WriteResult, create_worker_ctx, map_on_threads
from radiobject.volume import Volume
from radiobject.volume_collection import VolumeCollection


class TestMapOnThreads:
    """Tests for map_on_threads utility."""

    def test_preserves_order(self):
        """Results are returned in input order regardless of completion order."""
        items = [3, 1, 4, 1, 5]
        results = map_on_threads(lambda x: x * 2, items)
        assert results == [6, 2, 8, 2, 10]

    def test_empty_input(self):
        """Empty input returns empty list."""
        assert map_on_threads(lambda x: x, []) == []

    def test_single_item(self):
        """Single item list works correctly."""
        results = map_on_threads(lambda x: x + 1, [5])
        assert results == [6]

    def test_propagates_exceptions(self):
        """Exceptions from workers are propagated to caller."""

        def failing(x: int) -> int:
            if x == 2:
                raise ValueError("fail on 2")
            return x

        with pytest.raises(ValueError, match="fail on 2"):
            map_on_threads(failing, [1, 2, 3])

    def test_respects_max_workers(self):
        """Max workers parameter limits concurrency."""
        results = map_on_threads(lambda x: x * 2, [1, 2, 3, 4, 5], max_workers=2)
        assert results == [2, 4, 6, 8, 10]


class TestCreateWorkerCtx:
    """Tests for create_worker_ctx utility."""

    def test_creates_ctx_from_global_config(self):
        """Creates context from global config when no base_ctx provided."""
        ctx = create_worker_ctx()
        assert ctx is not None

    def test_creates_ctx_from_base_ctx(self, custom_tiledb_ctx):
        """Creates context by copying config from base context."""
        ctx = create_worker_ctx(custom_tiledb_ctx)
        assert ctx is not None


class TestWriteResult:
    """Tests for WriteResult dataclass."""

    def test_success_result(self):
        """Successful write result has no error."""
        result = WriteResult(index=0, uri="/path/to/vol", obs_id="PAT001", success=True)
        assert result.success is True
        assert result.error is None

    def test_failure_result(self):
        """Failed write result captures error."""
        err = ValueError("test error")
        result = WriteResult(
            index=0, uri="/path/to/vol", obs_id="PAT001", success=False, error=err
        )
        assert result.success is False
        assert result.error is err

    def test_immutable(self):
        """WriteResult is frozen/immutable."""
        result = WriteResult(index=0, uri="/path", obs_id="PAT001", success=True)
        with pytest.raises(Exception):
            result.index = 1  # type: ignore


class TestParallelVolumeWrites:
    """Integration tests for parallel volume writes."""

    def test_from_volumes_parallel_produces_correct_data(
        self, temp_dir: Path, collection_shape: tuple[int, int, int]
    ):
        """Parallel writes in from_volumes produce identical data to sequential."""
        rng = np.random.default_rng(42)
        volumes = []
        original_data = {}

        # Use smaller shape for faster tests
        test_shape = (64, 64, 32)
        for i in range(4):
            obs_id = f"PAT{i:03d}"
            uri = str(temp_dir / f"src_vol_{i}")
            data = rng.random(test_shape, dtype=np.float32)
            original_data[obs_id] = data.copy()
            vol = Volume.from_numpy(uri, data)
            vol.set_obs_id(obs_id)
            volumes.append((obs_id, vol))

        collection_uri = str(temp_dir / "parallel_collection")
        collection = VolumeCollection._from_volumes(collection_uri, volumes)

        assert len(collection) == 4
        for i, obs_id in enumerate(collection.obs_ids):
            stored_data = collection.iloc[i].to_numpy()
            np.testing.assert_array_almost_equal(stored_data, original_data[obs_id])

    def test_from_volumes_parallel_with_obs_data(
        self, temp_dir: Path, collection_shape: tuple[int, int, int]
    ):
        """Parallel writes preserve obs metadata correctly."""
        rng = np.random.default_rng(42)
        volumes = []

        # Use smaller shape for faster tests
        test_shape = (64, 64, 32)
        for i in range(3):
            obs_id = f"PAT{i:03d}"
            uri = str(temp_dir / f"src_vol_{i}")
            data = rng.random(test_shape, dtype=np.float32)
            vol = Volume.from_numpy(uri, data)
            vol.set_obs_id(obs_id)
            volumes.append((obs_id, vol))

        obs_df = pd.DataFrame(
            {
                "obs_id": ["PAT000", "PAT001", "PAT002"],
                "obs_subject_id": ["PAT000", "PAT001", "PAT002"],
                "age": [45, 52, 38],
                "diagnosis": ["healthy", "tumor", "healthy"],
            }
        )

        collection_uri = str(temp_dir / "parallel_collection_obs")
        collection = VolumeCollection._from_volumes(
            collection_uri, volumes, obs_data=obs_df
        )

        assert "age" in collection.obs.columns
        obs_id = collection.index.get_key(1)
        row = collection.get_obs_row_by_obs_id(obs_id)
        assert row["age"].iloc[0] == 52
        assert row["diagnosis"].iloc[0] == "tumor"

    def test_parallel_collection_roundtrip(
        self, temp_dir: Path, volumes_module: list[tuple[str, Volume]]
    ):
        """VolumeCollection created with parallel writes can be reopened."""
        uri = str(temp_dir / "roundtrip_parallel")
        collection = VolumeCollection._from_volumes(uri, volumes_module)

        reopened = VolumeCollection(uri)
        assert len(reopened) == 3

        for i in range(3):
            original = collection.iloc[i].to_numpy()
            reloaded = reopened.iloc[i].to_numpy()
            np.testing.assert_array_equal(original, reloaded)
