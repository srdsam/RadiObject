"""Tests for StreamingWriter and RadiObjectWriter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from radiobject.streaming import StreamingWriter, RadiObjectWriter
from radiobject.radi_object import RadiObject
from radiobject.volume_collection import VolumeCollection


class TestStreamingWriterBasic:
    """Tests for StreamingWriter context manager."""

    def test_creates_volume_collection(self, temp_dir: Path):
        """StreamingWriter creates a valid VolumeCollection."""
        uri = str(temp_dir / "streaming_vc")
        shape = (64, 64, 32)

        with StreamingWriter(uri, shape=shape) as writer:
            data = np.random.randn(*shape).astype(np.float32)
            writer.write_volume(data, obs_id="vol_001", obs_subject_id="sub-001")

        vc = VolumeCollection(uri)
        assert len(vc) == 1
        assert vc.shape == shape

    def test_write_multiple_volumes(self, temp_dir: Path):
        """StreamingWriter can write multiple volumes sequentially."""
        uri = str(temp_dir / "streaming_multi")
        shape = (32, 32, 16)

        with StreamingWriter(uri, shape=shape) as writer:
            for i in range(3):
                data = np.random.randn(*shape).astype(np.float32)
                writer.write_volume(
                    data,
                    obs_id=f"vol_{i:03d}",
                    obs_subject_id=f"sub-{i:03d}",
                )

        vc = VolumeCollection(uri)
        assert len(vc) == 3

    def test_n_written_tracks_count(self, temp_dir: Path):
        """n_written property tracks volumes written so far."""
        uri = str(temp_dir / "streaming_count")
        shape = (16, 16, 8)

        with StreamingWriter(uri, shape=shape) as writer:
            assert writer.n_written == 0

            data = np.random.randn(*shape).astype(np.float32)
            writer.write_volume(data, obs_id="vol_001", obs_subject_id="sub-001")
            assert writer.n_written == 1

            writer.write_volume(data, obs_id="vol_002", obs_subject_id="sub-002")
            assert writer.n_written == 2


class TestStreamingWriterWithSchema:
    """Tests for StreamingWriter with obs schema."""

    def test_write_with_obs_attrs(self, temp_dir: Path):
        """StreamingWriter writes additional obs attributes."""
        uri = str(temp_dir / "streaming_attrs")
        shape = (32, 32, 16)
        obs_schema = {
            "age": np.dtype("int32"),
            "diagnosis": np.dtype("U64"),
        }

        with StreamingWriter(uri, shape=shape, obs_schema=obs_schema) as writer:
            data = np.random.randn(*shape).astype(np.float32)
            writer.write_volume(
                data,
                obs_id="vol_001",
                obs_subject_id="sub-001",
                age=45,
                diagnosis="healthy",
            )

        vc = VolumeCollection(uri)
        obs_df = vc.obs.read()
        assert "age" in obs_df.columns
        assert "diagnosis" in obs_df.columns
        assert obs_df.iloc[0]["age"] == 45


class TestStreamingWriterValidation:
    """Tests for StreamingWriter validation."""

    def test_shape_mismatch_raises(self, temp_dir: Path):
        """Writing volume with wrong shape raises ValueError."""
        uri = str(temp_dir / "streaming_shape_error")
        shape = (32, 32, 16)

        with pytest.raises(ValueError, match="shape"):
            with StreamingWriter(uri, shape=shape) as writer:
                wrong_shape = (64, 64, 32)
                data = np.random.randn(*wrong_shape).astype(np.float32)
                writer.write_volume(data, obs_id="vol_001", obs_subject_id="sub-001")

    def test_context_manager_required(self, temp_dir: Path):
        """Writing without context manager raises RuntimeError."""
        uri = str(temp_dir / "streaming_no_ctx")
        shape = (32, 32, 16)

        writer = StreamingWriter(uri, shape=shape)
        data = np.random.randn(*shape).astype(np.float32)

        with pytest.raises(RuntimeError, match="not initialized"):
            writer.write_volume(data, obs_id="vol_001", obs_subject_id="sub-001")

    def test_double_init_raises(self, temp_dir: Path):
        """Entering context twice raises RuntimeError."""
        uri = str(temp_dir / "streaming_double_init")
        shape = (32, 32, 16)

        writer = StreamingWriter(uri, shape=shape)
        writer.__enter__()

        with pytest.raises(RuntimeError, match="already initialized"):
            writer.__enter__()


class TestStreamingWriterBatch:
    """Tests for StreamingWriter.write_batch()."""

    def test_write_batch(self, temp_dir: Path):
        """write_batch() writes multiple volumes at once."""
        uri = str(temp_dir / "streaming_batch")
        shape = (32, 32, 16)

        volumes = [
            (np.random.randn(*shape).astype(np.float32), "vol_001", "sub-001", {}),
            (np.random.randn(*shape).astype(np.float32), "vol_002", "sub-002", {}),
            (np.random.randn(*shape).astype(np.float32), "vol_003", "sub-003", {}),
        ]

        with StreamingWriter(uri, shape=shape) as writer:
            writer.write_batch(volumes)

        vc = VolumeCollection(uri)
        assert len(vc) == 3


class TestStreamingWriterName:
    """Tests for StreamingWriter with collection name."""

    def test_name_stored_in_metadata(self, temp_dir: Path):
        """Collection name is stored in metadata."""
        uri = str(temp_dir / "streaming_named")
        shape = (32, 32, 16)

        with StreamingWriter(uri, shape=shape, name="T1w") as writer:
            data = np.random.randn(*shape).astype(np.float32)
            writer.write_volume(data, obs_id="vol_001", obs_subject_id="sub-001")

        vc = VolumeCollection(uri)
        assert vc.name == "T1w"


# =============================================================================
# RadiObjectWriter Tests
# =============================================================================


class TestRadiObjectWriterBasic:
    """Tests for RadiObjectWriter context manager."""

    def test_creates_radi_object(self, temp_dir: Path):
        """RadiObjectWriter creates a valid RadiObject."""
        uri = str(temp_dir / "radi_writer")
        shape = (32, 32, 16)

        with RadiObjectWriter(uri) as writer:
            obs_meta_df = pd.DataFrame({
                "obs_subject_id": ["sub-001"],
            })
            writer.write_obs_meta(obs_meta_df)

            with writer.add_collection("T1w", shape=shape) as coll_writer:
                data = np.random.randn(*shape).astype(np.float32)
                coll_writer.write_volume(data, obs_id="sub-001_T1w", obs_subject_id="sub-001")

        radi = RadiObject(uri)
        assert len(radi) == 1
        assert radi.n_collections == 1
        assert "T1w" in radi.collection_names

    def test_multiple_collections(self, temp_dir: Path):
        """RadiObjectWriter can create multiple collections."""
        uri = str(temp_dir / "radi_multi_coll")
        shape = (32, 32, 16)

        with RadiObjectWriter(uri) as writer:
            obs_meta_df = pd.DataFrame({
                "obs_subject_id": ["sub-001", "sub-002"],
            })
            writer.write_obs_meta(obs_meta_df)

            with writer.add_collection("T1w", shape=shape) as coll_writer:
                for i, sid in enumerate(["sub-001", "sub-002"]):
                    data = np.random.randn(*shape).astype(np.float32)
                    coll_writer.write_volume(data, obs_id=f"{sid}_T1w", obs_subject_id=sid)

            with writer.add_collection("FLAIR", shape=shape) as coll_writer:
                for i, sid in enumerate(["sub-001", "sub-002"]):
                    data = np.random.randn(*shape).astype(np.float32)
                    coll_writer.write_volume(data, obs_id=f"{sid}_FLAIR", obs_subject_id=sid)

        radi = RadiObject(uri)
        assert len(radi) == 2
        assert radi.n_collections == 2
        assert "T1w" in radi.collection_names
        assert "FLAIR" in radi.collection_names


class TestRadiObjectWriterObsMeta:
    """Tests for RadiObjectWriter.write_obs_meta()."""

    def test_obs_meta_with_attributes(self, temp_dir: Path):
        """write_obs_meta() stores additional attributes."""
        uri = str(temp_dir / "radi_obs_meta_attrs")
        shape = (32, 32, 16)

        obs_meta_schema = {
            "age": np.dtype("int32"),
            "diagnosis": np.dtype("U64"),
        }

        with RadiObjectWriter(uri, obs_meta_schema=obs_meta_schema) as writer:
            obs_meta_df = pd.DataFrame({
                "obs_subject_id": ["sub-001", "sub-002"],
                "age": [45, 52],
                "diagnosis": ["healthy", "tumor"],
            })
            writer.write_obs_meta(obs_meta_df)

            with writer.add_collection("T1w", shape=shape) as coll_writer:
                for sid in ["sub-001", "sub-002"]:
                    data = np.random.randn(*shape).astype(np.float32)
                    coll_writer.write_volume(data, obs_id=f"{sid}_T1w", obs_subject_id=sid)

        radi = RadiObject(uri)
        obs_meta = radi.obs_meta.read()
        assert "age" in obs_meta.columns
        assert "diagnosis" in obs_meta.columns

    def test_obs_meta_required_column(self, temp_dir: Path):
        """write_obs_meta() requires obs_subject_id column."""
        uri = str(temp_dir / "radi_missing_col")

        with pytest.raises(ValueError, match="obs_subject_id"):
            with RadiObjectWriter(uri) as writer:
                bad_df = pd.DataFrame({"name": ["test"]})
                writer.write_obs_meta(bad_df)


class TestRadiObjectWriterValidation:
    """Tests for RadiObjectWriter validation."""

    def test_duplicate_collection_raises(self, temp_dir: Path):
        """Adding duplicate collection name raises ValueError."""
        uri = str(temp_dir / "radi_dup_coll")
        shape = (32, 32, 16)

        with pytest.raises(ValueError, match="already added"):
            with RadiObjectWriter(uri) as writer:
                with writer.add_collection("T1w", shape=shape):
                    pass
                with writer.add_collection("T1w", shape=shape):
                    pass

    def test_context_manager_required(self, temp_dir: Path):
        """Operations without context manager raise RuntimeError."""
        uri = str(temp_dir / "radi_no_ctx")

        writer = RadiObjectWriter(uri)
        obs_meta_df = pd.DataFrame({"obs_subject_id": ["sub-001"]})

        with pytest.raises(RuntimeError, match="not initialized"):
            writer.write_obs_meta(obs_meta_df)


class TestRadiObjectWriterFinalize:
    """Tests for RadiObjectWriter.finalize()."""

    def test_finalize_returns_radi_object(self, temp_dir: Path):
        """finalize() returns the created RadiObject."""
        uri = str(temp_dir / "radi_finalize")
        shape = (32, 32, 16)

        with RadiObjectWriter(uri) as writer:
            obs_meta_df = pd.DataFrame({"obs_subject_id": ["sub-001"]})
            writer.write_obs_meta(obs_meta_df)

            with writer.add_collection("T1w", shape=shape) as coll_writer:
                data = np.random.randn(*shape).astype(np.float32)
                coll_writer.write_volume(data, obs_id="sub-001_T1w", obs_subject_id="sub-001")

            radi = writer.finalize()

        assert isinstance(radi, RadiObject)
        assert len(radi) == 1


class TestRadiObjectWriterCollectionNames:
    """Tests for RadiObjectWriter.collection_names property."""

    def test_collection_names_tracks_added(self, temp_dir: Path):
        """collection_names tracks collections added so far."""
        uri = str(temp_dir / "radi_coll_names")
        shape = (32, 32, 16)

        with RadiObjectWriter(uri) as writer:
            assert writer.collection_names == []

            with writer.add_collection("T1w", shape=shape):
                pass
            assert "T1w" in writer.collection_names

            with writer.add_collection("FLAIR", shape=shape):
                pass
            assert "FLAIR" in writer.collection_names


class TestStreamingIntegration:
    """Integration tests for streaming writers with real data."""

    def test_roundtrip_data_integrity(
        self,
        temp_dir: Path,
        populated_radi_object: RadiObject,
    ):
        """Data written via streaming matches source data."""
        src_radi = populated_radi_object
        dst_uri = str(temp_dir / "streaming_roundtrip")

        # Get source data
        src_vol = src_radi.T1w.iloc[0]
        src_data = src_vol.to_numpy()
        shape = src_data.shape[:3]

        # Write via streaming
        with RadiObjectWriter(dst_uri) as writer:
            obs_meta_df = src_radi.obs_meta.read().head(1)
            writer.write_obs_meta(obs_meta_df)

            with writer.add_collection("T1w", shape=shape) as coll_writer:
                sid = obs_meta_df.iloc[0]["obs_subject_id"]
                coll_writer.write_volume(
                    src_data,
                    obs_id=f"{sid}_T1w",
                    obs_subject_id=sid,
                )

        # Verify data integrity
        dst_radi = RadiObject(dst_uri)
        dst_vol = dst_radi.T1w.iloc[0]
        dst_data = dst_vol.to_numpy()

        np.testing.assert_array_almost_equal(src_data, dst_data)
