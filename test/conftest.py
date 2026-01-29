"""Pytest fixtures for RadiObject test data."""

from __future__ import annotations

import os
import tempfile
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import tiledb

from data import get_test_data_path
from data.sync import get_manifest
from src.volume import Volume
from src.volume_collection import VolumeCollection

if TYPE_CHECKING:
    from src.radi_object import RadiObject


# ----- Constants -----

DATA_DIR = get_test_data_path()
NIFTI_DIR = DATA_DIR / "nifti" / "msd_brain_tumour"
DICOM_DIR = DATA_DIR / "dicom" / "nsclc_radiomics"


# ----- S3 Configuration -----

S3_TEST_BUCKET = "souzy-scratch"
S3_TEST_PREFIX = "radiobject-tests"
S3_TEST_REGION = "us-east-2"

_s3_ctx: tiledb.Ctx | None = None


def _get_s3_ctx() -> tiledb.Ctx | None:
    """Get TileDB context configured for S3 access, supporting SSO."""
    global _s3_ctx
    if _s3_ctx is not None:
        return _s3_ctx

    import subprocess

    try:
        result = subprocess.run(
            ["aws", "configure", "export-credentials", "--format", "env"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line.startswith("export "):
                    line = line[7:]
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        return None

    cfg = tiledb.Config()
    cfg["vfs.s3.aws_access_key_id"] = os.environ["AWS_ACCESS_KEY_ID"]
    cfg["vfs.s3.aws_secret_access_key"] = os.environ["AWS_SECRET_ACCESS_KEY"]
    if os.environ.get("AWS_SESSION_TOKEN"):
        cfg["vfs.s3.aws_session_token"] = os.environ["AWS_SESSION_TOKEN"]
    cfg["vfs.s3.region"] = S3_TEST_REGION
    cfg["vfs.s3.endpoint_override"] = f"s3.{S3_TEST_REGION}.amazonaws.com"
    cfg["vfs.s3.scheme"] = "https"
    cfg["vfs.s3.use_virtual_addressing"] = "false"

    _s3_ctx = tiledb.Ctx(cfg)
    return _s3_ctx


def _s3_bucket_accessible(bucket: str) -> bool:
    """Check if S3 bucket is accessible via TileDB VFS."""
    ctx = _get_s3_ctx()
    if ctx is None:
        return False
    try:
        vfs = tiledb.VFS(ctx=ctx)
        test_uri = f"s3://{bucket}/{S3_TEST_PREFIX}/.access_test_{uuid.uuid4().hex[:8]}"
        vfs.touch(test_uri)
        vfs.remove_file(test_uri)
        return True
    except Exception:
        return False


def _delete_s3_uri(uri: str) -> None:
    """Recursively delete a TileDB group/array at the given S3 URI."""
    ctx = _get_s3_ctx()
    vfs = tiledb.VFS(ctx=ctx)
    if vfs.is_dir(uri):
        vfs.remove_dir(uri)


StorageBackend = Literal["local", "s3"]


# ----- Helper Functions -----

_manifest_cache: dict[str, list[dict]] = {}


def _get_manifest(data_type: Literal["nifti", "dicom"]) -> list[dict]:
    """Load manifest (cached). Skips test if not found."""
    if data_type in _manifest_cache:
        return _manifest_cache[data_type]

    try:
        manifest = get_manifest(data_type)
        _manifest_cache[data_type] = manifest
        return manifest
    except FileNotFoundError:
        pytest.skip(
            f"{data_type.upper()} manifest not found. "
            "Run: python -c \"from data import sync_test_data; sync_test_data()\""
        )


def _load_nifti_volumes(
    temp_dir: Path,
    manifest: list[dict],
    num: int = 3,
    channel: int = 0,
    modality: str = "flair",
) -> list[tuple[str, Volume]]:
    """Load NIfTI files as Volume objects."""
    volumes = []
    for i in range(min(num, len(manifest))):
        entry = manifest[i]
        sample_id = entry["sample_id"]
        img_path = DATA_DIR / entry["image_path"]
        img = nib.load(img_path)
        data = np.asarray(img.dataobj, dtype=np.float32)

        if data.ndim == 4:
            data = data[..., channel]

        obs_id = f"{sample_id}_{modality}"
        vol_uri = str(temp_dir / f"vol_{i}")
        vol = Volume.from_numpy(vol_uri, data)
        vol.set_obs_id(obs_id)
        volumes.append((obs_id, vol))

    return volumes


def _build_volume_collections(
    temp_dir: Path,
    manifest: list[dict],
    modalities: list[str],
    uri_prefix: str = "",
    ctx: tiledb.Ctx | None = None,
) -> dict[str, VolumeCollection]:
    """Build VolumeCollections from real NIfTI data."""
    subject_ids = [entry["sample_id"] for entry in manifest[:3]]
    collections: dict[str, VolumeCollection] = {}

    for mod_idx, modality in enumerate(modalities):
        volumes = []
        obs_data_rows = []

        for subj_idx, subj_id in enumerate(subject_ids):
            img_path = DATA_DIR / manifest[subj_idx]["image_path"]
            img = nib.load(img_path)
            data = np.asarray(img.dataobj, dtype=np.float32)

            if data.ndim == 4:
                data = data[..., mod_idx]

            obs_id = f"{subj_id}_{modality}"
            vol_uri = str(temp_dir / f"{modality}_vol_{subj_idx}")
            vol = Volume.from_numpy(vol_uri, data)
            vol.set_obs_id(obs_id)
            volumes.append((obs_id, vol))
            obs_data_rows.append({
                "obs_id": obs_id,
                "obs_subject_id": subj_id,
            })

        obs_df = pd.DataFrame(obs_data_rows)
        vc_uri = f"{uri_prefix}/vc_{modality}" if uri_prefix else str(temp_dir / f"vc_{modality}")
        vc = VolumeCollection._from_volumes(vc_uri, volumes, obs_data=obs_df, ctx=ctx)
        collections[modality] = vc

    return collections


# ----- Core Fixtures -----

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Temporary directory for TileDB arrays (function-scoped)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="module")
def temp_dir_module() -> Generator[Path, None, None]:
    """Temporary directory for TileDB arrays (module-scoped, shared across tests)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def volume_uri(temp_dir: Path) -> str:
    """URI for a test Volume."""
    return str(temp_dir / "test_volume")


@pytest.fixture
def collection_uri(temp_dir: Path) -> str:
    """URI for a test VolumeCollection."""
    return str(temp_dir / "test_collection")


@pytest.fixture
def radi_object_uri(temp_dir: Path) -> str:
    """URI for a test RadiObject."""
    return str(temp_dir / "test_radi_object")


@pytest.fixture
def collection_shape() -> tuple[int, int, int]:
    """Standard shape for collection tests (real BraTS data shape)."""
    return (240, 240, 155)


@pytest.fixture
def custom_tiledb_ctx() -> tiledb.Ctx:
    """Custom TileDB context for testing."""
    cfg = tiledb.Config()
    cfg["sm.memory_budget"] = str(512 * 1024 * 1024)
    return tiledb.Ctx(cfg)


# ----- Manifest Fixtures -----

@pytest.fixture
def nifti_manifest() -> list[dict]:
    """Load NIfTI manifest.json."""
    return _get_manifest("nifti")


@pytest.fixture
def dicom_manifest() -> list[dict]:
    """Load DICOM manifest.json."""
    return _get_manifest("dicom")


# ----- Path Fixtures -----

@pytest.fixture
def sample_nifti_image(nifti_manifest: list[dict]) -> Path:
    """Path to first NIfTI image sample."""
    if not nifti_manifest:
        pytest.skip("No NIfTI samples available")
    return DATA_DIR / nifti_manifest[0]["image_path"]


@pytest.fixture
def sample_nifti_label(nifti_manifest: list[dict]) -> Path | None:
    """Path to first NIfTI label sample (may be None)."""
    if not nifti_manifest:
        pytest.skip("No NIfTI samples available")
    label_path = nifti_manifest[0].get("label_path")
    return DATA_DIR / label_path if label_path else None


@pytest.fixture
def sample_dicom_series(dicom_manifest: list[dict]) -> Path:
    """Path to first DICOM series directory."""
    if not dicom_manifest:
        pytest.skip("No DICOM samples available")
    return DATA_DIR / dicom_manifest[0]["image_dir"]


@pytest.fixture
def nifti_4d_path(nifti_manifest: list[dict]) -> Path:
    """Path to first MSD BraTS 4D image (240x240x155x4)."""
    return DATA_DIR / nifti_manifest[0]["image_path"]


@pytest.fixture
def nifti_3d_path(nifti_manifest: list[dict]) -> Path:
    """Path to first MSD BraTS 3D label (segmentation mask)."""
    return DATA_DIR / nifti_manifest[0]["label_path"]


# ----- Array Fixtures -----

@pytest.fixture
def array_4d(nifti_4d_path: Path) -> np.ndarray:
    """Load first MSD BraTS 4D image as numpy array."""
    img = nib.load(nifti_4d_path)
    return np.asarray(img.dataobj, dtype=np.float32)


@pytest.fixture
def array_3d(nifti_3d_path: Path) -> np.ndarray:
    """Load first MSD BraTS 3D label as numpy array."""
    img = nib.load(nifti_3d_path)
    return np.asarray(img.dataobj, dtype=np.float32)


# ----- Volume Fixtures -----

@pytest.fixture
def volumes(temp_dir: Path, nifti_manifest: list[dict]) -> list[tuple[str, Volume]]:
    """Create 3 Volumes from BRATS_001-003 (first channel of 4D data)."""
    return _load_nifti_volumes(temp_dir, nifti_manifest, num=3, channel=0, modality="flair")


@pytest.fixture(scope="module")
def volumes_module(temp_dir_module: Path) -> list[tuple[str, Volume]]:
    """Module-scoped: Create 3 Volumes from BRATS_001-003."""
    manifest = _get_manifest("nifti")
    return _load_nifti_volumes(temp_dir_module, manifest, num=3, channel=0, modality="flair")


@pytest.fixture
def populated_collection(temp_dir: Path, volumes: list[tuple[str, Volume]]) -> VolumeCollection:
    """Pre-created VolumeCollection with 3 real volumes."""
    uri = str(temp_dir / "populated_collection")
    return VolumeCollection._from_volumes(uri, volumes)


@pytest.fixture(scope="module")
def populated_collection_module(
    temp_dir_module: Path, volumes_module: list[tuple[str, Volume]]
) -> VolumeCollection:
    """Module-scoped: Pre-created VolumeCollection with 3 real volumes."""
    uri = str(temp_dir_module / "populated_collection")
    return VolumeCollection._from_volumes(uri, volumes_module)


# ----- VolumeCollection Fixtures -----

@pytest.fixture
def volume_collections(temp_dir: Path, nifti_manifest: list[dict]) -> dict[str, VolumeCollection]:
    """Create 4 VolumeCollections (flair, T1w, T1gd, T2w) from first 3 BraTS samples."""
    modalities = ["flair", "T1w", "T1gd", "T2w"]
    return _build_volume_collections(temp_dir, nifti_manifest, modalities)


@pytest.fixture(scope="module")
def volume_collections_module(temp_dir_module: Path) -> dict[str, VolumeCollection]:
    """Module-scoped: Create 4 VolumeCollections from first 3 BraTS samples."""
    manifest = _get_manifest("nifti")
    modalities = ["flair", "T1w", "T1gd", "T2w"]
    return _build_volume_collections(temp_dir_module, manifest, modalities)


# ----- RadiObject Fixtures -----

@pytest.fixture
def populated_radi_object(
    temp_dir: Path,
    volume_collections: dict[str, VolumeCollection],
    nifti_manifest: list[dict],
) -> "RadiObject":
    """Pre-created RadiObject with 3 subjects and 4 modalities from real data."""
    from src.radi_object import RadiObject

    uri = str(temp_dir / "populated_radi_object")
    subject_ids = [entry["sample_id"] for entry in nifti_manifest[:3]]

    obs_meta_df = pd.DataFrame({
        "obs_subject_id": subject_ids,
    })

    return RadiObject._from_volume_collections(
        uri,
        collections=volume_collections,
        obs_meta=obs_meta_df,
    )


@pytest.fixture(scope="module")
def populated_radi_object_module(
    temp_dir_module: Path,
    volume_collections_module: dict[str, VolumeCollection],
) -> "RadiObject":
    """Module-scoped: Pre-created RadiObject with 3 subjects and 4 modalities."""
    from src.radi_object import RadiObject

    manifest = _get_manifest("nifti")
    uri = str(temp_dir_module / "populated_radi_object")
    subject_ids = [entry["sample_id"] for entry in manifest[:3]]

    obs_meta_df = pd.DataFrame({
        "obs_subject_id": subject_ids,
    })

    return RadiObject._from_volume_collections(
        uri,
        collections=volume_collections_module,
        obs_meta=obs_meta_df,
    )


# ----- Storage Backend Parameterized Fixtures -----

@pytest.fixture(params=["local", "s3"])
def storage_backend(request: pytest.FixtureRequest) -> StorageBackend:
    """Parameterized storage backend (local or s3)."""
    backend: StorageBackend = request.param
    if backend == "s3" and not _s3_bucket_accessible(S3_TEST_BUCKET):
        pytest.skip(
            f"S3 bucket '{S3_TEST_BUCKET}' not accessible. "
            "Configure AWS credentials: aws configure"
        )
    return backend


@pytest.fixture
def tiledb_ctx(storage_backend: StorageBackend) -> tiledb.Ctx | None:
    """TileDB context for the current storage backend."""
    if storage_backend == "s3":
        return _get_s3_ctx()
    return None


@pytest.fixture
def test_base_uri(
    storage_backend: StorageBackend,
    temp_dir: Path,
) -> Generator[str, None, None]:
    """Base URI for test data, parameterized by storage backend."""
    if storage_backend == "local":
        yield str(temp_dir)
    else:
        test_id = uuid.uuid4().hex[:8]
        s3_uri = f"s3://{S3_TEST_BUCKET}/{S3_TEST_PREFIX}/{test_id}"
        yield s3_uri
        _delete_s3_uri(s3_uri)


@pytest.fixture
def radi_object_uri_param(test_base_uri: str) -> str:
    """URI for a test RadiObject, parameterized by storage backend."""
    return f"{test_base_uri}/test_radi_object"


@pytest.fixture
def volume_collections_param(
    storage_backend: StorageBackend,
    test_base_uri: str,
    temp_dir: Path,
    nifti_manifest: list[dict],
    tiledb_ctx: tiledb.Ctx | None,
) -> dict[str, VolumeCollection]:
    """Create 2 VolumeCollections parameterized by storage backend."""
    modalities = ["flair", "T1w"]
    return _build_volume_collections(
        temp_dir, nifti_manifest, modalities, uri_prefix=test_base_uri, ctx=tiledb_ctx
    )


@pytest.fixture
def populated_radi_object_param(
    test_base_uri: str,
    volume_collections_param: dict[str, VolumeCollection],
    nifti_manifest: list[dict],
    tiledb_ctx: tiledb.Ctx | None,
) -> "RadiObject":
    """Pre-created RadiObject parameterized by storage backend."""
    from src.radi_object import RadiObject

    uri = f"{test_base_uri}/populated_radi_object"
    subject_ids = [entry["sample_id"] for entry in nifti_manifest[:3]]

    obs_meta_df = pd.DataFrame({"obs_subject_id": subject_ids})

    return RadiObject._from_volume_collections(
        uri,
        collections=volume_collections_param,
        obs_meta=obs_meta_df,
        ctx=tiledb_ctx,
    )


@pytest.fixture
def populated_collection_param(
    storage_backend: StorageBackend,
    test_base_uri: str,
    temp_dir: Path,
    nifti_manifest: list[dict],
    tiledb_ctx: tiledb.Ctx | None,
) -> VolumeCollection:
    """Pre-created VolumeCollection with real data, parameterized by storage backend."""
    volumes = _load_nifti_volumes(temp_dir, nifti_manifest, num=3, channel=0, modality="flair")
    uri = f"{test_base_uri}/populated_collection"
    return VolumeCollection._from_volumes(uri, volumes, ctx=tiledb_ctx)


# ----- Pre-built S3 RadiObject Fixture -----

@pytest.fixture(scope="session")
def prebuilt_radiobject() -> "RadiObject | None":
    """Pre-built RadiObject from S3 for fast tests (3 BRATS subjects).

    Returns None if S3 is not accessible. Tests using this fixture should
    check for None and skip appropriately.
    """
    from data import DATASETS

    uri = DATASETS.get("brats-3subjects")
    if not uri:
        return None

    ctx = _get_s3_ctx()
    if ctx is None:
        return None

    try:
        from src.radi_object import RadiObject
        vfs = tiledb.VFS(ctx=ctx)
        if not vfs.is_dir(uri):
            return None
        return RadiObject(uri, ctx=ctx)
    except Exception:
        return None
