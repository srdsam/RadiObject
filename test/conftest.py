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

from radiobject.data import S3_BUCKET, S3_REGION, get_dataset, get_manifest
from radiobject.volume import Volume
from radiobject.volume_collection import VolumeCollection

if TYPE_CHECKING:
    from radiobject.radi_object import RadiObject


# ----- Dataset Paths (resolved on first access) -----

_nifti_data_dir: Path | None = None
_dicom_data_dir: Path | None = None


def _get_nifti_data_dir() -> Path:
    """Get NIfTI dataset directory, downloading if needed."""
    global _nifti_data_dir
    if _nifti_data_dir is None:
        _nifti_data_dir = get_dataset("msd-brain-tumour")
    return _nifti_data_dir


def _get_dicom_data_dir() -> Path:
    """Get DICOM dataset directory, downloading if needed."""
    global _dicom_data_dir
    if _dicom_data_dir is None:
        _dicom_data_dir = get_dataset("nsclc-radiomics")
    return _dicom_data_dir


# ----- S3 Configuration -----

S3_TEST_BUCKET = S3_BUCKET
S3_TEST_PREFIX = "radiobject-tests"
S3_TEST_REGION = S3_REGION

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


def _get_manifest_cached(dataset: str) -> list[dict]:
    """Load manifest (cached). Skips test if not found."""
    if dataset in _manifest_cache:
        return _manifest_cache[dataset]

    try:
        manifest = get_manifest(dataset)
        _manifest_cache[dataset] = manifest
        return manifest
    except FileNotFoundError:
        pytest.skip(
            f"{dataset} manifest not found. Run: python scripts/download_dataset.py --all-tests"
        )


def _load_nifti_volumes(
    temp_dir: Path,
    manifest: list[dict],
    num: int = 3,
    channel: int = 0,
    modality: str = "flair",
) -> list[tuple[str, Volume]]:
    """Load NIfTI files as Volume objects."""
    data_dir = _get_nifti_data_dir()
    volumes = []
    for i in range(min(num, len(manifest))):
        entry = manifest[i]
        sample_id = entry["sample_id"]
        img_path = data_dir / entry["image_path"]
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
    data_dir = _get_nifti_data_dir()
    subject_ids = [entry["sample_id"] for entry in manifest[:3]]
    collections: dict[str, VolumeCollection] = {}

    for mod_idx, modality in enumerate(modalities):
        volumes = []
        obs_data_rows = []

        for subj_idx, subj_id in enumerate(subject_ids):
            img_path = data_dir / manifest[subj_idx]["image_path"]
            img = nib.load(img_path)
            data = np.asarray(img.dataobj, dtype=np.float32)

            if data.ndim == 4:
                data = data[..., mod_idx]

            obs_id = f"{subj_id}_{modality}"
            vol_uri = str(temp_dir / f"{modality}_vol_{subj_idx}")
            vol = Volume.from_numpy(vol_uri, data)
            vol.set_obs_id(obs_id)
            volumes.append((obs_id, vol))
            obs_data_rows.append(
                {
                    "obs_id": obs_id,
                    "obs_subject_id": subj_id,
                }
            )

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
    return _get_manifest_cached("msd-brain-tumour")


@pytest.fixture
def dicom_manifest() -> list[dict]:
    """Load DICOM manifest.json."""
    return _get_manifest_cached("nsclc-radiomics")


# ----- Path Fixtures -----


@pytest.fixture
def sample_nifti_image(nifti_manifest: list[dict]) -> Path:
    """Path to first NIfTI image sample."""
    if not nifti_manifest:
        pytest.skip("No NIfTI samples available")
    data_dir = _get_nifti_data_dir()
    return data_dir / nifti_manifest[0]["image_path"]


@pytest.fixture
def sample_nifti_label(nifti_manifest: list[dict]) -> Path | None:
    """Path to first NIfTI label sample (may be None)."""
    if not nifti_manifest:
        pytest.skip("No NIfTI samples available")
    label_path = nifti_manifest[0].get("label_path")
    if label_path:
        data_dir = _get_nifti_data_dir()
        return data_dir / label_path
    return None


@pytest.fixture
def sample_dicom_series(dicom_manifest: list[dict]) -> Path:
    """Path to first DICOM series directory."""
    if not dicom_manifest:
        pytest.skip("No DICOM samples available")
    data_dir = _get_dicom_data_dir()
    return data_dir / dicom_manifest[0]["image_dir"]


@pytest.fixture
def nifti_4d_path(nifti_manifest: list[dict]) -> Path:
    """Path to first MSD BraTS 4D image (240x240x155x4)."""
    data_dir = _get_nifti_data_dir()
    return data_dir / nifti_manifest[0]["image_path"]


@pytest.fixture
def nifti_3d_path(nifti_manifest: list[dict]) -> Path:
    """Path to first MSD BraTS 3D label (segmentation mask)."""
    data_dir = _get_nifti_data_dir()
    return data_dir / nifti_manifest[0]["label_path"]


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
    manifest = _get_manifest_cached("msd-brain-tumour")
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
    manifest = _get_manifest_cached("msd-brain-tumour")
    modalities = ["flair", "T1w", "T1gd", "T2w"]
    return _build_volume_collections(temp_dir_module, manifest, modalities)


# ----- RadiObject Fixtures -----


def _create_radi_object(
    uri: str,
    collections: dict[str, VolumeCollection],
    manifest: list[dict],
    ctx: tiledb.Ctx | None = None,
) -> "RadiObject":
    """Factory function for creating RadiObject instances."""
    from radiobject.radi_object import RadiObject

    subject_ids = [entry["sample_id"] for entry in manifest[:3]]
    obs_meta_df = pd.DataFrame({"obs_subject_id": subject_ids})
    return RadiObject._from_volume_collections(
        uri,
        collections=collections,
        obs_meta=obs_meta_df,
        ctx=ctx,
    )


@pytest.fixture
def populated_radi_object(
    temp_dir: Path,
    volume_collections: dict[str, VolumeCollection],
    nifti_manifest: list[dict],
) -> "RadiObject":
    """Pre-created RadiObject with 3 subjects and 4 modalities from real data."""
    uri = str(temp_dir / "populated_radi_object")
    return _create_radi_object(uri, volume_collections, nifti_manifest)


@pytest.fixture(scope="module")
def populated_radi_object_module(
    temp_dir_module: Path,
    volume_collections_module: dict[str, VolumeCollection],
) -> "RadiObject":
    """Module-scoped: Pre-created RadiObject with 3 subjects and 4 modalities."""
    manifest = _get_manifest_cached("msd-brain-tumour")
    uri = str(temp_dir_module / "populated_radi_object")
    return _create_radi_object(uri, volume_collections_module, manifest)


# ----- S3-Only Fixtures -----
# These fixtures skip at fixture level if S3 is not available,
# preventing fixture setup pollution when running the full test suite.

_S3_AVAILABLE: bool | None = None


def _is_s3_available() -> bool:
    """Check S3 availability (cached to avoid repeated subprocess calls)."""
    global _S3_AVAILABLE
    if _S3_AVAILABLE is None:
        _S3_AVAILABLE = _s3_bucket_accessible(S3_TEST_BUCKET)
    return _S3_AVAILABLE


@pytest.fixture
def s3_tiledb_ctx() -> tiledb.Ctx:
    """TileDB context for S3 tests. Skips if S3 unavailable."""
    if not _is_s3_available():
        pytest.skip(
            f"S3 bucket '{S3_TEST_BUCKET}' not accessible. Configure AWS credentials: aws configure"
        )
    ctx = _get_s3_ctx()
    if ctx is None:
        pytest.skip("Failed to create S3 TileDB context")
    return ctx


@pytest.fixture
def s3_test_base_uri(
    s3_tiledb_ctx: tiledb.Ctx,
) -> Generator[str, None, None]:
    """Base URI for S3 test data. Skips if S3 unavailable."""
    test_id = uuid.uuid4().hex[:8]
    s3_uri = f"s3://{S3_TEST_BUCKET}/{S3_TEST_PREFIX}/{test_id}"
    yield s3_uri
    _delete_s3_uri(s3_uri)


@pytest.fixture
def s3_radi_object_uri(s3_test_base_uri: str) -> str:
    """URI for a test RadiObject on S3."""
    return f"{s3_test_base_uri}/test_radi_object"


@pytest.fixture
def s3_volume_collections(
    s3_test_base_uri: str,
    temp_dir: Path,
    nifti_manifest: list[dict],
    s3_tiledb_ctx: tiledb.Ctx,
) -> dict[str, VolumeCollection]:
    """Create 2 VolumeCollections on S3."""
    modalities = ["flair", "T1w"]
    return _build_volume_collections(
        temp_dir, nifti_manifest, modalities, uri_prefix=s3_test_base_uri, ctx=s3_tiledb_ctx
    )


@pytest.fixture
def s3_populated_radi_object(
    s3_test_base_uri: str,
    s3_volume_collections: dict[str, VolumeCollection],
    nifti_manifest: list[dict],
    s3_tiledb_ctx: tiledb.Ctx,
) -> "RadiObject":
    """Pre-created RadiObject on S3."""
    uri = f"{s3_test_base_uri}/populated_radi_object"
    return _create_radi_object(uri, s3_volume_collections, nifti_manifest, ctx=s3_tiledb_ctx)


# ----- Shared Synthetic Fixtures -----


@pytest.fixture
def synthetic_nifti_files(temp_dir: Path) -> list[tuple[Path, str]]:
    """Create synthetic NIfTI files (3 subjects x 2 series types)."""
    niftis = []
    shape = (32, 32, 16)
    affine = np.eye(4)
    affine[0, 0] = 1.0
    affine[1, 1] = 1.0
    affine[2, 2] = 2.0

    for subject_id in ["sub-01", "sub-02", "sub-03"]:
        for series_type in ["T1w", "FLAIR"]:
            data = np.random.rand(*shape).astype(np.float32)
            img = nib.Nifti1Image(data, affine)
            img.header.set_qform(affine, code=1)
            img.header.set_sform(affine, code=1)

            filename = f"{subject_id}_{series_type}.nii.gz"
            filepath = temp_dir / filename
            nib.save(img, filepath)
            niftis.append((filepath, subject_id))

    return niftis


@pytest.fixture
def synthetic_nifti_images(
    synthetic_nifti_files: list[tuple[Path, str]],
) -> dict[str, list[tuple[Path, str]]]:
    """Group synthetic_nifti_files by series type into images dict format."""
    images: dict[str, list[tuple[Path, str]]] = {"T1w": [], "FLAIR": []}
    for path, subject_id in synthetic_nifti_files:
        if "T1w" in path.name:
            images["T1w"].append((path, subject_id))
        elif "FLAIR" in path.name:
            images["FLAIR"].append((path, subject_id))
    return images


@pytest.fixture
def small_volume(temp_dir: Path) -> Volume:
    """A small (32^3) Volume for lightweight tests."""
    uri = str(temp_dir / "small_vol")
    return Volume.from_numpy(uri, np.random.rand(32, 32, 32).astype(np.float32))


def _create_test_nifti(
    temp_dir: Path,
    nifti_manifest: list[dict],
    filename: str,
    entry_index: int = 0,
) -> tuple[Path, np.ndarray]:
    """Load a real NIfTI from manifest and save a 3D copy to temp_dir."""
    data_dir = _get_nifti_data_dir()
    src_path = data_dir / nifti_manifest[entry_index]["image_path"]
    img = nib.load(src_path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]

    new_path = temp_dir / filename
    nib.save(nib.Nifti1Image(data, img.affine), new_path)
    return new_path, data


@pytest.fixture
def create_test_nifti(temp_dir: Path, nifti_manifest: list[dict]):
    """Fixture returning a factory to create test NIfTIs from manifest data."""

    def _factory(filename: str, entry_index: int = 0) -> tuple[Path, np.ndarray]:
        return _create_test_nifti(temp_dir, nifti_manifest, filename, entry_index)

    return _factory


# ----- Auto-Marker Infrastructure -----


_DATA_FIXTURES = {
    "nifti_manifest",
    "dicom_manifest",
    "sample_nifti_image",
    "sample_nifti_label",
    "sample_dicom_series",
    "nifti_4d_path",
    "nifti_3d_path",
    "array_3d",
    "array_4d",
    "volumes",
    "volumes_module",
    "populated_collection",
    "populated_collection_module",
    "volume_collections",
    "volume_collections_module",
    "populated_radi_object",
    "populated_radi_object_module",
}

_S3_FIXTURES = {
    "s3_tiledb_ctx",
    "s3_test_base_uri",
    "s3_radi_object_uri",
    "s3_volume_collections",
    "s3_populated_radi_object",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-apply markers based on fixture dependencies."""
    for item in items:
        fixturenames = set(getattr(item, "fixturenames", []))
        if fixturenames & _DATA_FIXTURES:
            item.add_marker(pytest.mark.requires_data)
        if fixturenames & _S3_FIXTURES:
            item.add_marker(pytest.mark.s3)
