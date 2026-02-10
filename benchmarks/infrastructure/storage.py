"""Dataset preparation and storage format conversion utilities."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np

from .profiler import DiskSpaceResult

if TYPE_CHECKING:
    from radiobject import RadiObject


def get_directory_size_bytes(path: Path) -> tuple[int, int]:
    """Recursively measure directory size. Returns (total_bytes, file_count)."""
    total = 0
    count = 0
    if path.is_file():
        return path.stat().st_size, 1
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
            count += 1
    return total, count


def get_directory_size_mb(path: Path) -> float:
    """Recursively measure directory size in MB."""
    size_bytes, _ = get_directory_size_bytes(path)
    return size_bytes / (1024 * 1024)


def create_uncompressed_nifti(src: Path, dst: Path) -> None:
    """Convert .nii.gz to .nii (uncompressed)."""
    img = nib.load(str(src))
    nib.save(img, str(dst))


def create_numpy_from_nifti(src: Path, dst: Path) -> None:
    """Convert NIfTI to .npy."""
    img = nib.load(str(src))
    data = img.get_fdata()
    np.save(str(dst), data)


def compute_raw_voxel_bytes(nifti_path: Path) -> int:
    """Compute raw voxel data size (uncompressed, no header)."""
    img = nib.load(str(nifti_path))
    shape = img.shape
    dtype = img.get_data_dtype()
    n_voxels = np.prod(shape)
    bytes_per_voxel = np.dtype(dtype).itemsize
    return int(n_voxels * bytes_per_voxel)


def compute_checksum(data: np.ndarray) -> str:
    """Compute MD5 checksum of array data for validation."""
    return hashlib.md5(data.tobytes()).hexdigest()


def measure_disk_space(
    path: Path,
    format_name: str,
    raw_voxel_bytes: int = 0,
) -> DiskSpaceResult:
    """Measure disk space for a storage format."""
    if not path.exists():
        return DiskSpaceResult(format_name, str(path), 0, 0.0, 0)

    size_bytes, n_files = get_directory_size_bytes(path)
    size_mb = size_bytes / (1024 * 1024)
    compression_ratio = raw_voxel_bytes / size_bytes if size_bytes > 0 else 0.0

    return DiskSpaceResult(
        format_name=format_name,
        path=str(path),
        size_bytes=size_bytes,
        size_mb=size_mb,
        n_files=n_files,
        compression_ratio=compression_ratio,
        raw_voxel_bytes=raw_voxel_bytes,
    )


def prepare_nifti_formats(
    source_niftis: list[Path],
    output_dir: Path,
    n_subjects: int = 20,
) -> dict[str, Path]:
    """Create NIfTI and NumPy storage formats from source files.

    Creates three directories:
    - nifti-compressed: Copies of .nii.gz files
    - nifti-uncompressed: Converted .nii files
    - numpy: Converted .npy files
    """
    formats_created = {}

    if not source_niftis:
        print("No source NIfTI files available")
        return formats_created

    # Limit to n_subjects
    source_niftis = source_niftis[:n_subjects]

    # 1. Create NIfTI compressed directory
    nifti_gz_dir = output_dir / "nifti-compressed"
    if not nifti_gz_dir.exists():
        nifti_gz_dir.mkdir(parents=True)
        print(f"Creating NIfTI compressed: {nifti_gz_dir}")
        for src in source_niftis:
            dst = nifti_gz_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
    else:
        print(f"NIfTI compressed exists: {nifti_gz_dir}")
    formats_created["nifti_gz"] = nifti_gz_dir

    # 2. Create uncompressed NIfTI
    nifti_dir = output_dir / "nifti-uncompressed"
    if not nifti_dir.exists():
        nifti_dir.mkdir(parents=True)
        print(f"Creating NIfTI uncompressed: {nifti_dir}")
        for src in source_niftis:
            dst = nifti_dir / src.name.replace(".nii.gz", ".nii")
            if not dst.exists():
                create_uncompressed_nifti(src, dst)
                print(f"  Created: {dst.name}")
    else:
        print(f"NIfTI uncompressed exists: {nifti_dir}")
    formats_created["nifti"] = nifti_dir

    # 3. Create NumPy files
    numpy_dir = output_dir / "numpy"
    if not numpy_dir.exists():
        numpy_dir.mkdir(parents=True)
        print(f"Creating NumPy: {numpy_dir}")
        for src in source_niftis:
            dst = numpy_dir / src.name.replace(".nii.gz", ".npy")
            if not dst.exists():
                create_numpy_from_nifti(src, dst)
                print(f"  Created: {dst.name}")
    else:
        print(f"NumPy exists: {numpy_dir}")
    formats_created["numpy"] = numpy_dir

    return formats_created


def create_tiledb_datasets(
    nifti_dir: Path,
    output_dir: Path,
    tiling_strategies: list[str],
    collection_name: str = "CT",
) -> dict[str, RadiObject]:
    """Create TileDB datasets with different tiling strategies.

    Strategies:
    - "axial": Optimized for axial slices (X, Y, 1)
    - "isotropic": Balanced 3D access (64, 64, 64)
    - "custom": Custom tiling specified via TileConfig
    """
    from radiobject import RadiObject
    from radiobject.ctx import TileConfig, configure
    from radiobject.volume_collection import SliceOrientation

    datasets = {}

    for strategy in tiling_strategies:
        uri = str(output_dir / f"radiobject-{strategy}")

        # Check if already exists
        if Path(uri).exists():
            print(f"Loading existing: {uri}")
            datasets[strategy] = RadiObject(uri)
            continue

        print(f"Creating RadiObject with {strategy} tiling: {uri}")

        # Configure tiling based on strategy
        if strategy == "axial":
            orientation = SliceOrientation.AXIAL
        elif strategy == "isotropic":
            orientation = SliceOrientation.ISOTROPIC
        else:
            orientation = SliceOrientation.ISOTROPIC

        configure(tile=TileConfig(orientation=orientation))

        radi = RadiObject.from_images(
            uri=uri,
            images={collection_name: str(nifti_dir)},
        )

        datasets[strategy] = radi
        print(f"  Created: {len(radi)} subjects")

    return datasets


def verify_tiledb_tiling(uri: str, name: str) -> None:
    """Verify tile extents in a RadiObject."""
    import tiledb

    from radiobject import RadiObject

    radi = RadiObject(uri)
    coll = radi.get_collection(list(radi.collection_names)[0])
    vol = coll.get_volume(0)
    array_uri = vol._tdb_uri

    with tiledb.open(array_uri) as arr:
        schema = arr.schema
        domain = schema.domain
        print(f"{name} tiling:")
        for i in range(domain.ndim):
            dim = domain.dim(i)
            print(f"  {dim.name}: tile={dim.tile}")
