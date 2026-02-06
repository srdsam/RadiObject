"""Volume - a single 3D or 4D radiology acquisition backed by TileDB."""

from __future__ import annotations

import builtins
from functools import cached_property
from pathlib import Path

import nibabel as nib
import numpy as np
import tiledb

from radiobject.ctx import SliceOrientation, get_radiobject_config, get_tiledb_ctx
from radiobject.orientation import (
    OrientationInfo,
    detect_dicom_orientation,
    detect_nifti_orientation,
    metadata_to_orientation_info,
    orientation_info_to_metadata,
    reorient_to_canonical,
)

VOXELS_ATTR = "voxels"  # TileDB attribute name for dense volume arrays


class Volume:
    """Single 3D/4D radiology acquisition indexed by obs_id in VolumeCollection."""

    def __init__(self, uri: str, ctx: tiledb.Ctx | None = None):
        self.uri: str = uri
        self._ctx: tiledb.Ctx = ctx  # None means use global
        self._shape: tuple[int, ...] | None = None  # Could be 3 or 4 dimensional

    def __repr__(self) -> str:
        """Concise representation of the Volume."""
        shape_str = "x".join(str(d) for d in self.shape)
        obs_id_str = f", obs_id='{self.obs_id}'" if self.obs_id else ""
        return f"Volume(shape={shape_str}, dtype={self.dtype}{obs_id_str})"

    def __getitem__(self, key: tuple[builtins.slice, ...] | builtins.slice) -> np.ndarray:
        """NumPy-like indexing for partial reads: vol[10:20, :, :]."""
        if isinstance(key, slice):
            key = (key,)
        if not isinstance(key, tuple):
            raise TypeError(f"Index must be slice or tuple of slices, got {type(key)}")

        # Pad with full slices if needed
        key = key + (slice(None),) * (self.ndim - len(key))

        with tiledb.open(self.uri, "r", ctx=self._effective_ctx()) as arr:
            return arr[key][VOXELS_ATTR]

    @property
    def shape(self) -> tuple[int, ...]:
        """Volume dimensions."""
        if self._shape is None:
            self._shape = tuple(
                int(self._schema.domain.dim(i).domain[1] + 1)
                for i in range(self._schema.domain.ndim)
            )
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype:
        """Data type of the volume."""
        return self._schema.attr(0).dtype

    @property
    def tile_orientation(self) -> SliceOrientation | None:
        """Tile slicing strategy used at array creation (None if missing)."""
        try:
            val = self._metadata.get("tile_orientation")
            return SliceOrientation(val) if val else None
        except (KeyError, ValueError):
            return None

    @property
    def orientation_info(self) -> OrientationInfo | None:
        """Anatomical orientation information from TileDB metadata."""
        try:
            return metadata_to_orientation_info(self._metadata)
        except Exception:
            return None

    @property
    def obs_id(self) -> str | None:
        """Observation identifier stored in array metadata."""
        try:
            return self._metadata.get("obs_id")
        except Exception:
            return None

    def axial(self, z: int, t: int | None = None) -> np.ndarray:
        """Get axial slice (X-Y plane at given Z)."""
        return self.slice(
            slice(None), slice(None), slice(z, z + 1), slice(t, t + 1) if t is not None else None
        ).squeeze()

    def sagittal(self, x: int, t: int | None = None) -> np.ndarray:
        """Get sagittal slice (Y-Z plane at given X)."""
        return self.slice(
            slice(x, x + 1), slice(None), slice(None), slice(t, t + 1) if t is not None else None
        ).squeeze()

    def coronal(self, y: int, t: int | None = None) -> np.ndarray:
        """Get coronal slice (X-Z plane at given Y)."""
        return self.slice(
            slice(None), slice(y, y + 1), slice(None), slice(t, t + 1) if t is not None else None
        ).squeeze()

    def slice(
        self,
        x: builtins.slice,
        y: builtins.slice,
        z: builtins.slice,
        t: builtins.slice | None = None,
    ) -> np.ndarray:
        """Partial read using slice objects. Prefer vol[x, y, z] for most cases."""
        with tiledb.open(self.uri, "r", ctx=self._effective_ctx()) as arr:
            if t is not None and self.ndim == 4:
                return arr[x, y, z, t][VOXELS_ATTR]
            return arr[x, y, z][VOXELS_ATTR]

    def to_numpy(self) -> np.ndarray:
        """Read entire volume into memory."""
        with tiledb.open(self.uri, "r", ctx=self._effective_ctx()) as arr:
            return arr[:][VOXELS_ATTR]

    def get_statistics(self, percentiles: list[float] | None = None) -> dict[str, float]:
        """Compute mean, std, min, max, median, and optional percentiles."""
        data = self.to_numpy()
        flat = data.ravel()  # Single flatten operation
        stats = {
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "median": float(np.median(flat)),
        }
        if percentiles:
            pct_values = np.percentile(flat, percentiles)  # Single percentile call
            for p, v in zip(percentiles, pct_values):
                stats[f"p{int(p)}"] = float(v)
        return stats

    def compute_histogram(
        self, bins: int = 256, value_range: tuple[float, float] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute intensity histogram. Returns (counts, bin_edges)."""
        data = self.to_numpy().flatten()
        return np.histogram(data, bins=bins, range=value_range)

    def to_nifti(self, file_path: str | Path, compression: bool = True) -> None:
        """Export to NIfTI with metadata preservation."""
        file_path = Path(file_path)
        if compression and not str(file_path).endswith(".gz"):
            file_path = Path(str(file_path) + ".gz")

        data = self.to_numpy()
        meta = self._metadata

        # Reconstruct affine from stored metadata
        affine = np.eye(4)
        orient_info = self.orientation_info
        if orient_info and orient_info.affine:
            affine = np.array(orient_info.affine)

        img = nib.Nifti1Image(data, affine)
        header = img.header

        # Restore NIfTI header fields if stored
        if "nifti_sform_code" in meta:
            header.set_sform(affine, code=int(meta["nifti_sform_code"]))
        if "nifti_qform_code" in meta:
            header.set_qform(affine, code=int(meta["nifti_qform_code"]))
        if "nifti_scl_slope" in meta:
            header["scl_slope"] = float(meta["nifti_scl_slope"])
        if "nifti_scl_inter" in meta:
            header["scl_inter"] = float(meta["nifti_scl_inter"])
        if "nifti_xyzt_units" in meta:
            header["xyzt_units"] = int(meta["nifti_xyzt_units"])

        nib.save(img, file_path)

    def set_obs_id(self, obs_id: str) -> None:
        """Store observation identifier in array metadata."""
        with tiledb.open(self.uri, "w", ctx=self._effective_ctx()) as arr:
            arr.meta["obs_id"] = obs_id

    def _effective_ctx(self) -> tiledb.Ctx:
        return self._ctx if self._ctx else get_tiledb_ctx()

    @cached_property
    def _schema(self) -> tiledb.ArraySchema:
        """Cached TileDB array schema."""
        return tiledb.ArraySchema.load(self.uri, ctx=self._effective_ctx())

    @cached_property
    def _metadata(self) -> dict:
        """Cached TileDB array metadata - single read for all metadata properties."""
        with tiledb.open(self.uri, "r", ctx=self._effective_ctx()) as arr:
            return dict(arr.meta)

    @classmethod
    def create(
        cls,
        uri: str,
        shape: tuple[int, ...],
        dtype: np.dtype = np.float32,
        ctx: tiledb.Ctx | None = None,
    ) -> Volume:
        """Create an empty Volume with explicit schema."""
        if len(shape) not in (3, 4):
            raise ValueError(f"Shape must be 3D or 4D, got {len(shape)}D")

        effective_ctx = ctx if ctx else get_tiledb_ctx()
        config = get_radiobject_config()

        # Build dimensions with orientation-aware tiling
        dim_names = ["x", "y", "z", "t"][: len(shape)]
        tile_extents = config.write.tile.extents_for_shape(shape)

        dims = [
            tiledb.Dim(
                name=dim_names[i],
                domain=(0, shape[i] - 1),
                tile=min(tile_extents[i], shape[i]),
                dtype=np.int32,
                ctx=effective_ctx,
            )
            for i in range(len(shape))
        ]
        domain = tiledb.Domain(*dims, ctx=effective_ctx)

        # Build attribute with compression
        filters = tiledb.FilterList()
        compression_filter = config.write.compression.as_filter()
        if compression_filter:
            filters.append(compression_filter)

        attr = tiledb.Attr(
            name=VOXELS_ATTR,
            dtype=dtype,
            filters=filters,
            ctx=effective_ctx,
        )

        schema = tiledb.ArraySchema(
            domain=domain,
            attrs=[attr],
            sparse=False,
            ctx=effective_ctx,
        )
        tiledb.Array.create(uri, schema, ctx=effective_ctx)

        # Persist tile orientation metadata
        with tiledb.open(uri, mode="w", ctx=effective_ctx) as arr:
            arr.meta["tile_orientation"] = config.write.tile.orientation.value

        return cls(uri, ctx=ctx)

    @classmethod
    def from_numpy(
        cls,
        uri: str,
        data: np.ndarray,
        ctx: tiledb.Ctx | None = None,
    ) -> Volume:
        """Create Volume from numpy array."""
        vol = cls.create(uri, shape=data.shape, dtype=data.dtype, ctx=ctx)
        effective_ctx = ctx if ctx else get_tiledb_ctx()
        with tiledb.open(uri, mode="w", ctx=effective_ctx) as arr:
            arr[:] = data
        return vol

    @classmethod
    def from_nifti(
        cls,
        uri: str,
        nifti_path: str | Path,
        ctx: tiledb.Ctx | None = None,
        reorient: bool | None = None,
    ) -> Volume:
        """Create a new Volume from a NIfTI file.

        Args:
            uri: TileDB array URI.
            nifti_path: Path to NIfTI file.
            ctx: TileDB context (uses global if None).
            reorient: Reorient to canonical orientation (None uses config default).
        """
        config = get_radiobject_config()
        should_reorient = (
            reorient if reorient is not None else config.write.orientation.reorient_on_load
        )

        img = nib.load(nifti_path)
        data = np.asarray(img.dataobj)
        original_affine = img.affine.copy()

        # Detect orientation from header
        orientation_info = detect_nifti_orientation(img)

        # Reorient if requested
        if should_reorient and not orientation_info.is_canonical:
            data, new_affine = reorient_to_canonical(
                data, original_affine, target=config.write.orientation.canonical_target
            )
            # Update orientation info for reoriented data
            reoriented_img = nib.Nifti1Image(data, new_affine)
            orientation_info = detect_nifti_orientation(reoriented_img)

        # Create volume
        vol = cls.from_numpy(uri, data, ctx=ctx)

        # Store orientation metadata
        effective_ctx = ctx if ctx else get_tiledb_ctx()
        metadata = orientation_info_to_metadata(
            orientation_info,
            original_affine=(
                original_affine
                if should_reorient and config.write.orientation.store_original_affine
                else None
            ),
        )

        # Store NIfTI header fields for roundtrip fidelity
        header = img.header
        metadata["nifti_sform_code"] = str(int(header.get("sform_code", 0)))
        metadata["nifti_qform_code"] = str(int(header.get("qform_code", 0)))

        # Handle scl_slope/scl_inter - nibabel returns nan for unset values
        scl_slope = header.get("scl_slope", 1.0)
        scl_inter = header.get("scl_inter", 0.0)
        metadata["nifti_scl_slope"] = str(1.0 if np.isnan(scl_slope) else float(scl_slope))
        metadata["nifti_scl_inter"] = str(0.0 if np.isnan(scl_inter) else float(scl_inter))

        metadata["nifti_xyzt_units"] = str(int(header.get("xyzt_units", 0)))

        with tiledb.open(uri, mode="w", ctx=effective_ctx) as arr:
            for key, value in metadata.items():
                arr.meta[key] = value

        return vol

    @classmethod
    def from_dicom(
        cls,
        uri: str,
        dicom_dir: str | Path,
        ctx: tiledb.Ctx | None = None,
        reorient: bool | None = None,
        dtype: np.dtype | type | None = None,
    ) -> Volume:
        """Create a new Volume from a DICOM series directory.

        Args:
            uri: TileDB array URI.
            dicom_dir: Path to directory containing DICOM files.
            ctx: TileDB context (uses global if None).
            reorient: Reorient to canonical orientation (None uses config default).
            dtype: Output dtype. None preserves original DICOM dtype (uint16/int16).
                Use np.float32 for backward-compatible behavior.
        """
        import pydicom

        config = get_radiobject_config()
        should_reorient = (
            reorient if reorient is not None else config.write.orientation.reorient_on_load
        )

        dicom_path = Path(dicom_dir)

        # Find DICOM files
        dicom_files = sorted(dicom_path.glob("*.dcm"))
        if not dicom_files:
            dicom_files = sorted(
                f for f in dicom_path.iterdir() if f.is_file() and not f.name.startswith(".")
            )

        if not dicom_files:
            raise ValueError(f"No DICOM files found in {dicom_dir}")

        # Read and sort slices by position
        slices = []
        for dcm_file in dicom_files:
            ds = pydicom.dcmread(dcm_file)
            slices.append(ds)

        # Sort by Instance Number or Image Position Patient
        if hasattr(slices[0], "InstanceNumber"):
            slices.sort(key=lambda x: int(x.InstanceNumber))
        elif hasattr(slices[0], "ImagePositionPatient"):
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

        # Stack pixel arrays into 3D volume
        pixel_arrays = [s.pixel_array for s in slices]
        data = np.stack(pixel_arrays, axis=-1)  # Stack along Z

        # Transpose to (X, Y, Z) from (row, col, slice)
        data = np.transpose(data, (1, 0, 2))

        # Detect orientation
        orientation_info = detect_dicom_orientation(dicom_path)
        original_affine = np.array(orientation_info.affine)

        # Reorient if requested
        if should_reorient and not orientation_info.is_canonical:
            data, new_affine = reorient_to_canonical(
                data, original_affine, target=config.write.orientation.canonical_target
            )
            # Update orientation info
            reoriented_img = nib.Nifti1Image(data, new_affine)
            orientation_info = detect_nifti_orientation(reoriented_img)

        # Create volume with requested dtype (preserve original if None)
        output_data = data if dtype is None else data.astype(dtype)
        vol = cls.from_numpy(uri, output_data, ctx=ctx)

        # Store orientation metadata
        effective_ctx = ctx if ctx else get_tiledb_ctx()
        metadata = orientation_info_to_metadata(
            orientation_info,
            original_affine=(
                original_affine
                if should_reorient and config.write.orientation.store_original_affine
                else None
            ),
        )
        with tiledb.open(uri, mode="w", ctx=effective_ctx) as arr:
            for key, value in metadata.items():
                arr.meta[key] = value

        return vol
