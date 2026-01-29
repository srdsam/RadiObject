"""TileDB context configuration for radiology data."""

from __future__ import annotations

from enum import Enum
from typing import Self

import boto3
import tiledb
from pydantic import BaseModel, Field, model_validator


class SliceOrientation(str, Enum):
    """Preferred slicing orientation for tile optimization."""

    AXIAL = "axial"  # X-Y slices, vary Z (most common for neuro)
    SAGITTAL = "sagittal"  # Y-Z slices, vary X
    CORONAL = "coronal"  # X-Z slices, vary Y
    ISOTROPIC = "isotropic"  # Balanced 64³ chunks for 3D ROI


class Compressor(str, Enum):
    """Compression algorithms suited for radiology data."""

    ZSTD = "zstd"  # Good balance of speed and ratio
    LZ4 = "lz4"  # Fast, lower ratio
    GZIP = "gzip"  # High ratio, slower
    NONE = "none"


class TileConfig(BaseModel):
    """Tile dimensions for chunked storage."""

    orientation: SliceOrientation = Field(
        default=SliceOrientation.AXIAL,
        description="Primary slicing orientation for tile optimization",
    )
    x: int | None = Field(default=None, ge=1, description="Tile extent in X (None = auto)")
    y: int | None = Field(default=None, ge=1, description="Tile extent in Y (None = auto)")
    z: int | None = Field(default=None, ge=1, description="Tile extent in Z (None = auto)")
    t: int = Field(default=1, ge=1, description="Tile extent in T dimension")

    def extents_for_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Compute optimal tile extents based on orientation and volume shape."""
        sx, sy, sz = shape[0], shape[1], shape[2]

        match self.orientation:
            case SliceOrientation.AXIAL:
                extents = (self.x or sx, self.y or sy, self.z or 1)
            case SliceOrientation.SAGITTAL:
                extents = (self.x or 1, self.y or sy, self.z or sz)
            case SliceOrientation.CORONAL:
                extents = (self.x or sx, self.y or 1, self.z or sz)
            case SliceOrientation.ISOTROPIC:
                extents = (self.x or 64, self.y or 64, self.z or 64)

        if len(shape) == 4:
            extents = (*extents, self.t)
        return extents


class CompressionConfig(BaseModel):
    """Compression settings for volume data."""

    algorithm: Compressor = Field(
        default=Compressor.ZSTD,
        description="Compression algorithm",
    )
    level: int = Field(
        default=3,
        ge=-1,
        le=22,
        description="Compression level (algorithm-dependent)",
    )

    def as_filter(self) -> tiledb.Filter | None:
        match self.algorithm:
            case Compressor.ZSTD:
                return tiledb.ZstdFilter(level=self.level)
            case Compressor.LZ4:
                return tiledb.LZ4Filter(level=self.level)
            case Compressor.GZIP:
                return tiledb.GzipFilter(level=self.level)
            case Compressor.NONE:
                return None


class IOConfig(BaseModel):
    """I/O and memory settings."""

    memory_budget_mb: int = Field(
        default=1024,
        ge=64,
        description="Memory budget for TileDB operations (MB)",
    )
    concurrency: int = Field(
        default=4,
        ge=1,
        le=64,
        description="Number of parallel I/O threads",
    )


class S3Config(BaseModel):
    """S3/cloud storage settings."""

    region: str = Field(default="us-east-1")
    endpoint: str | None = Field(default=None, description="Custom S3 endpoint")
    use_virtual_addressing: bool = Field(default=True)
    max_parallel_ops: int = Field(default=8, ge=1)
    multipart_part_size_mb: int = Field(default=50, ge=5)


class OrientationConfig(BaseModel):
    """Orientation detection and standardization settings."""

    auto_detect: bool = Field(
        default=True,
        description="Automatically detect orientation from file headers",
    )
    canonical_target: str = Field(
        default="RAS",
        description="Target canonical orientation (RAS, LAS, or LPS)",
    )
    reorient_on_load: bool = Field(
        default=False,
        description="Reorient to canonical when loading (preserves original by default)",
    )
    store_original_affine: bool = Field(
        default=True,
        description="Store original affine in metadata when reorienting",
    )

    @model_validator(mode="after")
    def validate_canonical_target(self) -> Self:
        """Ensure canonical target is valid."""
        valid_targets = {"RAS", "LAS", "LPS"}
        if self.canonical_target not in valid_targets:
            raise ValueError(
                f"canonical_target must be one of {valid_targets}, got {self.canonical_target}"
            )
        return self


class RadiObjectConfig(BaseModel):
    """Configuration for RadiObject TileDB context."""

    tile: TileConfig = Field(default_factory=TileConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    io: IOConfig = Field(default_factory=IOConfig)
    s3: S3Config = Field(default_factory=S3Config)
    orientation: OrientationConfig = Field(default_factory=OrientationConfig)

    @model_validator(mode="after")
    def validate_compression_level(self) -> Self:
        """Ensure compression level is valid for the algorithm."""
        max_levels = {
            Compressor.ZSTD: 22,
            Compressor.LZ4: 16,
            Compressor.GZIP: 9,
            Compressor.NONE: 0,
        }
        max_level = max_levels[self.compression.algorithm]
        if self.compression.level > max_level:
            self.compression.level = max_level
        return self

    def to_tiledb_config(self, include_s3_credentials: bool = False) -> tiledb.Config:
        """Convert to TileDB Config object.

        Args:
            include_s3_credentials: If True, fetch AWS credentials from boto3.
                This is off by default to avoid expensive/failing credential
                lookups for local-only operations.
        """
        cfg = tiledb.Config()

        # Memory settings
        cfg["sm.memory_budget"] = str(self.io.memory_budget_mb * 1024 * 1024)
        cfg["sm.compute_concurrency_level"] = str(self.io.concurrency)
        cfg["sm.io_concurrency_level"] = str(self.io.concurrency)

        # S3 settings (configuration only, no credential lookup)
        cfg["vfs.s3.region"] = self.s3.region
        cfg["vfs.s3.use_virtual_addressing"] = (
            "true" if self.s3.use_virtual_addressing else "false"
        )
        cfg["vfs.s3.max_parallel_ops"] = str(self.s3.max_parallel_ops)
        cfg["vfs.s3.multipart_part_size"] = str(
            self.s3.multipart_part_size_mb * 1024 * 1024
        )
        if self.s3.endpoint:
            cfg["vfs.s3.endpoint_override"] = self.s3.endpoint

        # Fetch AWS credentials only when explicitly requested
        if include_s3_credentials:
            self._add_s3_credentials(cfg)

        return cfg

    def _add_s3_credentials(self, cfg: tiledb.Config) -> None:
        """Add AWS credentials to config from boto3 session."""
        try:
            session = boto3.Session()
            credentials = session.get_credentials()
            if credentials:
                frozen = credentials.get_frozen_credentials()
                cfg["vfs.s3.aws_access_key_id"] = frozen.access_key
                cfg["vfs.s3.aws_secret_access_key"] = frozen.secret_key
                if frozen.token:
                    cfg["vfs.s3.aws_session_token"] = frozen.token
        except Exception:
            pass  # AWS credentials unavailable; S3 operations will fail

    def to_tiledb_ctx(self, include_s3_credentials: bool = False) -> tiledb.Ctx:
        """Convert to TileDB Ctx object.

        Args:
            include_s3_credentials: If True, fetch AWS credentials from boto3.
        """
        return tiledb.Ctx(self.to_tiledb_config(include_s3_credentials))


# Global mutable configuration
_config: RadiObjectConfig = RadiObjectConfig()
_ctx: tiledb.Ctx | None = None


def get_config() -> RadiObjectConfig:
    """Get the current global configuration."""
    return _config


def ctx() -> tiledb.Ctx:
    """Get the global TileDB context (lazily built from config)."""
    global _ctx
    if _ctx is None:
        _ctx = _config.to_tiledb_ctx()
    return _ctx


def configure(**kwargs) -> None:
    """Convenience function to update config fields.

    Example:
        configure(tile=TileConfig(x=128, y=128, z=32))
        configure(compression=CompressionConfig(algorithm=Compressor.LZ4))
    """
    global _config, _ctx
    _config = _config.model_copy(update=kwargs)
    _ctx = None
