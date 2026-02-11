"""TileDB context configuration for radiology data."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Self

import boto3
import tiledb
from pydantic import BaseModel, Field, model_validator

log = logging.getLogger(__name__)


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


class WriteConfig(BaseModel):
    """Settings applied when creating new TileDB arrays (immutable after creation)."""

    tile: TileConfig = Field(default_factory=TileConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    orientation: "OrientationConfig" = Field(default_factory=lambda: OrientationConfig())


class ReadConfig(BaseModel):
    """Settings for reading TileDB arrays."""

    memory_budget_mb: int = Field(
        default=1024,
        ge=64,
        description="Memory budget for TileDB operations (MB)",
    )
    tile_cache_size_mb: int = Field(
        default=512,
        ge=10,
        description="TileDB in-memory LRU tile cache size (MB)",
    )
    concurrency: int = Field(
        default=4,
        ge=1,
        le=64,
        description="Number of TileDB internal I/O threads",
    )
    max_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Max parallel workers for volume I/O operations",
    )


class S3Config(BaseModel):
    """S3/cloud storage settings."""

    region: str = Field(default="us-east-1")
    endpoint: str | None = Field(default=None, description="Custom S3 endpoint")
    use_virtual_addressing: bool = Field(default=True)
    max_parallel_ops: int = Field(default=8, ge=1)
    multipart_part_size_mb: int = Field(default=50, ge=5)
    include_credentials: bool = Field(
        default=True, description="Include AWS credentials from boto3 session"
    )


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
        from radiobject.exceptions import ConfigurationError

        valid_targets = {"RAS", "LAS", "LPS"}
        if self.canonical_target not in valid_targets:
            raise ConfigurationError(
                f"canonical_target must be one of {valid_targets}, got {self.canonical_target}"
            )
        return self


class RadiObjectConfig(BaseModel):
    """Configuration for RadiObject TileDB context."""

    write: WriteConfig = Field(default_factory=WriteConfig)
    read: ReadConfig = Field(default_factory=ReadConfig)
    s3: S3Config = Field(default_factory=S3Config)

    @model_validator(mode="after")
    def validate_compression_level(self) -> Self:
        """Ensure compression level is valid for the algorithm."""
        max_levels = {
            Compressor.ZSTD: 22,
            Compressor.LZ4: 16,
            Compressor.GZIP: 9,
            Compressor.NONE: 0,
        }
        max_level = max_levels[self.write.compression.algorithm]
        if self.write.compression.level > max_level:
            self.write.compression.level = max_level
        return self

    def to_tiledb_config(self, include_s3_credentials: bool = False) -> tiledb.Config:
        """Convert to TileDB Config object.

        Args:
            include_s3_credentials: If True, fetch AWS credentials from boto3.
                This is off by default to avoid expensive/failing credential
                lookups for local-only operations.
        """
        cfg = tiledb.Config()

        # Memory settings from read config
        cfg["sm.memory_budget"] = str(self.read.memory_budget_mb * 1024 * 1024)
        cfg["sm.tile_cache_size"] = str(self.read.tile_cache_size_mb * 1024 * 1024)
        cfg["sm.compute_concurrency_level"] = str(self.read.concurrency)
        cfg["sm.io_concurrency_level"] = str(self.read.concurrency)

        # S3 settings (configuration only, no credential lookup)
        cfg["vfs.s3.region"] = self.s3.region
        cfg["vfs.s3.use_virtual_addressing"] = "true" if self.s3.use_virtual_addressing else "false"
        cfg["vfs.s3.max_parallel_ops"] = str(self.s3.max_parallel_ops)
        cfg["vfs.s3.multipart_part_size"] = str(self.s3.multipart_part_size_mb * 1024 * 1024)
        if self.s3.endpoint:
            cfg["vfs.s3.endpoint_override"] = self.s3.endpoint

        # Fetch AWS credentials if configured or explicitly requested
        if include_s3_credentials or self.s3.include_credentials:
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
        except (boto3.exceptions.Boto3Error, Exception) as e:
            log.warning("AWS credentials unavailable: %s. S3 operations will fail.", e)

    def to_tiledb_ctx(self, include_s3_credentials: bool = False) -> tiledb.Ctx:
        """Convert to TileDB Ctx object.

        Args:
            include_s3_credentials: If True, fetch AWS credentials from boto3.
        """
        return tiledb.Ctx(self.to_tiledb_config(include_s3_credentials))


# Global mutable configuration
_config: RadiObjectConfig = RadiObjectConfig()
_ctx: tiledb.Ctx | None = None


def get_radiobject_config() -> RadiObjectConfig:
    """Get the current RadiObject configuration."""
    return _config


def reset_radiobject_config() -> None:
    """Reset RadiObject configuration to defaults."""
    global _config, _ctx
    _config = RadiObjectConfig()
    _ctx = None


def get_tiledb_ctx() -> tiledb.Ctx:
    """Get the global TileDB context (lazily built from config)."""
    global _ctx
    if _ctx is None:
        _ctx = _config.to_tiledb_ctx()
    return _ctx


def get_tiledb_config() -> tiledb.Config:
    """Get the underlying TileDB Config object for advanced users."""
    return _config.to_tiledb_config()


def configure(
    *,
    write: WriteConfig | None = None,
    read: ReadConfig | None = None,
    s3: S3Config | None = None,
) -> None:
    """Update global configuration.

    Examples:
        Configure write settings (tile strategy, compression):

            configure(write=WriteConfig(
                tile=TileConfig(orientation=SliceOrientation.AXIAL),
                compression=CompressionConfig(algorithm=Compressor.ZSTD, level=5),
            ))

        Configure read settings (memory, concurrency):

            configure(read=ReadConfig(memory_budget_mb=2048, max_workers=8))

        Configure S3 settings:

            configure(s3=S3Config(region="us-west-2"))
    """
    global _config, _ctx

    updates: dict = {}
    if write is not None:
        updates["write"] = write
    if read is not None:
        updates["read"] = read
    if s3 is not None:
        updates["s3"] = s3

    _config = _config.model_copy(update=updates)
    _ctx = None
