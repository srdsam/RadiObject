"""RadiObject - TileDB-backed data structure for radiology data at scale."""

from radiobject._types import AttrValue, LabelSource, TransformFn
from radiobject.ctx import (
    CompressionConfig,
    Compressor,
    OrientationConfig,
    ReadConfig,
    S3Config,
    SliceOrientation,
    TileConfig,
    WriteConfig,
    configure,
    radi_cfg,
    radi_reset,
    tdb_cfg,
    tdb_ctx,
)
from radiobject.dataframe import Dataframe
from radiobject.indexing import Index, align
from radiobject.radi_object import RadiObject
from radiobject.stats import CacheStats, S3Stats, TileDBStats
from radiobject.utils import delete_tiledb_uri, uri_exists
from radiobject.volume import Volume
from radiobject.volume_collection import VolumeCollection

__version__ = "0.1.1"
__all__ = [
    # Core classes
    "RadiObject",
    "Volume",
    "VolumeCollection",
    "Dataframe",
    "Index",
    "align",
    # Configuration functions
    "configure",
    "radi_cfg",
    "radi_reset",
    "tdb_ctx",
    "tdb_cfg",
    # Configuration classes
    "WriteConfig",
    "ReadConfig",
    "TileConfig",
    "CompressionConfig",
    "OrientationConfig",
    "S3Config",
    # Enums
    "SliceOrientation",
    "Compressor",
    # Types
    "TransformFn",
    "AttrValue",
    "LabelSource",
    # Stats
    "TileDBStats",
    "CacheStats",
    "S3Stats",
    # Utilities
    "uri_exists",
    "delete_tiledb_uri",
]
