"""RadiObject - TileDB-backed data structure for radiology data at scale."""

from radiobject._types import AttrValue, BatchTransformFn, LabelSource, TransformFn
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
    get_radiobject_config,
    get_tiledb_config,
    get_tiledb_ctx,
    reset_radiobject_config,
)
from radiobject.dataframe import Dataframe
from radiobject.indexing import Index, align
from radiobject.ingest import ImageFormat
from radiobject.query import EagerQuery, LazyQuery
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
    "get_radiobject_config",
    "reset_radiobject_config",
    "get_tiledb_ctx",
    "get_tiledb_config",
    # Configuration classes
    "WriteConfig",
    "ReadConfig",
    "TileConfig",
    "CompressionConfig",
    "OrientationConfig",
    "S3Config",
    # Enums
    "ImageFormat",
    "SliceOrientation",
    "Compressor",
    # Query classes
    "EagerQuery",
    "LazyQuery",
    # Types
    "TransformFn",
    "BatchTransformFn",
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
