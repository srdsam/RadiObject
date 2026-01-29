"""Thread-safe volume reading utilities."""

from __future__ import annotations

import hashlib
import os
import threading
from typing import TYPE_CHECKING

import numpy as np
import tiledb

from src.parallel import create_worker_ctx
from src.volume import Volume

if TYPE_CHECKING:
    from src.volume_collection import VolumeCollection

_PROCESS_CTX_CACHE: dict[tuple[int, str], tiledb.Ctx] = {}
_CTX_LOCK = threading.Lock()


def _config_hash(config: tiledb.Config | None) -> str:
    """Compute a hash of the config for cache keying."""
    if config is None:
        return "default"
    key_vals = sorted((k, v) for k, v in config.items())
    return hashlib.md5(str(key_vals).encode()).hexdigest()[:12]


class VolumeReader:
    """Thread-safe wrapper for reading volumes from VolumeCollection."""

    def __init__(self, collection: VolumeCollection, ctx: tiledb.Ctx | None = None):
        self._uri = collection.uri
        self._base_ctx_config = ctx.config() if ctx else None
        self._config_hash = _config_hash(self._base_ctx_config)
        self._shape = collection.shape
        self._obs_ids = list(collection.obs_ids)

    def _get_ctx(self) -> tiledb.Ctx:
        """Get process-local TileDB context (safe for multiprocessing)."""
        cache_key = (os.getpid(), self._config_hash)
        with _CTX_LOCK:
            if cache_key not in _PROCESS_CTX_CACHE:
                if self._base_ctx_config:
                    _PROCESS_CTX_CACHE[cache_key] = tiledb.Ctx(self._base_ctx_config)
                else:
                    _PROCESS_CTX_CACHE[cache_key] = create_worker_ctx()
            return _PROCESS_CTX_CACHE[cache_key]

    def _get_volume(self, idx: int) -> Volume:
        """Get Volume at index with process-local context."""
        volume_uri = f"{self._uri}/volumes/{idx}"
        return Volume(volume_uri, ctx=self._get_ctx())

    @property
    def shape(self) -> tuple[int, int, int]:
        """Volume dimensions (X, Y, Z)."""
        return self._shape

    def __len__(self) -> int:
        """Number of volumes."""
        return len(self._obs_ids)

    def read_full(self, idx: int) -> np.ndarray:
        """Read entire volume at index."""
        vol = self._get_volume(idx)
        return vol.to_numpy()

    def read_patch(
        self,
        idx: int,
        start: tuple[int, int, int],
        size: tuple[int, int, int],
    ) -> np.ndarray:
        """Read a 3D patch from volume at index."""
        vol = self._get_volume(idx)
        x_start, y_start, z_start = start
        x_size, y_size, z_size = size
        return vol.slice(
            slice(x_start, x_start + x_size),
            slice(y_start, y_start + y_size),
            slice(z_start, z_start + z_size),
        )

    def read_slice(self, idx: int, axis: int, position: int) -> np.ndarray:
        """Read a 2D slice from volume at index."""
        vol = self._get_volume(idx)
        if axis == 0:
            return vol.sagittal(position)
        elif axis == 1:
            return vol.coronal(position)
        elif axis == 2:
            return vol.axial(position)
        else:
            raise ValueError(f"axis must be 0, 1, or 2, got {axis}")

    def get_obs_id(self, idx: int) -> str:
        """Get obs_id at index."""
        return self._obs_ids[idx]
