# Volume Operations

How to read, inspect, and export individual volumes from a RadiObject. For full method signatures, see the [Volume API](../api/volume.md).

## Accessing a Volume

Volumes are accessed through VolumeCollections via indexing or selection:

```python
radi = RadiObject("./dataset")

# By position
vol = radi.CT.iloc[0]

# By volume ID
vol = radi.CT.loc["BraTS001_CT"]

# By subject ID
vol = radi.CT.sel(subject="BraTS001")
```

For filtering and subsetting before accessing volumes, see [Indexing & Filtering](query-filter-data.md).

## Volume Properties

```python
vol.shape          # (240, 240, 155)
vol.ndim           # 3 (or 4 for temporal data)
vol.dtype          # float32
vol.obs_id         # "BraTS001_CT"
vol.tile_orientation   # SliceOrientation.AXIAL
vol.orientation_info   # OrientationInfo(axcodes=('R', 'A', 'S'), ...)
```

## Partial Reads

RadiObject stores volumes as tiled arrays, enabling reads of individual slices or sub-regions without loading the full volume. This is 200-600x faster than NIfTI for single slices (see [Benchmarks](../reference/benchmarks.md)).

### Named Slice Methods

```python
# 2D axial slice (X-Y plane at given Z index)
axial_slice = vol.axial(z=77)        # shape: (240, 240)

# 2D sagittal slice (Y-Z plane at given X index)
sagittal_slice = vol.sagittal(x=120) # shape: (240, 155)

# 2D coronal slice (X-Z plane at given Y index)
coronal_slice = vol.coronal(y=120)   # shape: (240, 155)
```

For 4D volumes, pass the temporal index:

```python
axial_slice = vol.axial(z=77, t=0)
```

### Arbitrary Sub-Regions

Use `slice()` for arbitrary 3D patches or ROIs:

```python
# 64x64x64 patch
patch = vol.slice(
    x=slice(50, 114),
    y=slice(50, 114),
    z=slice(30, 94),
)  # shape: (64, 64, 64)
```

### NumPy-Style Indexing

Volumes also support direct bracket indexing:

```python
# Equivalent to vol.axial(z=77)
axial_slice = vol[slice(None), slice(None), 77:78].squeeze()

# Arbitrary sub-region
patch = vol[50:114, 50:114, 30:94]
```

### Tiling and Performance

Partial read performance depends on the tile orientation set at ingestion time. Choose the tiling strategy that matches your access pattern — see [Configuration: TileConfig](../reference/configuration.md#tileconfig) for options.

## Full Volume Read

```python
data = vol.to_numpy()  # np.ndarray, shape matches vol.shape
```

## Statistics

Compute summary statistics over the full volume:

```python
stats = vol.get_statistics()
# {'mean': 142.3, 'std': 89.1, 'min': 0.0, 'max': 1024.0, 'median': 118.0}

# With custom percentiles
stats = vol.get_statistics(percentiles=[5, 25, 75, 95])
# Adds 'p5', 'p25', 'p75', 'p95' keys
```

## Histogram

Compute an intensity histogram:

```python
counts, bin_edges = vol.compute_histogram(bins=256)

# Restrict to a value range
counts, bin_edges = vol.compute_histogram(bins=100, value_range=(0.0, 500.0))
```

## NIfTI Export

Export a volume back to NIfTI format with metadata preservation (affine, header fields):

```python
vol.to_nifti("./output/scan.nii.gz")

# Without compression
vol.to_nifti("./output/scan.nii", compression=False)
```

The export preserves the original affine matrix, sform/qform codes, and scaling parameters stored during ingestion.

## Standalone Volume Creation

Create volumes outside of a RadiObject — useful for testing, prototyping, or one-off conversions.

### From NumPy

```python
import numpy as np
from radiobject import Volume

data = np.random.randn(128, 128, 64).astype(np.float32)
vol = Volume.from_numpy("./standalone-volume", data)
```

### From NIfTI

```python
vol = Volume.from_nifti("./tiledb-volume", "scan.nii.gz")

# With reorientation to canonical (RAS)
vol = Volume.from_nifti("./tiledb-volume", "scan.nii.gz", reorient=True)
```

For batch ingestion of many NIfTI files into a RadiObject, use [Ingest Data](ingest-data.md) instead.

## Next Step

**Need to process volumes at scale?** Build transform pipelines with [Lazy Pipelines](lazy-queries.md), or set up ML training with [ML Integration](ml-training.md).

## Related Documentation

- [Indexing & Filtering](query-filter-data.md) - Selecting which volumes to access
- [Volume API](../api/volume.md) - Full method reference
- [Benchmarks](../reference/benchmarks.md) - Partial read performance numbers
