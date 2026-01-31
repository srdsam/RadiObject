# RadiObject

**What?** A TileDB-backed data structure to represent radiology data.

**Why?** The two standard formats for radiology data (NIfTI and DICOM) are challenging to use for analysis at scale. They must be read from local disk and don't fully support partial reads. 

**Why TileDB?** TileDB supports cloud-native object stores, partial reads, and a hierarchical format (for organising multiple volumes into an atlas). *The limitation being, transform/filter operations are less smooth UX given that the in-memory TileDB data is tied directly to the on-disk TileDB data.*

## Design

Inspired by the [SOMA specification](https://github.com/single-cell-data/SOMA/blob/main/abstract_specification.md), RadiObject is a hierarchical composition of entities aligned on shared indexes.

### TileDB Structure

The hierarchy maps to TileDB primitives (Groups and Arrays) with explicit dimensions and attributes:

```
RadiObject (TileDB Group)
├── metadata: { subject_count, n_collections }
│
├── obs_meta (Sparse Array) ────────────────────────────────────────────────┐
│   │                                                                        │
│   │  DIMENSIONS (Indexes):                                                 │
│   │    dim[0]: obs_subject_id  (ascii)  <- Primary subject identifier      │
│   │    dim[1]: obs_id          (ascii)  <- Observation identifier          │
│   │                                                                        │
│   │  ATTRIBUTES (Data):                                                    │
│   │    User-defined columns (age, sex, diagnosis, labels, etc.)            │
│   └────────────────────────────────────────────────────────────────────────┘
│
└── collections (TileDB Group)
    │
    ├── T1w (VolumeCollection - TileDB Group)
    │   ├── metadata: { n_volumes, name, [x_dim, y_dim, z_dim]? }
    │   │              └── shape fields only present if collection is uniform
    │   │
    │   ├── obs (Sparse Array) ─────────────────────────────────────────────┐
    │   │   │                                                                │
    │   │   │  DIMENSIONS (Indexes):                                         │
    │   │   │    dim[0]: obs_subject_id  (ascii)  <- FK to RadiObject.obs_meta
    │   │   │    dim[1]: obs_id          (ascii)  <- Unique volume ID        │
    │   │   │                                                                │
    │   │   │  ATTRIBUTES (Data):                                            │
    │   │   │    series_type      <- "T1w", "FLAIR", etc.                    │
    │   │   │    voxel_spacing    <- "(1.0, 1.0, 1.0)" per-volume spacing    │
    │   │   │    dimensions       <- "(240, 240, 155)" per-volume shape      │
    │   │   │    axcodes, affine_json, datatype, bitpix, ...                 │
    │   │   └────────────────────────────────────────────────────────────────┘
    │   │
    │   └── volumes (TileDB Group)
    │       │
    │       ├── 0 (Volume - Dense Array) ───────────────────────────────────┐
    │       │   │  Each volume has its OWN shape (can differ across volumes) │
    │       │   │                                                            │
    │       │   │  DIMENSIONS (Indexes):                                     │
    │       │   │    dim[0]: x  (int32, 0..X-1)                              │
    │       │   │    dim[1]: y  (int32, 0..Y-1)                              │
    │       │   │    dim[2]: z  (int32, 0..Z-1)                              │
    │       │   │   [dim[3]: t  (int32, 0..T-1)]  <- 4D volumes only         │
    │       │   │                                                            │
    │       │   │  ATTRIBUTES (Data):                                        │
    │       │   │    voxels  (float32/int16)  <- Intensity values            │
    │       │   │                                                            │
    │       │   │  METADATA: obs_id, slice_orientation, orientation info     │
    │       │   └────────────────────────────────────────────────────────────┘
    │       │
    │       ├── 1 (Volume) ...  <- may have different shape than Volume[0]
    │       └── N (Volume) ...
    │
    ├── FLAIR (VolumeCollection) ...
    └── seg (VolumeCollection) ...
```

### Key Relationships

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INDEX RELATIONSHIPS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  RadiObject.obs_meta                     VolumeCollection.obs                │
│  ┌────────────────────┐                  ┌─────────────────────┐             │
│  │ obs_subject_id (D) │<────────────────>│ obs_subject_id (D)  │  FK         │
│  │ obs_id         (D) │                  │ obs_id          (D) │             │
│  │ age            (A) │                  │ series_type     (A) │             │
│  │ sex            (A) │                  │ voxel_spacing   (A) │             │
│  │ tumor_grade    (A) │                  │ dimensions      (A) │             │
│  └────────────────────┘                  └─────────────────────┘             │
│         │                                          │                         │
│         │ 1:N relationship                         │ 1:1 relationship        │
│         │ (one subject, many obs)                  │                         │
│         v                                          v                         │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                        Volume (Dense Array)                   │           │
│  │  ┌──────────────────────────────────────────────────────┐    │           │
│  │  │  DIMENSIONS: x, y, z [, t]                           │    │           │
│  │  │  ATTRIBUTES: voxels                                  │    │           │
│  │  │  METADATA:   obs_id (links to obs dataframe)         │    │           │
│  │  └──────────────────────────────────────────────────────┘    │           │
│  └──────────────────────────────────────────────────────────────┘           │
│                                                                              │
│  (D) = Dimension    (A) = Attribute                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Component | TileDB Type | Dimensions (Indexes) | Attributes (Data) |
|-----------|-------------|----------------------|-------------------|
| **RadiObject** | Group | — | metadata: subject_count, n_collections |
| **obs_meta** | Sparse Array | `obs_subject_id`, `obs_id` | User-defined (age, labels, etc.) |
| **VolumeCollection** | Group | — | metadata: n_volumes, name, [shape]? |
| **obs** | Sparse Array | `obs_subject_id`, `obs_id` | series_type, voxel_spacing, **dimensions**, etc. |
| **Volume** | Dense Array | `x`, `y`, `z` [, `t`] | `voxels` (intensity values) |

**Note on shapes:**
- **Uniform collections**: `x_dim, y_dim, z_dim` stored in group metadata; `is_uniform=True`
- **Heterogeneous collections**: No shape in group metadata; each volume's shape stored in `obs.dimensions`
- **4D volumes**: Temporal dimension (`t`) is per-volume; not tracked at collection level

## Organisation

Radiology datasets follow the DICOM hierarchy: patients undergo studies containing multiple series (acquisitions), each composed of instances (slices/frames). Volume dimensionality varies by acquisition type—structural scans are 3D while functional and diffusion data are 4D.

Critically, **dimensions are irregular across a dataset**:
- Different scanners produce different matrix sizes
- Protocols vary by site and evolve over time  
- Preprocessing (resampling, registration) changes dimensions

This irregularity makes batch analysis and ML challenging, as most frameworks expect uniform tensor shapes.

`VolumeCollection` addresses this by grouping volumes with consistent X/Y/Z dimensions, while `RadiObject` organizes heterogeneous collections (e.g., T1w at 1mm³, fMRI at 3mm³) under a unified structure.

### Composition

The TileDB entities are a public property of each given entity. This allows direct access to the TileDB object for power users, while persisting a simple API surface. The individual entites are **stateless** - meaning file handles are not cached in memory (to prevent file handle exhaustion).

### Context

RadiObject uses a global configuration pattern to manage TileDB settings. Configuration is organized into **write-time** settings (immutable after array creation) and **read-time** settings (affect all reads):

```python
from radiobject import ctx, configure, WriteConfig, ReadConfig
from radiobject.ctx import TileConfig, SliceOrientation, CompressionConfig

# Use defaults
array = tiledb.open(uri, ctx=ctx())

# Configure write-time settings (affect new arrays only)
configure(write=WriteConfig(
    tile=TileConfig(orientation=SliceOrientation.ISOTROPIC),
    compression=CompressionConfig(level=5)
))

# Configure read-time settings (affect all reads)
configure(read=ReadConfig(memory_budget_mb=2048, concurrency=8))

# Legacy flat API still supported (maps to write.*)
configure(tile=TileConfig(orientation=SliceOrientation.AXIAL))
```

**Defaults:**

| Setting | Path | Default | Description |
|---------|------|---------|-------------|
| Tile orientation | `write.tile.orientation` | `AXIAL` | X-Y slices optimized for neuroimaging |
| Compression | `write.compression` | `ZSTD` level 3 | Balanced speed/ratio |
| Canonical orientation | `write.orientation.canonical_target` | `RAS` | Target coordinate system |
| Reorient on load | `write.orientation.reorient_on_load` | `False` | Preserves original orientation |
| Memory budget | `read.memory_budget_mb` | 1024 MB | TileDB operation limit |
| I/O concurrency | `read.concurrency` | 4 threads | Parallel read/write |

Tile size is auto-computed from array shape based on orientation. For example, `AXIAL` uses full X-Y slices with Z=1, `ISOTROPIC` uses 64³ chunks (in all directions).

## Data Access

RadiObject provides two complementary patterns for filtering and accessing data, each designed for different workflows.

### Exploration Mode (Pandas-like)

Direct indexing with `iloc`, `loc`, `head()`, `tail()`, and `sample()` returns lightweight views for interactive exploration:

```python
radi = RadiObject("s3://bucket/study")

# Direct indexing - immediate results
subset = radi.iloc[0:10]        # First 10 subjects
subset = radi.loc["sub-01"]     # Single subject by ID
subset = radi.head(5)           # Quick preview

# Access collections within view
vol = subset.T1w.iloc[0]        # Get first T1w volume
data = vol.to_numpy()           # Load into memory
```

Views (`RadiObjectView`) provide immediate access to data and feel like working with pandas DataFrames. Use this mode when interactively exploring data, debugging, or building analysis notebooks.

### Pipeline Mode (Lazy Query Builder)

The `query()` method returns a lazy filter builder for ETL pipelines and ML training:

```python
# Lazy filtering - no data loaded until materialized
result = (
    radi.query()
    .filter("age > 40 and diagnosis == 'tumor'")
    .select_collections(["T1w", "FLAIR"])
    .sample(100, seed=42)
    .to_radi_object("s3://bucket/subset", streaming=True)
)

# Streaming iteration for ML training
for batch in radi.query().filter("split == 'train'").iter_batches(batch_size=32):
    train_step(batch.volumes["T1w"], batch.volumes["FLAIR"])
```

Queries (`Query`) accumulate filters without touching data. Explicit materialization methods (`to_radi_object()`, `iter_volumes()`, `count()`) trigger execution. Use this mode for reproducible pipelines, memory-efficient exports, and ML data loading.

### When to Use Each

| Use Case | Mode | Why |
|----------|------|-----|
| Debugging / quick inspection | Exploration | Immediate feedback |
| Jupyter notebooks | Exploration | Interactive feel |
| ETL pipelines | Pipeline | Explicit execution, streaming |
| ML training data | Pipeline | Batched iteration, memory control |
| Subsetting for export | Pipeline | Memory-efficient streaming writes |

Both modes can be combined: start with exploration to understand your data, then use `view.to_query()` to transition to pipeline mode for production.

## Benchmarking

**TL;DR**: RadiObject enables **200-660x faster** partial reads and native S3 access.

![I/O Performance](assets/benchmark/benchmark_hero.png)

### Key Results

| Operation | RadiObject (local) | RadiObject (S3) | MONAI | TorchIO |
|-----------|-------------------|-----------------|-------|---------|
| 2D Slice | **3.8 ms** | **152 ms** | 2502 ms | 777 ms |
| 64³ ROI | **2.2 ms** | **151 ms** | 1229 ms | 760 ms |
| Full Volume | 525 ms | 7135 ms | 1244 ms | 756 ms |

S3 partial reads are **5-16x faster** than local NIfTI frameworks because MONAI/TorchIO must decompress entire volumes.

### Why the Difference?

```
NIfTI:  [full blob] → decompress all → slice
TileDB: [tile][tile] → read 1 tile   → slice
```

### Storage Tradeoff

| Format | Size | Can Partial Read? |
|--------|------|-------------------|
| NIfTI (.nii.gz) | 2.1 GB | No |
| TileDB | 5.7 GB | Yes (local & S3) |

### When to Use Each

| Scenario | Framework |
|----------|-----------|
| Cloud storage (S3/GCS) | **RadiObject** |
| Partial reads (slices, patches) | **RadiObject** |
| Rich augmentation pipelines | TorchIO |
| Existing MONAI workflows | MONAI |

<details>
<summary><strong>Deep Dive</strong> (for performance engineers)</summary>

### Disk Space by Format

| Format | Size | Files | Compression |
|--------|------|-------|-------------|
| NIfTI (.nii.gz) | 2.1 GB | 20 | 3.1x |
| TileDB ISOTROPIC | 5.7 GB | 488 | 1.2x |
| TileDB AXIAL | 6.2 GB | 488 | 1.1x |
| NIfTI (.nii) | 6.7 GB | 20 | 1.0x |
| NumPy (.npy) | 13.4 GB | 20 | 0.5x |

![Disk Space](assets/benchmark/disk_space_comparison.png)

### Memory Pressure (Peak Heap)

| Operation | RadiObject | MONAI | TorchIO |
|-----------|------------|-------|---------|
| Slice extraction | **1 MB** | 912 MB | 304 MB |
| Full volume | 304 MB | 608 MB | 304 MB |
| Random access (10 vols) | 896 MB | 1482 MB | 589 MB |

### CPU Utilization

| Operation | RadiObject | MONAI | TorchIO |
|-----------|------------|-------|---------|
| Full volume (peak) | 75.5% | 37.7% | 37.5% |
| Slice (peak) | 45.4% | 44.4% | 53.8% |

TileDB parallelizes tile decompression across cores.

### Format Overhead (nibabel baseline)

| Format | Time | Notes |
|--------|------|-------|
| .nii.gz | 457 ms | Gzip decompression |
| .nii | 38 ms | Raw mmap |
| .npy | 46 ms | NumPy load |
| TileDB AXIAL | 505 ms | Full read |
| TileDB ISOTROPIC | 120 ms | Parallel tiles |

### Random Access Pattern (shuffled training)

| Framework | Total (10 vols) | Per Volume |
|-----------|-----------------|------------|
| RadiObject (local) | 6264 ms | 626 ms |
| TorchIO | 7538 ms | 754 ms |
| MONAI | 14112 ms | 1411 ms |

### Tiling Strategy

| Strategy | Best For | Config |
|----------|----------|--------|
| **AXIAL** | 2D slice viewing | `SliceOrientation.AXIAL` |
| **ISOTROPIC** | 3D patch extraction | `SliceOrientation.ISOTROPIC` |

### Methodology

- **Dataset**: 20 CT volumes (512×512×~300 voxels, ~7 GB raw)
- **Runs**: 10 measurement, 5 warmup
- **Seeds**: Fixed (42), GC forced between tests
- **Memory**: tracemalloc (heap), psutil (RSS)
- **Full notebook**: `benchmarks/framework_benchmark.ipynb`

</details>

## Using with MONAI/TorchIO

RadiObject focuses on efficient data loading from TileDB/S3 with partial reads. Use MONAI or TorchIO for transforms and augmentation.

### With MONAI Transforms

RadiObjectDataset outputs `{"image": tensor, ...}` - compatible with MONAI dict transforms:

```python
from monai.transforms import Compose, NormalizeIntensityd, RandFlipd
from radiobject.ml import create_training_dataloader

transform = Compose([
    NormalizeIntensityd(keys="image"),
    RandFlipd(keys="image", prob=0.5, spatial_axis=[0, 1, 2]),
])

loader = create_training_dataloader(radi, modalities=["CT"], transform=transform)
```

### With TorchIO Transforms

Use `RadiObjectSubjectsDataset` for TorchIO's Queue-based training:

```python
from radiobject.ml import RadiObjectSubjectsDataset
import torchio as tio

dataset = RadiObjectSubjectsDataset(radi, modalities=["T1w"])
transform = tio.Compose([tio.ZNormalization(), tio.RandomFlip()])
queue = tio.Queue(dataset, max_length=100, samples_per_volume=10)
```

## DevEx

Package management is done with `uv`. Reference the [pyproject](pyproject.toml) to see available package and dependency groups.
