# RadiObject Design

Inspired by the [SOMA specification](https://github.com/single-cell-data/SOMA/blob/main/abstract_specification.md), RadiObject is a hierarchical composition of entities aligned on shared indexes.

## TileDB Structure

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

## Key Relationships

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

## Component Summary

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

## Composition

The TileDB entities are a public property of each given entity. This allows direct access to the TileDB object for power users, while persisting a simple API surface. The individual entities are **stateless** - meaning file handles are not cached in memory (to prevent file handle exhaustion).

## Anatomical Orientation

Medical images encode spatial orientation via an **affine matrix** that maps voxel indices to physical (world) coordinates. RadiObject preserves this information and optionally standardizes orientation during ingestion.

### Orientation Codes

Orientation is described by three letters indicating the direction each axis points:

| Letter | Direction | Axis Points Toward |
|--------|-----------|-------------------|
| R / L | Right / Left | Patient's right or left |
| A / P | Anterior / Posterior | Patient's front or back |
| S / I | Superior / Inferior | Patient's head or feet |

Common conventions:
- **RAS** (neuroimaging standard): X→Right, Y→Anterior, Z→Superior
- **LPS** (DICOM standard): X→Left, Y→Posterior, Z→Superior
- **LAS**: X→Left, Y→Anterior, Z→Superior

```
           Superior (S)
                ↑
                |
    Left (L) ←──┼──→ Right (R)
                |
                ↓
           Inferior (I)

         Anterior (A) = front
         Posterior (P) = back
```

### Orientation Handling

RadiObject detects orientation from file headers and stores it as metadata:

```python
vol = Volume.from_nifti(uri, "scan.nii.gz")

# Access orientation info
info = vol.orientation_info
print(info.axcodes)      # ('R', 'A', 'S')
print(info.source)       # 'nifti_sform' or 'dicom_iop'
print(info.confidence)   # 'header', 'inferred', or 'unknown'
```

### Reorientation Options

By default, RadiObject preserves original orientation. To standardize during ingestion:

```python
from radiobject import configure, WriteConfig
from radiobject.ctx import OrientationConfig

# Reorient all ingested data to RAS
configure(write=WriteConfig(
    orientation=OrientationConfig(
        reorient_on_load=True,
        canonical_target="RAS"  # or "LAS", "LPS"
    )
))

# Or per-volume
vol = Volume.from_nifti(uri, "scan.nii.gz", reorient=True)
```

When reorienting, the original affine is preserved in metadata for provenance.

### Tile Orientation vs Anatomical Orientation

These are distinct concepts:
- **Anatomical orientation** (`orientation_info`): Physical coordinate system (RAS/LPS)
- **Tile orientation** (`tile_orientation`): Storage chunking strategy for I/O performance

```python
vol = Volume.from_nifti(uri, "scan.nii.gz")

# Anatomical: how axes map to patient anatomy
vol.orientation_info.axcodes  # ('R', 'A', 'S')

# Tile: how data is chunked on disk
vol.tile_orientation  # SliceOrientation.AXIAL
```

Choose tile orientation based on access patterns (see [Benchmarks](BENCHMARKS.md)), not anatomical convention.
