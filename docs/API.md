# API Reference

Full API documentation is auto-generated from docstrings using pdoc.

## Generate Docs Locally

```bash
# Install docs dependencies
uv sync --extra docs

# Generate HTML documentation
uv run pdoc src/radiobject -o docs/api

# View in browser
open docs/api/index.html
```

## Core Classes

| Class | Description |
|-------|-------------|
| `RadiObject` | Top-level container for multi-collection radiology data with subject metadata |
| `RadiObjectView` | Immutable filtered view into a RadiObject |
| `VolumeCollection` | Collection of volumes organized by obs_id, supports heterogeneous shapes |
| `Volume` | Single 3D or 4D radiology acquisition backed by TileDB |
| `Query` | Lazy filter builder for RadiObject with explicit materialization |
| `CollectionQuery` | Lazy filter builder for VolumeCollection |
| `Dataframe` | TileDB-backed DataFrame for obs/obs_meta storage |

## Configuration Classes

| Class | Description |
|-------|-------------|
| `configure()` | Update global configuration |
| `get_config()` | Get current global configuration |
| `ctx()` | Get global TileDB context |
| `WriteConfig` | Settings for creating new TileDB arrays |
| `ReadConfig` | Settings for reading TileDB arrays |
| `TileConfig` | Tile dimensions for chunked storage |
| `CompressionConfig` | Compression settings for volume data |
| `S3Config` | S3/cloud storage settings |
| `OrientationConfig` | Orientation detection and standardization |

## Quick Reference

### RadiObject

```python
# Create from NIfTI files
radi = RadiObject.from_niftis(uri, niftis=[(path, subject_id), ...])
radi = RadiObject.from_niftis(uri, image_dir="./images")

# Create from DICOM series
radi = RadiObject.from_dicoms(uri, dicom_dirs=[(path, subject_id), ...])

# Access collections
radi.T1w                    # Attribute access
radi.collection("T1w")      # Method access
radi.collection_names       # All collection names

# Access subjects
radi.iloc[0]                # By position
radi.loc["sub-01"]          # By obs_subject_id
radi["sub-01"]              # Shorthand for .loc
radi.obs_subject_ids        # All subject IDs

# Filtering
radi.filter("age > 40")     # Query expression
radi.head(10)               # First n subjects
radi.sample(50, seed=42)    # Random sample
radi.select_collections(["T1w", "FLAIR"])

# Query builder (pipeline mode)
radi.query().filter(...).to_radi_object(uri)
```

### VolumeCollection

```python
# Access volumes
vc.iloc[0]                  # By position
vc.loc["obs-123"]           # By obs_id
vc[0]                       # Shorthand

# Filtering
vc.filter("voxel_spacing == '1.0x1.0x1.0'")
vc.head(10)
vc.sample(50)

# Query builder
vc.query().filter(...).to_volume_collection(uri)

# Transform and materialize
vc.map(lambda v: v * 2).to_volume_collection(uri)
```

### Volume

```python
# Read data
vol.to_numpy()              # Full volume
vol[100:200, :, :]          # Partial read (efficient)
vol.axial(z=50)             # Single axial slice
vol.sagittal(x=100)         # Single sagittal slice
vol.coronal(y=100)          # Single coronal slice

# Properties
vol.shape                   # (X, Y, Z) or (X, Y, Z, T)
vol.dtype                   # numpy dtype
vol.obs_id                  # Observation identifier
vol.orientation_info        # Anatomical orientation

# Analysis
vol.get_statistics()        # mean, std, min, max, median
vol.compute_histogram()     # Intensity histogram

# Export
vol.to_nifti("output.nii.gz")
```

### Configuration

```python
from radiobject import configure, WriteConfig, ReadConfig, TileConfig

# Configure write settings
configure(write=WriteConfig(
    tile=TileConfig(orientation=SliceOrientation.AXIAL),
    compression=CompressionConfig(algorithm=Compressor.ZSTD, level=3),
))

# Configure read settings
configure(read=ReadConfig(
    memory_budget_mb=2048,
    max_workers=8,
))
```
