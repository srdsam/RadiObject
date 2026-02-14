# ML Architecture

RadiObject's ML module bridges TileDB-backed medical imaging storage with PyTorch's training ecosystem. This document explains the data flow, design decisions, and trade-offs.

## Data Flow

```
TileDB Dense Array (on disk / S3)
       │
       ▼
VolumeCollection.iloc[i] → Volume
       │
       ├── .to_numpy()     → full 3D array (FULL_VOLUME mode)
       ├── .slice(x,y,z)   → sub-array read (PATCH mode)
       └── .axial(z)       → 2D slice (SLICE_2D mode)
       │
       ▼
Dataset.__getitem__(idx)
       │  Returns dict: {"image": Tensor, "mask": Tensor, "idx": int, ...}
       │
       ▼
transform(sample_dict)     → MONAI/TorchIO dict transforms
       │
       ▼
DataLoader (batching, shuffling, multi-process prefetch)
       │
       ▼
Training loop
```

**Key insight**: PATCH & SLICE_2D mode leverages TileDB's cloud-native sub-array reads — only the requested voxels are decompressed and transferred. No full volume is loaded into memory.

## Dataset Types

| | `VolumeCollectionDataset` | `SegmentationDataset` |
|---|---|---|
| **Input** | 1+ VolumeCollections (stacked as channels) | Separate image + mask collections |
| **Output keys** | `"image"` (multi-channel) | `"image"`, `"mask"` (single-channel each) |
| **Use case** | Classification, regression, multi-modal | Segmentation with per-key transforms |
| **Labels** | Flexible `LabelSource` (column, dict, fn) | Mask is the label |
| **Foreground sampling** | No | Yes (pre-computed coordinates) |

Both datasets share the same loading modes (`FULL_VOLUME`, `PATCH`, `SLICE_2D`) and configuration model (`DatasetConfig`).

## Loading Modes

### FULL_VOLUME

Loads entire 3D volumes. Best for small volumes or when the model expects full spatial context (e.g., whole-brain classification).

- Length: `n_volumes`
- Requires uniform shapes across all collections

### PATCH

Extracts random 3D patches. The primary mode for training on large medical volumes.

- Length: `n_volumes × patches_per_volume`
- Supports heterogeneous volume shapes (each volume can differ)
- Foreground sampling: pre-computes nonzero mask coordinates at init, then samples patch centers from those coordinates (no extra I/O during training)

### SLICE_2D

Extracts 2D slices along a configurable orientation axis.

- Length: `n_volumes × n_slices` (where n_slices depends on orientation)
- `slice_orientation`: `AXIAL` (default, Z-axis), `SAGITTAL` (X-axis), `CORONAL` (Y-axis)
- Requires uniform shapes across all collections

## Transform Integration

Both datasets accept a single `transform` callable that receives the full sample dict. This aligns with MONAI's dict-transform pattern:

```python
from monai.transforms import Compose, RandFlipd, NormalizeIntensityd

transform = Compose([
    RandFlipd(keys=["image", "mask"], prob=0.5),       # spatial: both
    NormalizeIntensityd(keys="image"),                   # intensity: image only
])

dataset = SegmentationDataset(image=ct, mask=seg, transform=transform)
```

MONAI dict transforms use `keys` to select which tensors to modify, so users control transform scope naturally. TorchIO `SubjectsDataset` is supported via the separate `VolumeCollectionSubjectsDataset` compatibility layer.

## Distributed Training

`create_distributed_dataloader` wraps `VolumeCollectionDataset` with PyTorch's `DistributedSampler` for DDP. It handles **data partitioning only** — process group initialization and model wrapping are the user's responsibility. `SegmentationDataset` does not have distributed support; users needing distributed segmentation training should manage the `DistributedSampler` manually.

```python
loader = create_distributed_dataloader(collections, rank=rank, world_size=world_size)
# User must call set_epoch(loader, epoch) each epoch for proper shuffling
```

## Worker Process Isolation

PyTorch DataLoader forks worker processes. Each forked process needs its own TileDB context because libtiledb's internal state (memory pools, S3 connections) cannot be shared across process boundaries.

`worker_init_fn` calls `ctx_for_process()` to create an isolated TileDB context per worker. This is configured automatically by all factory functions when `num_workers > 0`.

## Trade-offs

| Optimized for | Not covered |
|---|---|
| Random patch access (TileDB sub-array reads) | Streaming / sequential access patterns |
| Medical imaging volumes (3D/4D dense arrays) | Sparse data (point clouds, meshes) |
| MONAI dict-transforms + TorchIO (via compat layer) | Custom transform protocols |
| Single-node + DDP training | Model-parallel or pipeline-parallel |
| Foreground-biased sampling (pre-computed) | Online hard-example mining |

## RNG Design

Patch sampling RNG differs by dataset: `SegmentationDataset` uses `np.random.default_rng(seed=None)` (unseeded, so each call produces a different patch), while `VolumeCollectionDataset` uses `np.random.default_rng(seed=idx)` (deterministic per index for reproducibility). Combined with DataLoader shuffling and `persistent_workers`, this ensures diversity across epochs. For distributed training, `DistributedSampler.set_epoch()` re-seeds the sampler each epoch.
