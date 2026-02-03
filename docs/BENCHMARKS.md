# Benchmarking

**TL;DR**: RadiObject enables **200-660x faster** partial reads and native S3 access.

## Key Results

| Operation | RadiObject (local) | RadiObject (S3) | MONAI | TorchIO |
|-----------|-------------------|-----------------|-------|---------|
| 2D Slice | **3.8 ms** | **152 ms** | 2502 ms | 777 ms |
| 64³ ROI | **2.2 ms** | **151 ms** | 1229 ms | 760 ms |
| Full Volume | 525 ms | 7135 ms | 1244 ms | 756 ms |

S3 partial reads are **5-16x faster** than local NIfTI frameworks because MONAI/TorchIO must decompress entire volumes.

## Why the Difference?

```
NIfTI:  [full blob] → decompress all → slice
TileDB: [tile][tile] → read 1 tile   → slice
```

## Storage Tradeoff

| Format | Size | Can Partial Read? |
|--------|------|-------------------|
| NIfTI (.nii.gz) | 2.1 GB | No |
| TileDB | 5.7 GB | Yes (local & S3) |

## When to Use Each

| Scenario | Framework |
|----------|-----------|
| Cloud storage (S3/GCS) | **RadiObject** |
| Partial reads (slices, patches) | **RadiObject** |
| Rich augmentation pipelines | TorchIO |
| Existing MONAI workflows | MONAI |

## Memory Efficiency (Peak Heap)

| Operation | RadiObject | MONAI | TorchIO |
|-----------|------------|-------|---------|
| Slice extraction | **1 MB** | 912 MB | 304 MB |
| Full volume | 304 MB | 608 MB | 304 MB |
| Random access (10 vols) | 896 MB | 1482 MB | 589 MB |

RadiObject reads only the tiles needed—slice extraction uses **912x less memory** than MONAI.

## CPU Utilization

| Operation | RadiObject | MONAI | TorchIO |
|-----------|------------|-------|---------|
| Full volume (peak) | 75.5% | 37.7% | 37.5% |
| Slice (peak) | 45.4% | 44.4% | 53.8% |

TileDB parallelizes tile decompression across cores—**2x better CPU utilization** for full volume loads.

---

For detailed analysis including scaling projections, ML training performance, and optimization recommendations, see [PERFORMANCE.md](PERFORMANCE.md).
