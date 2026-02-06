# Benchmark Results Changelog

Record of significant changes to benchmark results.

## Format

Each entry follows:
```
## [Date] - [Version/Run ID]
### Changed
- Description of changes

### Results
- Key metrics affected
```

---

## [2026-02-05] - Initial Restructure

### Changed
- Restructured monolithic `framework_benchmark.ipynb` into modular experiments
- Created infrastructure modules: `profiler.py`, `visualization.py`, `storage.py`
- Added scientific template with hypothesis/rationale/methodology sections

### Baseline Results (from legacy benchmark)
| Benchmark | RadiObject (axial) | MONAI | TorchIO |
|-----------|-------------------|-------|---------|
| Full Volume Load | 525 ms | 1244 ms | 756 ms |
| 2D Slice Extraction | 3.8 ms | 2502 ms | 777 ms |
| 3D ROI Extraction | 2.2 ms (isotropic) | 1229 ms | 760 ms |
| DataLoader Throughput | N/A | N/A | 1.3 samples/sec |

### Notes
- Legacy results preserved in `assets/benchmark_results.json`
- New experiments will generate results in `results/` directory
