# RadiObject Benchmarks

I/O performance analysis. RadiObject provides fast storage/loading; use it **with** MONAI/TorchIO transforms.

## Quick Start

```bash
# Export AWS credentials for S3 benchmarks (use your own profile)
eval $(aws configure export-credentials --profile <your-profile> --format env)
python run_experiments.py --all
```

## Configuration Presets

```bash
python run_experiments.py --all --config quick    # Fast iteration
python run_experiments.py --all --config default  # Full suite
```

## Prerequisites

1. **Dataset:** `python scripts/download_dataset.py msd-brain-tumour`
2. **S3 credentials (mandatory):** `eval $(aws configure export-credentials --profile <your-profile> --format env)`
3. **Benchmark deps:** `uv sync --group benchmark` (installs pyinstrument, zarr, s3fs)

## Experiments

| # | Notebook | What it measures |
|---|----------|-----------------|
| 01 | `storage_format_analysis` | Format loading times and disk space (incl. Zarr) |
| 02 | `chunked_format_deep_dive` | TileDB/Zarr chunking strategy vs access pattern |
| 03 | `framework_comparison` | RadiObject vs MONAI vs TorchIO vs Zarr |
| 04 | `ml_dataloader_throughput` | Training dataloader throughput (incl. Zarr) |
| 05 | `call_tree_profiling` | pyinstrument call-tree flamegraphs |

## After Running Benchmarks

Figures are generated in `results/figures/`. These must be **manually copied** to `docs/assets/benchmarks/` for the documentation site:

```bash
cp results/figures/*.png ../docs/assets/benchmarks/
```

## Directory Structure

```
benchmarks/
├── config.py              # Configuration
├── run_experiments.py     # Orchestrator
├── infrastructure/        # Profiling utilities
├── experiments/           # Jupyter notebooks
├── results/raw/           # Executed notebooks
└── assets/                # Charts
```
