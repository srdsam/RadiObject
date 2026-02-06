# RadiObject Benchmarks

I/O performance analysis. RadiObject provides fast storage/loading; use it **with** MONAI/TorchIO transforms.

## Quick Start

```bash
eval $(aws configure export-credentials --profile souzy-s3 --format env)
python run_experiments.py --all
```

## Configuration Presets

```bash
python run_experiments.py --all --config quick    # Fast iteration
python run_experiments.py --all --config default  # Full suite
```

## Prerequisites

1. **Dataset:** `python scripts/download_dataset.py msd-brain-tumour`
2. **S3 credentials (mandatory):** `eval $(aws configure export-credentials --profile souzy-s3 --format env)`
3. **Profiling deps:** `uv pip install -e ".[benchmark]"` (installs pyinstrument)

## Experiments

| # | Notebook | What it measures |
|---|----------|-----------------|
| 01 | `storage_format_analysis` | Format loading times and disk space |
| 02 | `tiledb_deep_dive` | Tiling strategy vs access pattern |
| 03 | `framework_comparison` | RadiObject vs MONAI vs TorchIO |
| 04 | `ml_dataloader_throughput` | Training dataloader throughput |
| 05 | `call_tree_profiling` | pyinstrument call-tree flamegraphs |

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
