# Directory Layout

```
RadiObject/
├── .gitignore
├── .python-version
├── CLAUDE.md
├── LICENSE
├── Lexicon.md
├── Performance.md
├── README.md
├── pyproject.toml
├── uv.lock
├── assets/
│   └── benchmark/
│       ├── benchmark_results.json      # Exported benchmark results with disk space
│       ├── disk_space_comparison.png   # Storage format size comparison
│       ├── format_overhead.png         # NIfTI vs NumPy vs TileDB load times
│       ├── memory_by_backend.png       # Heap/RSS memory comparison
│       ├── full_volume_load.png        # Framework comparison chart
│       ├── slice_extraction.png        # Tiling strategy comparison
│       ├── roi_extraction.png          # 3D ROI extraction comparison
│       ├── dataloader_throughput.png   # ML training throughput
│       ├── s3_vs_local_full.png        # S3 overhead for full reads
│       └── s3_vs_local_slice.png       # S3 overhead for partial reads
├── benchmarks/
│   ├── config.py                   # Benchmark configuration (S3 region)
│   ├── run_benchmarks.py           # Papermill runner for benchmarks
│   └── framework_benchmark.ipynb   # Comprehensive benchmark suite
├── docs/
│   ├── BENCHMARKS.md               # Performance analysis and comparisons
│   ├── CONTRIBUTING.md             # Development setup, testing
│   ├── DATA_ACCESS.md              # Exploration vs pipeline modes, queries
│   ├── DESIGN.md                   # TileDB structure, orientation handling
│   ├── ML_INTEGRATION.md           # MONAI and TorchIO usage
│   └── S3_SETUP.md                 # Optional cloud storage configuration
├── notebooks/
│   ├── README.md                   # Tutorial setup guide
│   ├── config.py                   # Tutorial configuration (URIs)
│   ├── 00_ingest_brats.ipynb
│   ├── 01_radi_object.ipynb
│   ├── 02_volume_collection.ipynb
│   ├── 03_volume.ipynb
│   ├── 04_storage_configuration.ipynb
│   ├── 05_ingest_msd.ipynb
│   └── 06_ml_training.ipynb
├── scripts/
│   └── download_tutorial_data.py   # Downloads BraTS sample data for tutorials
├── src/
│   └── radiobject/
│       ├── __init__.py
│       ├── py.typed
│       ├── ctx.py
│       ├── dataframe.py
│       ├── imaging_metadata.py
│       ├── indexing.py
│       ├── ingest.py
│       ├── orientation.py
│       ├── parallel.py
│       ├── query.py
│       ├── radi_object.py
│       ├── streaming.py
│       ├── utils.py
│       ├── volume.py
│       ├── volume_collection.py
│       └── ml/
│           ├── __init__.py
│           ├── config.py
│           ├── distributed.py
│           ├── factory.py
│           ├── reader.py
│           ├── compat/
│           │   ├── __init__.py
│           │   └── torchio.py
│           ├── datasets/
│           │   ├── __init__.py
│           │   ├── collection_dataset.py
│           │   ├── patch_dataset.py
│           │   └── segmentation_dataset.py
│           └── utils/
│               ├── __init__.py
│               ├── labels.py
│               ├── validation.py
│               └── worker_init.py
└── test/
    ├── __init__.py
    ├── conftest.py
    ├── data/
    │   └── __init__.py
    ├── test_append.py
    ├── test_dataframe.py
    ├── test_from_niftis.py
    ├── test_imaging_metadata.py
    ├── test_indexing.py
    ├── test_ingest.py
    ├── test_orientation.py
    ├── test_parallel.py
    ├── test_query.py
    ├── test_radi_object.py
    ├── test_streaming.py
    ├── test_threading_investigation.py
    ├── test_utils.py
    ├── test_volume_collection.py
    ├── test_volume.py
    └── ml/
        ├── __init__.py
        ├── conftest.py
        ├── test_dataset.py
        ├── test_distributed.py
        ├── test_performance.py
        ├── test_reader.py
        ├── test_threading_ml.py
        ├── test_training.py
        ├── test_compat.py
        └── test_segmentation_dataset.py
```

## Excluded (gitignored)

- `.venv/` - Python virtual environment
- `__pycache__/` - Python bytecode cache
- `.pytest_cache/` - Pytest cache
- `.ruff_cache/` - Ruff linter cache
- `data/` - Local test data (synced from S3)
- `data/benchmark/` - Locally-created tiled RadiObject datasets for benchmarking
