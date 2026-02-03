# Directory Layout

```
RadiObject/
├── .gitignore
├── .python-version
├── CLAUDE.md
├── LICENSE
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
│   ├── index.md                    # Homepage
│   ├── tutorials/
│   │   └── index.md                # Tutorials landing page
│   ├── how-to/
│   │   ├── index.md                # How-to guides landing page
│   │   ├── ingest-data.md          # NIfTI/DICOM ingestion
│   │   ├── query-filter-data.md    # Data access patterns
│   │   ├── lazy-queries.md         # ETL pipelines with transforms
│   │   ├── streaming-writes.md     # Large dataset ingestion
│   │   ├── append-data.md          # Incremental updates
│   │   ├── ml-training.md          # MONAI/TorchIO setup
│   │   ├── tuning-concurrency.md   # Concurrency tuning guide
│   │   ├── s3-setup.md             # Cloud storage setup
│   │   ├── datasets.md             # Download test data
│   │   └── contributing.md         # Development setup
│   ├── reference/
│   │   ├── index.md                # Reference landing page
│   │   ├── configuration.md        # All configuration classes
│   │   ├── benchmarks.md           # Performance comparison tables
│   │   ├── lexicon.md              # Terminology
│   │   └── layout.md               # Codebase structure (this file)
│   ├── explanation/
│   │   ├── index.md                # Explanation landing page
│   │   ├── architecture.md         # TileDB structure and design
│   │   ├── threading-model.md      # Threading and context management
│   │   └── performance-analysis.md # Scaling and optimization
│   ├── notebooks -> ../notebooks   # Symlink to tutorials
│   └── api/                        # Generated API docs (mkdocstrings)
│       ├── radi_object.md
│       ├── volume_collection.md
│       ├── volume.md
│       ├── query.md
│       └── ctx.md
├── notebooks/
│   ├── README.md                   # Tutorial setup guide
│   ├── config.py                   # Tutorial configuration (URIs)
│   ├── 00_ingest_brats.ipynb
│   ├── 01_radi_object.ipynb
│   ├── 02_volume_collection.ipynb
│   ├── 03_volume.ipynb
│   ├── 04_configuration.ipynb
│   ├── 05_ingest_msd.ipynb
│   └── 06_ml_training.ipynb
├── scripts/
│   └── download_dataset.py         # Unified dataset download script
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
│       ├── stats.py                # TileDB statistics collection
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
│           │   └── volume_dataset.py
│           └── utils/
│               ├── __init__.py
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
    ├── test_performance_regression.py  # Performance threshold tests
    ├── test_stats.py                   # TileDB stats collection tests
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
        └── test_compat.py
```

## Excluded (gitignored)

- `.venv/` - Python virtual environment
- `__pycache__/` - Python bytecode cache
- `.pytest_cache/` - Pytest cache
- `.ruff_cache/` - Ruff linter cache
- `data/` - Local test data (synced from S3)
- `data/benchmark/` - Locally-created tiled RadiObject datasets for benchmarking
- `docs/api/` - Generated HTML API documentation
