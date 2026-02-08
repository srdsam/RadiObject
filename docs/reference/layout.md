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
├── benchmarks/
│   ├── README.md                   # Benchmark suite documentation
│   ├── config.py                   # BenchmarkConfig dataclass and presets
│   ├── run_experiments.py          # Orchestrator for experiments
│   ├── infrastructure/             # Profiling and visualization utilities
│   │   ├── __init__.py
│   │   ├── profiler.py             # CPUSampler, BenchmarkResult, benchmark_operation()
│   │   ├── visualization.py        # Plotting with colorblind-friendly palette
│   │   └── storage.py              # Dataset preparation, format conversion
│   ├── experiments/                # Modular experiment notebooks
│   │   ├── __init__.py
│   │   ├── 01_storage_format_analysis.ipynb   # Format I/O costs
│   │   ├── 02_tiledb_deep_dive.ipynb          # TileDB internals
│   │   ├── 03_framework_comparison.ipynb      # RadiObject vs MONAI vs TorchIO
│   │   └── 04_ml_dataloader_throughput.ipynb  # Training throughput
│   ├── results/                    # Output from experiments
│   │   ├── raw/                    # Executed notebooks
│   │   ├── figures/                # Generated charts
│   │   └── CHANGELOG.md            # Record of result changes
│   └── assets/                     # Output charts and results
│       ├── benchmark_hero.png
│       ├── benchmark_results.json
│       └── ...
├── docs/
│   ├── index.md                    # Homepage
│   ├── tutorials/
│   │   └── index.md                # Tutorials landing page
│   ├── how-to/
│   │   ├── index.md                # How-to guides landing page
│   │   ├── ingest-data.md          # NIfTI/DICOM ingestion
│   │   ├── query-filter-data.md    # Indexing & filtering
│   │   ├── pipelines.md            # Transform pipelines (eager & lazy)
│   │   ├── working-with-metadata.md # Subject and volume metadata
│   │   ├── volume-operations.md    # Partial reads, stats, NIfTI export
│   │   ├── streaming-writes.md     # Large dataset ingestion
│   │   ├── append-data.md          # Incremental updates
│   │   ├── ml-training.md          # MONAI/TorchIO setup
│   │   ├── tuning-concurrency.md   # Concurrency tuning guide
│   │   ├── s3-setup.md             # Cloud storage setup
│   │   ├── troubleshooting.md      # Common issues and solutions
│   │   ├── profiling.md            # Performance monitoring
│   │   └── contributing.md         # Development setup
│   ├── reference/
│   │   ├── index.md                # Reference landing page
│   │   ├── configuration.md        # All configuration classes
│   │   ├── benchmarks.md           # Performance comparison tables
│   │   ├── datasets.md             # Available datasets catalog
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
│       ├── dataframe.md
│       ├── query.md
│       ├── ctx.md
│       ├── stats.md
│       └── ml.md
├── notebooks/
│   ├── README.md                   # Tutorial setup guide
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
│       ├── writers.py
│       ├── utils.py
│       ├── volume.py
│       ├── volume_collection.py
│       ├── data/
│       │   ├── __init__.py         # Dataset URIs and helpers
│       │   ├── registry.py         # Dataset registry
│       │   └── sync.py             # Dataset download/sync
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
│           │   └── segmentation_dataset.py
│           └── utils/
│               ├── __init__.py
│               ├── worker_init.py
│               ├── labels.py
│               └── validation.py
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
    ├── test_writers.py
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
        ├── test_segmentation_dataset.py
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
