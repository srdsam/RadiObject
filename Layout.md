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
│       └── .gitkeep
├── notebooks/
│   ├── config.py
│   ├── 00_ingest_brats.ipynb
│   ├── 01_radi_object.ipynb
│   ├── 02_volume_collection.ipynb
│   ├── 03_volume.ipynb
│   ├── 04_storage_configuration.ipynb
│   ├── 05_ingest_msd.ipynb
│   ├── 06_ml_training.ipynb
│   └── framework_benchmark.ipynb
├── src/
│   └── radiobject/
│       ├── __init__.py
│       ├── py.typed
│       ├── ctx.py
│       ├── dataframe.py
│       ├── imaging_metadata.py
│       ├── indexing.py
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
│           ├── cache.py
│           ├── config.py
│           ├── distributed.py
│           ├── factory.py
│           ├── reader.py
│           ├── datasets/
│           │   ├── __init__.py
│           │   ├── multimodal.py
│           │   ├── patch_dataset.py
│           │   └── volume_dataset.py
│           ├── transforms/
│           │   ├── __init__.py
│           │   ├── intensity.py
│           │   └── spatial.py
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
    ├── test_orientation.py
    ├── test_parallel.py
    ├── test_query.py
    ├── test_radi_object.py
    ├── test_streaming.py
    ├── test_volume_collection.py
    ├── test_volume.py
    └── ml/
        ├── __init__.py
        ├── conftest.py
        ├── test_dataset.py
        ├── test_performance.py
        └── test_training.py
```

## Excluded (gitignored)

- `.venv/` - Python virtual environment
- `__pycache__/` - Python bytecode cache
- `.pytest_cache/` - Pytest cache
- `.ruff_cache/` - Ruff linter cache
- `data/` - Local test data (synced from S3)
