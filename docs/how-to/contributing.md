# Contributing

See [Layout](../reference/layout.md) for codebase structure and file locations.

## Development Setup

```bash
# Clone and install
git clone https://github.com/srdsam/RadiObject.git
cd radiobject

# Install with dev dependencies
uv sync --extra dev

# Or with ML dependencies for full test coverage
uv sync --all-extras
```

## Running Tests

```bash
uv run pytest              # All tests
uv run pytest test/        # Specific directory
uv run pytest -v           # Verbose output
uv run pytest -x           # Stop on first failure
```

Tests require S3 access for integration tests. Set up AWS credentials per [S3 Setup](s3-setup.md).

**Quick test setup:**

```bash
# Activate S3 credentials (required for integration tests)
eval $(aws configure export-credentials --profile souzy-s3 --format env)

# Download test datasets
python scripts/download_dataset.py --all-tests
```

## Running Benchmarks

Benchmarks require MONAI, TorchIO, and S3 access:

```bash
# Install benchmark dependencies
uv sync --extra ml --extra dev

# Execute notebook in-place
uv run jupyter nbconvert --execute --inplace benchmarks/framework_benchmark.ipynb \
    --ExecutePreprocessor.timeout=3600

# Or run parameterized benchmarks via papermill
cd benchmarks && uv run python run_benchmarks.py
```

Benchmark outputs (charts, tables) are saved to `assets/benchmark/`.

## Linting & Formatting

```bash
uv run ruff check .        # Lint
uv run ruff check . --fix  # Auto-fix
uv run ruff format .       # Format
```

## Generating API Documentation

API documentation is auto-generated from docstrings using MkDocs + mkdocstrings:

```bash
uv sync --extra docs
uv run mkdocs serve    # Live preview at http://127.0.0.1:8000
uv run mkdocs build    # Build to site/
```

Pre-commit hooks run automatically on commit:

```bash
uv run pre-commit install  # Set up hooks (one-time)
uv run pre-commit run -a   # Run manually on all files
```

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `src/radiobject/` | Core library |
| `test/` | Test suite |
| `notebooks/` | Tutorial notebooks |
| `benchmarks/` | Performance benchmark suite |
| `docs/` | Documentation |
| `scripts/` | Utility scripts |
| `assets/` | Generated charts and images |
| `data/` | Local test data (gitignored) |

## Pull Request Workflow

1. Create a feature branch from `main`
2. Make changes and add tests
3. Run linting and tests locally:
   ```bash
   uv run ruff check . --fix
   uv run pytest test/ -v
   ```
4. Push and open a PR against `main`
5. Ensure CI passes before requesting review
