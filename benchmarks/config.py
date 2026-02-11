"""Benchmark configuration settings."""

from dataclasses import dataclass, field
from pathlib import Path

# S3 region for TileDB VFS
S3_REGION = "us-east-2"

# AWS profile for S3 access (set None to use default credentials)
AWS_PROFILE = None  # Set via env: eval $(aws configure export-credentials --profile <your-profile> --format env)

# S3 bucket for remote benchmarks
S3_BUCKET = "souzy-scratch"

# Absolute paths anchored to benchmarks/ directory
_BENCHMARKS_DIR = Path(__file__).parent
DATA_DIR = _BENCHMARKS_DIR.parent / "data"
BENCHMARK_DIR = DATA_DIR / "benchmark"
NIFTI_DIR = DATA_DIR / "msd-brain-tumour" / "imagesTr"
ASSETS_DIR = _BENCHMARKS_DIR / "assets"
RESULTS_DIR = _BENCHMARKS_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    # DataLoader settings
    batch_size: int = 4
    patch_size: tuple[int, int, int] = (64, 64, 64)
    num_workers: int = 0

    # Benchmark settings
    n_warmup: int = 5
    n_runs: int = 10
    n_batches: int = 20
    n_subjects: int = 20

    # Reproducibility
    random_seed: int = 42

    # S3 settings
    s3_bucket: str = S3_BUCKET

    # Tiling strategies to test
    tiling_strategies: list[str] = field(default_factory=lambda: ["axial", "isotropic"])


# Pre-defined configurations for different scenarios
CONFIGS = {
    "default": BenchmarkConfig(),
    "large_batch": BenchmarkConfig(
        batch_size=8,
        n_runs=3,
    ),
    "large_patch": BenchmarkConfig(
        patch_size=(96, 96, 96),
        n_runs=3,
    ),
    "quick": BenchmarkConfig(
        n_warmup=2,
        n_runs=3,
        n_batches=5,
        n_subjects=5,
    ),
}
