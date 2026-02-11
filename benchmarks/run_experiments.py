"""Orchestrator for running benchmark experiments."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import papermill as pm
from config import CONFIGS, FIGURES_DIR, RESULTS_DIR, BenchmarkConfig

EXPERIMENTS = [
    "01_storage_format_analysis",
    "02_chunked_format_deep_dive",
    "03_framework_comparison",
    "04_ml_dataloader_throughput",
    "05_call_tree_profiling",
]

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
RAW_DIR = RESULTS_DIR / "raw"


def config_to_params(config: BenchmarkConfig) -> dict:
    """Convert BenchmarkConfig to papermill parameters."""
    return {
        "BATCH_SIZE": config.batch_size,
        "PATCH_SIZE": config.patch_size,
        "NUM_WORKERS": config.num_workers,
        "N_WARMUP": config.n_warmup,
        "N_RUNS": config.n_runs,
        "N_BATCHES": config.n_batches,
        "N_SUBJECTS": config.n_subjects,
        "RANDOM_SEED": config.random_seed,
        "S3_BUCKET": config.s3_bucket,
        "TILING_STRATEGIES": config.tiling_strategies,
    }


def run_experiment(
    experiment: str,
    config: BenchmarkConfig,
    output_dir: Path,
) -> bool:
    """Execute a single experiment notebook."""
    notebook_path = EXPERIMENTS_DIR / f"{experiment}.ipynb"
    if not notebook_path.exists():
        print(f"Experiment not found: {notebook_path}")
        return False

    output_path = output_dir / f"{experiment}_executed.ipynb"
    params = config_to_params(config)

    print(f"\n{'=' * 60}")
    print(f"Running: {experiment}")
    print(f"Config: {params}")
    print(f"Output: {output_path}")
    print("=" * 60)

    try:
        pm.execute_notebook(
            str(notebook_path),
            str(output_path),
            parameters=params,
            kernel_name="radiobject",
        )
        print(f"Completed: {experiment}")
        return True
    except Exception as e:
        print(f"Failed: {experiment}")
        print(f"Error: {e}")
        return False


def run_all_experiments(
    config_name: str = "default",
    experiments: list[str] | None = None,
) -> dict:
    """Run all or specified experiments with the given configuration."""
    config = CONFIGS.get(config_name, CONFIGS["default"])
    experiments = experiments or EXPERIMENTS

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RAW_DIR / f"run_{timestamp}_{config_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config_name": config_name,
        "config": config_to_params(config),
        "experiments": {},
    }

    for experiment in experiments:
        success = run_experiment(experiment, config, run_dir)
        results["experiments"][experiment] = {"success": success}

    # Save summary
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run RadiObject benchmark experiments")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all experiments (S3 required)",
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        help="Run a specific experiment",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=list(CONFIGS.keys()),
        help="Configuration preset to use",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments and configs",
    )

    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        for exp in EXPERIMENTS:
            path = EXPERIMENTS_DIR / f"{exp}.ipynb"
            status = "ready" if path.exists() else "missing"
            print(f"  {exp} [{status}]")
        print("\nAvailable configurations:")
        for name, cfg in CONFIGS.items():
            print(f"  {name}: batch={cfg.batch_size}, runs={cfg.n_runs}")
        return

    if args.all:
        run_all_experiments(args.config)
    elif args.experiment:
        config = CONFIGS.get(args.config, CONFIGS["default"])
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        run_experiment(args.experiment, config, RAW_DIR)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
