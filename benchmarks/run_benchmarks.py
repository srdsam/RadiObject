"""Papermill runner for framework benchmarks."""

from pathlib import Path

import papermill as pm

NOTEBOOK = "framework_benchmark.ipynb"
OUTPUT_DIR = Path("assets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Benchmark configurations
CONFIGS = [
    {
        "name": "default",
        "params": {
            "BATCH_SIZE": 4,
            "PATCH_SIZE": (64, 64, 64),
            "NUM_WORKERS": 0,
            "N_RUNS": 5,
            "RUN_S3_BENCHMARKS": True,
        },
    },
    {
        "name": "large_batch",
        "params": {
            "BATCH_SIZE": 8,
            "PATCH_SIZE": (64, 64, 64),
            "NUM_WORKERS": 0,
            "N_RUNS": 3,
            "RUN_S3_BENCHMARKS": False,
        },
    },
    {
        "name": "large_patch",
        "params": {
            "BATCH_SIZE": 4,
            "PATCH_SIZE": (96, 96, 96),
            "NUM_WORKERS": 0,
            "N_RUNS": 3,
            "RUN_S3_BENCHMARKS": False,
        },
    },
]


def run_benchmarks() -> None:
    """Execute benchmark notebook with each configuration."""
    for config in CONFIGS:
        name = config["name"]
        params = config["params"]

        output_notebook = OUTPUT_DIR / f"benchmark_{name}.ipynb"
        print(f"\n{'='*60}")
        print(f"Running benchmark: {name}")
        print(f"Parameters: {params}")
        print(f"Output: {output_notebook}")
        print("=" * 60)

        try:
            pm.execute_notebook(
                NOTEBOOK,
                str(output_notebook),
                parameters=params,
                kernel_name="radiobject",
            )
            print(f"Completed: {name}")
        except Exception as e:
            print(f"Failed: {name}")
            print(f"Error: {e}")


if __name__ == "__main__":
    run_benchmarks()
