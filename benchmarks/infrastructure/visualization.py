"""Plotting utilities with colorblind-friendly palette."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from benchmarks.infrastructure.profiler import BenchmarkResult

# Colorblind-friendly palette (Wong 2011)
COLORBLIND_PALETTE = {
    "RadiObject": "#0077BB",
    "MONAI": "#EE7733",
    "TorchIO": "#009988",
    "File": "#888888",
    "nibabel": "#CC3311",
    "numpy": "#33BBEE",
    "local": "#0077BB",
    "s3": "#CC3311",
    "axial": "#0077BB",
    "isotropic": "#EE7733",
    "custom": "#009988",
    "tiledb": "#0077BB",
    "nifti_gz": "#EE7733",
    "nifti": "#009988",
    "zarr": "#EE3377",
}

# Configure matplotlib defaults
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
    }
)


def plot_bar_comparison(
    data: dict[str, float],
    title: str,
    ylabel: str,
    output_path: Path | str,
    errors: dict[str, float] | None = None,
    color_key: str = "framework",
) -> None:
    """Create a bar chart comparing frameworks or formats."""
    labels = list(data.keys())
    values = list(data.values())

    # Determine colors
    if color_key == "framework":
        colors = [COLORBLIND_PALETTE.get(label.split()[0], "#999999") for label in labels]
    else:
        colors = [COLORBLIND_PALETTE.get(label.lower(), "#999999") for label in labels]

    error_vals = [errors.get(label, 0) for label in labels] if errors else None

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        labels,
        values,
        yerr=error_vals,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=1,
    )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Rotate labels if needed
    if len(labels) > 4 or max(len(label) for label in labels) > 15:
        plt.xticks(rotation=30, ha="right")

    # Add value labels
    max_val = max(values) if values else 1
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + (max_val * 0.02),
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_heatmap(
    data: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    output_path: Path | str,
    cmap: str = "viridis",
    fmt: str = ".1f",
) -> None:
    """Create a heatmap for comparing metrics across dimensions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(data, cmap=cmap, aspect="auto")

    # Set ticks
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, f"{data[i, j]:{fmt}}", ha="center", va="center", color="white")

    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


def create_hero_chart(
    results: list[BenchmarkResult],
    output_path: Path | str,
    operations: list[str] | None = None,
) -> None:
    """Create consolidated benchmark comparison chart."""
    import pandas as pd

    if operations is None:
        operations = ["full_volume", "slice_2d", "roi_3d"]

    filtered = [r for r in results if r.benchmark_name in operations]
    if not filtered:
        print("No results found for hero chart")
        return

    # Build rows, collapsing tiling strategies into best time per framework/operation/scenario
    best: dict[tuple[str, str, str], float] = {}
    for r in filtered:
        label = r.framework
        if r.scenario == "s3":
            label += " [S3]"
        key = (label, r.benchmark_name, r.framework)
        if key not in best or r.time_mean_ms < best[key]:
            best[key] = r.time_mean_ms

    rows = [
        {"label": label, "operation": op, "time_ms": t, "framework": fw}
        for (label, op, fw), t in best.items()
    ]

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="operation", columns="label", values="time_ms")

    # Consistent ordering: RadiObject first, then RadiObject [S3], then alphabetical others
    radi_cols = [c for c in pivot.columns if c == "RadiObject"]
    s3_cols = [c for c in pivot.columns if c == "RadiObject [S3]"]
    other_cols = sorted(c for c in pivot.columns if c not in radi_cols + s3_cols)
    pivot = pivot[radi_cols + s3_cols + other_cols]

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax, width=0.8, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Time (ms)")
    ax.set_title("Benchmark Comparison: RadiObject vs MONAI vs TorchIO vs Zarr")
    ax.set_xlabel("")
    ax.legend(title="Backend", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.xticks(rotation=0)
    ax.set_yscale("log")

    # Add value annotations on bars
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_speedup_ratio(
    results: list[BenchmarkResult],
    baseline_framework: str,
    output_path: Path | str,
) -> None:
    """Bar chart showing Nx speedup of RadiObject over a baseline framework per operation."""
    # Group results by operation
    operations: dict[str, dict[str, float]] = {}
    for r in results:
        if r.framework in ("RadiObject", baseline_framework) and r.scenario == "local":
            op = r.benchmark_name
            if op not in operations:
                operations[op] = {}
            # For RadiObject, pick the best tiling time per operation
            key = r.framework
            if key == "RadiObject" and op in operations and "RadiObject" in operations[op]:
                operations[op][key] = min(operations[op][key], r.time_mean_ms)
            else:
                operations[op][key] = r.time_mean_ms

    # Compute speedup ratios (only where both frameworks have data)
    speedups = {}
    for op, times in sorted(operations.items()):
        if "RadiObject" in times and baseline_framework in times and times["RadiObject"] > 0:
            speedups[op] = times[baseline_framework] / times["RadiObject"]

    if not speedups:
        print(f"No comparable results for RadiObject vs {baseline_framework}")
        return

    labels = list(speedups.keys())
    values = list(speedups.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    color = COLORBLIND_PALETTE.get(baseline_framework, "#999999")
    bars = ax.bar(labels, values, color=color, edgecolor="black", linewidth=1)

    ax.set_ylabel("Speedup (x)")
    ax.set_title(f"RadiObject Speedup vs {baseline_framework}")
    ax.axhline(y=1, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    if len(labels) > 4 or max(len(label) for label in labels) > 15:
        plt.xticks(rotation=30, ha="right")

    # Add value annotations
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + (max(values) * 0.02),
            f"{val:.1f}x",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)
