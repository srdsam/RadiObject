"""CPU, memory, and GPU profiling utilities for benchmarks."""

from __future__ import annotations

import gc
import threading
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Callable

import numpy as np
import psutil

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

# Check for CUDA availability
try:
    import torch

    HAVE_CUDA = torch.cuda.is_available()
except ImportError:
    HAVE_CUDA = False


class CPUSampler:
    """Background thread for CPU sampling during an operation."""

    def __init__(self, interval_ms: int = 100):
        self.interval = interval_ms / 1000
        self.samples: list[float] = []
        self._stop = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start background CPU sampling."""
        self._stop = False
        self.samples = []
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        while not self._stop:
            self.samples.append(psutil.cpu_percent(interval=None))
            time.sleep(self.interval)

    def stop(self) -> tuple[float, float]:
        """Stop sampling and return (mean, peak) CPU percentages."""
        self._stop = True
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.samples:
            return float(np.mean(self.samples)), float(max(self.samples))
        return 0.0, 0.0

    def __enter__(self) -> CPUSampler:
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


@dataclass
class BenchmarkResult:
    """Single benchmark run result with comprehensive profiling."""

    framework: str
    benchmark_name: str
    scenario: str  # "local" or "s3"
    tiling_strategy: str = ""  # "axial", "isotropic", "custom"
    storage_format: str = ""  # "tiledb", "nifti_gz", "nifti", "numpy"

    # Timing
    time_mean_ms: float = 0.0
    time_std_ms: float = 0.0
    cold_start_ms: float = 0.0
    batch_times_ms: list[float] = field(default_factory=list)

    # CPU metrics
    cpu_percent_mean: float = 0.0
    cpu_percent_peak: float = 0.0

    # Memory metrics (CPU)
    peak_heap_mb: float = 0.0  # tracemalloc
    peak_rss_mb: float = 0.0  # psutil

    # GPU metrics
    peak_gpu_allocated_mb: float = 0.0  # torch.cuda.max_memory_allocated
    peak_gpu_reserved_mb: float = 0.0  # torch.cuda.max_memory_reserved

    # Disk metrics
    disk_size_mb: float = 0.0

    # Throughput
    throughput_samples_per_sec: float = 0.0
    throughput_mb_per_sec: float = 0.0

    # Metadata
    data_size_mb: float = 0.0
    n_samples: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "framework": self.framework,
            "benchmark_name": self.benchmark_name,
            "scenario": self.scenario,
            "tiling_strategy": self.tiling_strategy,
            "storage_format": self.storage_format,
            "time_mean_ms": round(self.time_mean_ms, 3),
            "time_std_ms": round(self.time_std_ms, 3),
            "cold_start_ms": round(self.cold_start_ms, 3),
            "cpu_percent_mean": round(self.cpu_percent_mean, 1),
            "cpu_percent_peak": round(self.cpu_percent_peak, 1),
            "peak_heap_mb": round(self.peak_heap_mb, 2),
            "peak_rss_mb": round(self.peak_rss_mb, 2),
            "peak_gpu_allocated_mb": round(self.peak_gpu_allocated_mb, 2),
            "peak_gpu_reserved_mb": round(self.peak_gpu_reserved_mb, 2),
            "disk_size_mb": round(self.disk_size_mb, 2),
            "throughput_samples_per_sec": round(self.throughput_samples_per_sec, 2),
            "n_samples": self.n_samples,
        }


@dataclass
class DiskSpaceResult:
    """Disk space measurement for a storage format."""

    format_name: str
    path: str
    size_bytes: int
    size_mb: float
    n_files: int
    compression_ratio: float = 0.0  # vs raw voxel data
    raw_voxel_bytes: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "format_name": self.format_name,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "size_mb": round(self.size_mb, 2),
            "n_files": self.n_files,
            "compression_ratio": round(self.compression_ratio, 3),
        }


def benchmark_operation(
    func: Callable,
    framework: str,
    benchmark_name: str,
    scenario: str = "local",
    tiling: str = "",
    storage_format: str = "",
    n_warmup: int = 5,
    n_runs: int = 10,
    track_gpu: bool = False,
) -> BenchmarkResult:
    """Benchmark an operation with CPU, memory, and GPU profiling."""
    process = psutil.Process()
    gc.collect()

    # Reset GPU memory stats if tracking
    if track_gpu and HAVE_CUDA:
        import torch

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Cold start with memory tracking
    tracemalloc.start()
    rss_before = process.memory_info().rss
    cpu_sampler = CPUSampler()
    cpu_sampler.start()

    cold_start = time.perf_counter()
    func()
    cold_time = (time.perf_counter() - cold_start) * 1000

    cpu_mean, cpu_peak = cpu_sampler.stop()
    _, peak_heap = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = process.memory_info().rss

    # Capture GPU metrics after cold start
    gpu_allocated = 0.0
    gpu_reserved = 0.0
    if track_gpu and HAVE_CUDA:
        import torch

        gpu_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
        gpu_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)

    # Warmup (remaining iterations)
    for _ in range(n_warmup - 1):
        func()

    # Timed runs with GC between each
    times = []
    for _ in range(n_runs):
        gc.collect()
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)

    return BenchmarkResult(
        framework=framework,
        benchmark_name=benchmark_name,
        scenario=scenario,
        tiling_strategy=tiling,
        storage_format=storage_format,
        time_mean_ms=float(np.mean(times)),
        time_std_ms=float(np.std(times)),
        cold_start_ms=cold_time,
        batch_times_ms=times,
        cpu_percent_mean=cpu_mean,
        cpu_percent_peak=cpu_peak,
        peak_heap_mb=peak_heap / (1024 * 1024),
        peak_rss_mb=(rss_after - rss_before) / (1024 * 1024),
        peak_gpu_allocated_mb=gpu_allocated,
        peak_gpu_reserved_mb=gpu_reserved,
        n_samples=n_runs,
    )


def benchmark_dataloader(
    loader: DataLoader,
    framework: str,
    benchmark_name: str,
    scenario: str,
    tiling: str = "",
    image_key: str = "image",
    batch_size: int = 4,
    n_warmup: int = 5,
    n_batches: int = 20,
) -> BenchmarkResult:
    """Benchmark a PyTorch DataLoader with CPU/memory profiling."""
    process = psutil.Process()
    gc.collect()
    tracemalloc.start()
    rss_before = process.memory_info().rss
    cpu_sampler = CPUSampler()
    cpu_sampler.start()

    # Cold start
    loader_iter = iter(loader)
    cold_start = time.perf_counter()
    first_batch = next(loader_iter)
    if isinstance(first_batch, dict):
        _ = first_batch[image_key].shape
    else:
        _ = first_batch.shape
    cold_start_time = (time.perf_counter() - cold_start) * 1000

    # Warmup
    for _ in range(n_warmup - 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

    # Benchmark batches
    batch_times = []
    for _ in range(n_batches):
        try:
            start = time.perf_counter()
            batch = next(loader_iter)
            if isinstance(batch, dict):
                _ = batch[image_key].shape
            else:
                _ = batch.shape
            batch_times.append((time.perf_counter() - start) * 1000)
        except StopIteration:
            break

    cpu_mean, cpu_peak = cpu_sampler.stop()
    _, peak_heap = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = process.memory_info().rss

    mean_batch = float(np.mean(batch_times)) if batch_times else 0.0
    throughput = (batch_size / (mean_batch / 1000)) if mean_batch > 0 else 0.0

    return BenchmarkResult(
        framework=framework,
        benchmark_name=benchmark_name,
        scenario=scenario,
        tiling_strategy=tiling,
        time_mean_ms=mean_batch,
        time_std_ms=float(np.std(batch_times)) if batch_times else 0.0,
        cold_start_ms=cold_start_time,
        batch_times_ms=batch_times,
        cpu_percent_mean=cpu_mean,
        cpu_percent_peak=cpu_peak,
        peak_heap_mb=peak_heap / (1024 * 1024),
        peak_rss_mb=(rss_after - rss_before) / (1024 * 1024),
        throughput_samples_per_sec=throughput,
        n_samples=len(batch_times) * batch_size,
    )
