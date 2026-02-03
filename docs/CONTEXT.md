# TileDB Configuration System

## Core Configuration Classes (tiledb-py)

**`tiledb.Config`** - A dictionary-like object holding configuration parameters:
```python
cfg = tiledb.Config()
cfg["sm.memory_budget"] = "1073741824"  # 1GB
cfg["sm.compute_concurrency_level"] = "4"
```

**`tiledb.Ctx`** - The execution context that owns thread pools and caches:
```python
ctx = tiledb.Ctx(cfg)  # Thread pools allocated here
```

## Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sm.compute_concurrency_level` | # CPU cores | Thread pool for compute-bound tasks |
| `sm.io_concurrency_level` | # CPU cores | Thread pool for I/O-bound tasks |
| `sm.memory_budget` | 5GB | Memory cap for fixed-length attributes |
| `sm.memory_budget_var` | 10GB | Memory cap for variable-length attributes |
| `vfs.s3.max_parallel_ops` | 8 | Max concurrent S3 operations |
| `vfs.s3.multipart_part_size` | 5MB | Multipart upload chunk size |

Sources:
- [TileDB Configuration Docs](https://docs.tiledb.com/main/how-to/configuration)
- [TileDB Parallelism Docs](https://docs.tiledb.com/main/background/internal-mechanics/parallelism)

---

## RadiObject's Current Integration

### Configuration Architecture

```
RadiObjectConfig (Pydantic)
├── WriteConfig
│   ├── TileConfig (orientation, tile extents)
│   ├── CompressionConfig (algorithm, level)
│   └── OrientationConfig (auto_detect, canonical_target)
├── ReadConfig
│   ├── memory_budget_mb (default: 1024)
│   ├── concurrency (default: 4) → sm.compute/io_concurrency_level
│   └── max_workers (default: 4) → Python ThreadPoolExecutor
└── S3Config
    ├── region, endpoint
    ├── max_parallel_ops (default: 8)
    └── multipart_part_size_mb (default: 50)
```

**Location:** `src/radiobject/ctx.py`

### Global State Management

```python
_config: RadiObjectConfig = RadiObjectConfig()  # Global config
_ctx: tiledb.Ctx | None = None                  # Lazily-built context

def ctx() -> tiledb.Ctx:
    """Lazy initialization of global context."""
    global _ctx
    if _ctx is None:
        _ctx = _config.to_tiledb_ctx()
    return _ctx
```

### Context Injection Pattern

All data objects follow this pattern:

```python
class Volume:
    def __init__(self, uri: str, ctx: tiledb.Ctx | None = None):
        self._ctx = ctx  # None = use global

    def _effective_ctx(self) -> tiledb.Ctx:
        return self._ctx if self._ctx else global_ctx()
```

**Pros:**
- Optional context injection for testing/isolation
- Falls back to shared global context
- Consistent across Volume, VolumeCollection, RadiObject

**Cons:**
- Each call to `_effective_ctx()` could be a function call overhead (negligible)

---

## Threading Analysis

### Multi-Layer Threading Architecture

RadiObject operates across **four concurrency layers**:

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 4: PyTorch DataLoader Workers (PROCESSES via fork)       │
│          num_workers=4, persistent_workers=True                 │
├─────────────────────────────────────────────────────────────────┤
│ Layer 3: Python ThreadPoolExecutor (THREADS)                   │
│          max_workers from ReadConfig (default: 4)              │
├─────────────────────────────────────────────────────────────────┤
│ Layer 2: TileDB Internal Threads                               │
│          sm.compute_concurrency_level, sm.io_concurrency_level │
├─────────────────────────────────────────────────────────────────┤
│ Layer 1: S3/VFS Level                                          │
│          vfs.s3.max_parallel_ops (default: 8)                  │
└─────────────────────────────────────────────────────────────────┘
```

### TileDB Threading Model (Critical)

From [TileDB Wiki - Threading Model](https://github.com/TileDB-Inc/TileDB/wiki/Threading-Model):

> "libtiledb is thread-safe, and **globally sharing one Ctx across the whole thread pool is better** because schema and fragment metadata is cached per-Ctx."

**Key insight:** Creating a new `tiledb.Ctx` per thread **loses caching benefits**.

### Worker Context Functions

**Location:** `src/radiobject/parallel.py`

RadiObject provides two semantically distinct context functions based on execution model:

```python
def ctx_for_threads(ctx: tiledb.Ctx | None = None) -> tiledb.Ctx:
    """Return context for thread pool workers.

    TileDB is thread-safe. Sharing a context across threads enables
    metadata caching, reducing I/O for repeated operations.
    """
    return ctx if ctx else global_ctx()


def ctx_for_process(base_ctx: tiledb.Ctx | None = None) -> tiledb.Ctx:
    """Create new context for a forked process.

    Forked processes (e.g., DataLoader workers) have separate memory
    and cannot share TileDB contexts with the parent process.
    """
    if base_ctx is not None:
        return tiledb.Ctx(base_ctx.config())
    return get_config().to_tiledb_ctx()
```

**Usage in volume_collection.py (ThreadPoolExecutor):**
```python
def write_volume(args):
    worker_ctx = ctx_for_threads(self._ctx)  # Returns same context (shared)
    Volume.from_numpy(uri, data, ctx=worker_ctx)
```

**Usage in worker_init.py (DataLoader processes):**
```python
def worker_init_fn(worker_id: int) -> None:
    _ = ctx_for_process()  # Creates new context for forked process
```

### Threads vs Processes Distinction

| Scenario | Context Handling | Function | Behavior |
|----------|------------------|----------|----------|
| `ThreadPoolExecutor` | Shared memory | `ctx_for_threads()` | Returns same context |
| `multiprocessing.Pool` | Isolated memory | `ctx_for_process()` | Creates new context |
| PyTorch DataLoader (num_workers>0) | Fork (isolated) | `ctx_for_process()` | Creates new context |

**Benefits of this design:**
1. Threads share context → metadata caching works across all workers
2. Processes get isolated contexts → no cross-process state corruption

### Test Coverage

From `test/test_threading_investigation.py`:
```python
class TestWorkerContextIsolation:
    def test_ctx_for_threads_returns_same_context(self):
        """Verify ctx_for_threads returns the same context when provided."""
        ...

    def test_ctx_for_process_returns_new_context(self):
        """Verify ctx_for_process returns distinct contexts."""
        ...
```

These tests verify the semantic distinction between thread and process contexts.

For benchmark results and practical performance tuning guidelines, see [PERFORMANCE.md](PERFORMANCE.md).

## References

- [TileDB Configuration Documentation](https://docs.tiledb.com/main/how-to/configuration)
- [TileDB Threading Model Wiki](https://github.com/TileDB-Inc/TileDB/wiki/Threading-Model)
- [TileDB Parallelism Internals](https://docs.tiledb.com/main/background/internal-mechanics/parallelism)
- [TileDB Concurrency & Consistency](https://tiledb-inc-tiledb.readthedocs-hosted.com/en/1.6.3/tutorials/concurrency-consistency.html)
- [TileDB-Py Issue #247: Context Thread Safety](https://github.com/TileDB-Inc/TileDB-Py/issues/247)
- [TileDB-Py Issue #440: Usage Tips](https://github.com/TileDB-Inc/TileDB-Py/issues/440)
