# Pipelines

Transform and persist volumes using eager and lazy pipelines on VolumeCollections. For the full API, see [Query API](../api/query.md).

## Eager vs Lazy

| Mode | Entry Point | Execution | Best For |
|------|-------------|-----------|----------|
| **Eager** | `vc.map(fn)` | Immediate — all volumes loaded and transformed | Small datasets, chaining, inspection |
| **Lazy** | `vc.lazy().map(fn)` | Deferred — single-pass on `.write()` | Large datasets, ETL pipelines |

## Transform Signature

All transforms receive `(volume, obs)` — the volume array and its obs metadata row:

```python
def normalize(volume: np.ndarray, obs: pd.Series) -> np.ndarray:
    return (volume - volume.mean()) / volume.std()

def resample(order):
    def fn(volume, obs):
        spacing = eval(obs["voxel_spacing"])
        factors = tuple(s / 2.0 for s in spacing)
        return zoom(volume, factors, order=order)
    return fn
```

### Returning obs Updates

Transforms can return a second element to annotate obs metadata:

```python
def compute_stats(volume, obs):
    return volume, {"mean_intensity": float(volume.mean())}
```

Updates accumulate through chained `.map()` calls and are written to obs on `.write()`.

## Eager Pipelines

`vc.map(fn)` applies the transform immediately and returns an `EagerQuery`:

```python
# Chain transforms, then extract results
results = radi.CT.map(normalize).map(crop).to_list()

# Write transformed volumes to storage
ct_norm = radi.CT.map(normalize).write("s3://bucket/ct_norm")

# Extract non-volume results (e.g. statistics)
stats = radi.CT.map(lambda v, obs: {"mean": v.mean()}).to_list()
```

## Lazy Pipelines

`vc.lazy()` returns a `LazyQuery` — transforms are deferred until `.write()`:

```python
ct_2mm = radi.CT.lazy().map(resample(order=1)).write("s3://bucket/ct_2mm")
```

### Filter Before Transform

Lazy queries support filtering to reduce the number of volumes processed:

```python
ct_subset = (
    radi.CT.lazy()
    .filter("voxel_spacing == '(1.0, 1.0, 1.0)'")
    .head(100)
    .map(normalize)
    .write("./output")
)
```

### Streaming

Stream volumes one at a time from a LazyQuery:

```python
for vol in radi.CT.lazy().filter("split == 'train'").iter_volumes():
    data = vol.to_numpy()
    process(data)
```

### Stack to Numpy

Load all matching volumes as a stacked array `(N, X, Y, Z)`:

```python
stack = radi.CT.lazy().head(10).to_numpy_stack()
```

## End-to-End Example

Resample and assemble a new RadiObject:

```python
from radiobject import RadiObject

URI = "s3://bucket/study_2mm"
radi_raw = RadiObject("s3://bucket/study_raw")

ct_2mm = radi_raw.CT.lazy().map(resample(order=1)).write(f"{URI}/collections/CT", name="CT")
seg_2mm = radi_raw.seg.lazy().map(resample(order=0)).write(f"{URI}/collections/seg", name="seg")

radi = RadiObject.from_collections(URI, collections={"CT": ct_2mm, "seg": seg_2mm})
```
