# Lexicon

RadiObject-specific terminology. For general medical imaging terms, see the [DICOM standard](https://www.dicomstandard.org/), [NIfTI specification](https://nifti.nimh.nih.gov/), and [BIDS specification](https://bids-specification.readthedocs.io/).

---

## Coordinate Systems and Orientation

| Term | Definition |
|------|------------|
| **RAS** | Right-Anterior-Superior — standard neuroimaging coordinate system (NIfTI default) |
| **LPS** | Left-Posterior-Superior — ITK/some DICOM coordinate system |
| **axcodes** | 3-character tuple indicating axis directions, e.g., `('R', 'A', 'S')` |
| **affine matrix** | 4x4 matrix mapping voxel indices `(i,j,k)` to world coordinates `(x,y,z)` |
| **axial / sagittal / coronal** | Anatomical planes: top-down, side, and front views respectively |

---

## RadiObject Core Terminology

| Term | Definition |
|------|------------|
| **RadiObject** | Top-level TileDB Group organizing subject metadata (ObsMeta) and multiple VolumeCollections. Filtering returns views (`is_view=True`). |
| **is_view** | Boolean property indicating whether a RadiObject/VolumeCollection is a filtered view |
| **obs_subject_id** | Unique string identifier for a subject. Primary key in `obs_meta`, foreign key in `obs` |
| **ObsMeta** | Subject-level observational metadata. Single-dimension TileDB sparse array keyed by obs_subject_id only. Contains system-managed `obs_ids` column (JSON list of volume obs_ids per subject). |
| **VolumeCollection** | TileDB Group organizing multiple Volumes with consistent X/Y/Z dimensions. Filtering returns views. |
| **Volume** | Single 3D or 4D radiology acquisition backed by TileDB dense array |
| **obs_id** | Unique string identifier for a Volume, unique across entire RadiObject |
| **obs** | Volume-level observation dataframe containing per-volume metadata |
| **Dataframe** | TileDB-backed sparse array with configurable index dimensions for tabular data |
| **INDEX_COLUMNS** | Default dimension tuple `(obs_subject_id, obs_id)` for VolumeCollection.obs. obs_meta uses `(obs_subject_id,)` only. |
| **obs_ids** | System-managed obs_meta attribute: JSON-serialized sorted list of volume obs_ids linked to each subject |
| **generate_obs_id()** | Creates obs_id from `f"{obs_subject_id}_{collection_name}"`. Ensures global uniqueness across collections |

---

## Indexing and Filtering

| Term | Definition |
|------|------------|
| **Index** | Immutable named dataclass providing bidirectional mapping between string IDs and integer positions. Supports set algebra (`&`, `\|`, `-`, `^`), alignment checking (`is_aligned`). |
| **align()** | Standalone function computing the intersection of multiple Index objects |
| **iloc** | Integer-location based indexer for selecting by position |
| **loc** | Label-based indexer for selecting by obs_subject_id or obs_id |
| **sel()** | Named-parameter selection by obs_subject_id |
| **subjects** | VolumeCollection property returning deduplicated subject IDs in insertion order |
| **groupby_subject()** | VolumeCollection iterator yielding `(subject_id, VolumeCollection)` pairs |
| **filter()** | Filter subjects using TileDB QueryCondition expression |
| **head() / tail() / sample()** | Return view of first/last/random n subjects |
| **select_collections()** | Filter RadiObject to specified collections (returns view) |
| **describe()** | Return summary showing subjects, collections, shapes, and label distributions |

---

## Query System

| Term | Definition |
|------|------------|
| **LazyQuery** | Lazy filter builder for VolumeCollection with deferred transforms. Entry point: `vc.lazy()` |
| **EagerQuery** | Computed query results with pipeline methods (`.map()`, `.write()`, `.to_list()`). Entry point: `vc.map(fn)` |
| **lazy()** | VolumeCollection method returning LazyQuery for deferred transforms |
| **map()** | Apply transform to each volume. On VolumeCollection returns EagerQuery (eager); on LazyQuery defers execution |
| **write()** | Persist query results to new VolumeCollection |
| **to_list()** | Extract raw results from EagerQuery |
| **to_dataframe()** | Summarize EagerQuery as a DataFrame with obs metadata and a `result` summary column. Accepts optional `columns` to override obs column selection |
| **iter_volumes()** | LazyQuery streaming iterator yielding Volume objects |
| **to_numpy_stack()** | Load all matching volumes as stacked array `(N, X, Y, Z)` |
| **TransformFn** | Type alias: `Callable[[np.ndarray, pd.Series], TransformResult]` — receives `(volume, obs_row)` |
| **TransformResult** | Union type: `np.ndarray` or `tuple[np.ndarray, dict[str, AttrValue]]` — volume alone or volume + obs updates |
| **BatchTransformFn** | Type alias for batch transform: receives list of `(volume, obs_row)` tuples |

---

## Writers

| Term | Definition |
|------|------------|
| **VolumeCollectionWriter** | Context manager for incremental VolumeCollection writes |
| **RadiObjectWriter** | Context manager for building RadiObject incrementally |
| **write_volume()** | Write single volume data to VolumeCollectionWriter |
| **write_batch()** | Write multiple volumes at once |
| **add_collection()** | Add new collection to RadiObjectWriter, returns VolumeCollectionWriter |
| **write()** | Persist a RadiObject or VolumeCollection (including views) to storage |

---

## Ingestion

| Term | Definition |
|------|------------|
| **from_images** | Factory method for bulk ingestion (NIfTI and DICOM) via `images` dict with auto format detection |
| **images** | `from_images` param: dict mapping collection names to image sources (glob, directory, or pre-resolved tuples) |
| **ImageFormat** | Enum (`NIFTI`, `DICOM`) representing detected or hinted image format |
| **format_hint** | Optional dict mapping collection names to format strings (`"nifti"` or `"dicom"`) for ambiguous sources |
| **validate_alignment** | Verify all collections have matching subject IDs |
| **series_type** | BIDS-aligned identifier for imaging sequence type (T1w, T2w, FLAIR, bold, dwi, CT) |
| **append()** | Atomic method to add new subjects to existing RadiObject or VolumeCollection |

---

## Volume Operations

| Term | Definition |
|------|------------|
| **partial read** | Reading a subset of volume data without loading the entire volume |
| **axial() / sagittal() / coronal()** | Partial read returning a 2D plane at specified coordinate |
| **slice()** | Partial read returning arbitrary 3D ROI |
| **to_numpy()** | Full volume read returning complete numpy array |
| **to_nifti()** | Export volume to NIfTI file format |
| **get_statistics()** | Compute descriptive statistics (mean, std, min, max, percentiles) |
| **compute_histogram()** | Compute intensity histogram |

---

## Orientation

| Term | Definition |
|------|------------|
| **OrientationInfo** | Pydantic model storing anatomical orientation metadata |
| **tile_orientation** | Volume property returning SliceOrientation enum for tile chunking strategy |
| **canonical orientation** | Standard RAS orientation where X+=Right, Y+=Anterior, Z+=Superior |
| **reorientation** | Transforming volume data and affine to match a target orientation |

---

## Configuration

| Term | Definition |
|------|------------|
| **RadiObjectConfig** | Top-level configuration with nested `write`, `read`, and `s3` settings |
| **WriteConfig** | Settings applied when creating new TileDB arrays. Immutable after creation |
| **ReadConfig** | Read settings (memory_budget_mb, concurrency, max_workers) |
| **TileConfig** | Tile chunking configuration (orientation, extents) |
| **CompressionConfig** | Compression settings (algorithm, level) |
| **OrientationConfig** | Anatomical orientation settings during ingestion |
| **S3Config** | Cloud storage settings (region, endpoint, max_parallel_ops) |
| **configure()** | Global function to update configuration |
| **get_tiledb_ctx()** | Returns lazily-built TileDB context from current configuration |

---

## ML Training

| Term | Definition |
|------|------------|
| **VolumeCollectionDataset** | PyTorch Dataset wrapping VolumeCollection(s) |
| **VolumeCollectionSubjectsDataset** | TorchIO-compatible Dataset yielding `tio.Subject` objects |
| **LoadingMode** | Enum: `FULL_VOLUME`, `PATCH`, or `SLICE_2D` |
| **DatasetConfig** | Pydantic model configuring dataset behavior |
| **create_training_dataloader** | Factory function producing configured DataLoader |

---

## Threading and Concurrency

| Term | Definition |
|------|------------|
| **ctx_for_threads()** | Returns shared TileDB context for thread pool workers (metadata caching works) |
| **ctx_for_process()** | Creates new TileDB context for forked processes (DataLoader workers) |
| **persistent_workers** | DataLoader setting to keep worker processes alive between epochs |
| **pin_memory** | DataLoader setting to copy tensors into CUDA pinned memory |
