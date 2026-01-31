# Lexicon

A comprehensive terminology reference for radiology data infrastructure. This lexicon bridges the language of clinicians, data scientists, and imaging informaticists.

---

## Table of Contents
1. [Imaging Modalities](#imaging-modalities)
2. [DICOM Standard](#dicom-standard)
3. [NIfTI Format](#nifti-format)
4. [BIDS (Brain Imaging Data Structure)](#bids-brain-imaging-data-structure)
5. [Image Acquisition Parameters](#image-acquisition-parameters)
6. [Coordinate Systems & Orientation](#coordinate-systems--orientation)
7. [Clinical Workflow Systems](#clinical-workflow-systems)
8. [Interoperability Standards](#interoperability-standards)
9. [AI/ML in Radiology](#aiml-in-radiology)
10. [Radiomics & Texture Analysis](#radiomics--texture-analysis)
11. [Clinical Terminology](#clinical-terminology)
12. [Data Processing Concepts](#data-processing-concepts)
13. [Quality & Validation](#quality--validation)

---

## Imaging Modalities

### Primary Modalities

| Term | Abbreviation | Description |
|------|--------------|-------------|
| Computed Tomography | CT/CAT | X-ray based cross-sectional imaging using rotating tube and detectors |
| Magnetic Resonance Imaging | MRI | Imaging using magnetic fields and radiofrequency pulses |
| Positron Emission Tomography | PET | Nuclear medicine technique using radioactive tracers |
| Single Photon Emission CT | SPECT | Nuclear medicine 3D imaging technique |
| Digital Radiography | DR/CR | 2D X-ray imaging (digital or computed) |
| Ultrasound | US | Sound wave-based imaging |
| Mammography | MG | Specialized breast X-ray imaging |
| Fluoroscopy | FL | Real-time X-ray video imaging |
| Digital Subtraction Angiography | DSA | Vascular imaging with contrast subtraction |

### MRI-Specific Modalities

| Term | Description |
|------|-------------|
| **fMRI** | Functional MRI - measures brain activity via blood oxygenation (BOLD signal) |
| **DWI/DTI** | Diffusion Weighted/Tensor Imaging - measures water molecule movement |
| **MRA** | MR Angiography - vascular imaging without contrast |
| **MRS** | MR Spectroscopy - chemical composition analysis |
| **ASL** | Arterial Spin Labeling - perfusion imaging without contrast |
| **SWI** | Susceptibility Weighted Imaging - detects blood products, calcification |
| **BOLD** | Blood Oxygen Level Dependent - signal basis for fMRI |

### CT-Specific Variants

| Term | Description |
|------|-------------|
| **HRCT** | High Resolution CT - thin-slice lung imaging |
| **CTA** | CT Angiography - contrast-enhanced vascular imaging |
| **CTPA** | CT Pulmonary Angiography - for pulmonary embolism |
| **Perfusion CT** | Dynamic contrast imaging for blood flow |
| **Dual-Energy CT** | Uses two X-ray energy levels for material differentiation |
| **Cone Beam CT** | 3D imaging using cone-shaped X-ray beam |

---

## DICOM Standard

### Core Concepts

| Term | Definition |
|------|------------|
| **DICOM** | Digital Imaging and Communications in Medicine - universal medical imaging standard |
| **SOP Class** | Service-Object Pair Class - defines what can be done with an object type |
| **SOP Instance** | A specific occurrence of an object (e.g., one CT image) |
| **IOD** | Information Object Definition - template defining object attributes |
| **Transfer Syntax** | Encoding rules for data exchange (byte order, compression) |
| **UID** | Unique Identifier - globally unique reference string |
| **VR** | Value Representation - data type specification (2-letter code) |

### DICOM Data Model Hierarchy

```
Patient
  └── Study (one imaging session/visit)
        └── Series (sequence of related images)
              └── Instance (single image/object)
```

### Essential DICOM Tags

| Tag | Name | Description |
|-----|------|-------------|
| (0008,0018) | SOP Instance UID | Unique identifier for this image |
| (0008,0060) | Modality | Type of equipment (CT, MR, etc.) |
| (0010,0010) | Patient Name | Patient's full name |
| (0010,0020) | Patient ID | Medical record number |
| (0020,000D) | Study Instance UID | Unique study identifier |
| (0020,000E) | Series Instance UID | Unique series identifier |
| (0020,0013) | Instance Number | Image number within series |
| (0020,0032) | Image Position Patient | XYZ coordinates of first voxel |
| (0020,0037) | Image Orientation Patient | Direction cosines |
| (0028,0010) | Rows | Image height in pixels |
| (0028,0011) | Columns | Image width in pixels |
| (0028,0030) | Pixel Spacing | Physical size of pixels (mm) |
| (0028,1050) | Window Center | Display center value |
| (0028,1051) | Window Width | Display range width |
| (7FE0,0010) | Pixel Data | The actual image data |

### Common Value Representations (VR)

| VR | Name | Example |
|----|------|---------|
| AE | Application Entity | "CTSCANNER1" |
| CS | Code String | "CT", "MR" |
| DA | Date | "20250128" |
| DS | Decimal String | "1.25" |
| DT | Date Time | "20250128143022.000000" |
| IS | Integer String | "512" |
| LO | Long String | "Chest CT with contrast" |
| PN | Person Name | "Doe^John" |
| SH | Short String | "ACC12345" |
| TM | Time | "143022" |
| UI | Unique Identifier | "1.2.840.10008.1.2" |
| US | Unsigned Short | Pixel values |

### DICOM Network Services (DIMSE)

| Service | Type | Description |
|---------|------|-------------|
| C-ECHO | Verification | Tests connectivity between DICOM nodes |
| C-STORE | Storage | Sends images to archive/PACS |
| C-FIND | Query | Searches for studies/series/images |
| C-MOVE | Retrieve | Requests images be sent to a destination |
| C-GET | Retrieve | Retrieves images directly |
| N-CREATE | Normalized | Creates new managed object |
| N-SET | Normalized | Modifies existing object |

### DICOM Port Conventions

| Port | Use |
|------|-----|
| 104 | Default DICOM port |
| 11112 | Common alternative |
| 4242 | Orthanc default |

### Modality Codes

| Code | Modality |
|------|----------|
| CT | Computed Tomography |
| MR | Magnetic Resonance |
| PT | PET |
| NM | Nuclear Medicine |
| US | Ultrasound |
| CR | Computed Radiography |
| DX | Digital Radiography |
| MG | Mammography |
| XA | X-Ray Angiography |
| RF | Radio Fluoroscopy |
| SR | Structured Report |
| SEG | Segmentation |
| RTSTRUCT | RT Structure Set |

---

## NIfTI Format

### Overview

| Term | Definition |
|------|------------|
| **NIfTI** | Neuroimaging Informatics Technology Initiative format |
| **NIfTI-1** | Original spec, 348-byte header, ANALYZE 7.5 compatible |
| **NIfTI-2** | Extended spec, 540-byte header, larger dimension support |
| **.nii** | Single-file format (header + data) |
| **.nii.gz** | Gzip-compressed single file |
| **.hdr/.img** | Dual-file format (separate header and image) |

### NIfTI Header Fields

| Field | Type | Description |
|-------|------|-------------|
| sizeof_hdr | int | Header size (348 for NIfTI-1, 540 for NIfTI-2) |
| dim | short[8] | Array dimensions [ndim, x, y, z, t, u, v, w] |
| datatype | short | Data type code |
| bitpix | short | Bits per voxel |
| pixdim | float[8] | Voxel dimensions [qfac, dx, dy, dz, dt, du, dv, dw] |
| vox_offset | float | Byte offset to data |
| scl_slope | float | Data scaling slope |
| scl_inter | float | Data scaling intercept |
| qform_code | short | Coordinate system code for quaternion |
| sform_code | short | Coordinate system code for affine |
| quatern_b/c/d | float | Quaternion rotation parameters |
| qoffset_x/y/z | float | Quaternion translation parameters |
| srow_x/y/z | float[4] | Affine transformation rows |
| intent_code | short | Statistical/intent type code |
| xyzt_units | char | Spatial and temporal units |

### NIfTI Data Types

| Code | Name | Size |
|------|------|------|
| 2 | UINT8 | 8-bit unsigned |
| 4 | INT16 | 16-bit signed |
| 8 | INT32 | 32-bit signed |
| 16 | FLOAT32 | 32-bit float |
| 32 | COMPLEX64 | 64-bit complex |
| 64 | FLOAT64 | 64-bit float |
| 256 | INT8 | 8-bit signed |
| 512 | UINT16 | 16-bit unsigned |
| 768 | UINT32 | 32-bit unsigned |

### Coordinate System Codes

| Code | Meaning |
|------|---------|
| 0 | Unknown |
| 1 | Scanner-based (SCANNER_ANAT) |
| 2 | Aligned to another file |
| 3 | Talairach space |
| 4 | MNI-152 space |

### qform vs sform

| Transform | Use Case |
|-----------|----------|
| **qform** | Scanner coordinates, uses quaternion + offset for rigid transform |
| **sform** | Standard space coordinates, full 4x4 affine matrix |

---

## BIDS (Brain Imaging Data Structure)

### Directory Structure

```
dataset/
├── dataset_description.json
├── participants.tsv
├── participants.json
├── sub-01/
│   ├── ses-01/
│   │   ├── anat/
│   │   │   └── sub-01_ses-01_T1w.nii.gz
│   │   ├── func/
│   │   │   ├── sub-01_ses-01_task-rest_bold.nii.gz
│   │   │   └── sub-01_ses-01_task-rest_bold.json
│   │   └── dwi/
│   │       ├── sub-01_ses-01_dwi.nii.gz
│   │       ├── sub-01_ses-01_dwi.bval
│   │       └── sub-01_ses-01_dwi.bvec
```

### BIDS Entity Labels

| Entity | Key | Description |
|--------|-----|-------------|
| Subject | sub | Participant identifier |
| Session | ses | Scanning session |
| Task | task | Task performed during acquisition |
| Acquisition | acq | Acquisition variant |
| Contrast Enhancement | ce | Contrast agent used |
| Reconstruction | rec | Reconstruction method |
| Phase-Encoding Direction | dir | Phase encoding direction |
| Run | run | Run number within session |
| Echo | echo | Echo number in multi-echo |
| Part | part | Part of complex data (mag/phase) |
| Chunk | chunk | Chunk number for split files |

### BIDS Suffixes by Modality

**Anatomical MRI:**
- `_T1w`, `_T2w`, `_T1rho`, `_T1map`, `_T2map`
- `_T2star`, `_FLAIR`, `_FLASH`, `_PD`
- `_PDmap`, `_PDT2`, `_inplaneT1`, `_inplaneT2`
- `_angio`, `_defacemask`

**Functional MRI:**
- `_bold`, `_cbv`, `_phase`

**Diffusion MRI:**
- `_dwi`

**Field Maps:**
- `_phasediff`, `_phase1`, `_phase2`
- `_magnitude`, `_magnitude1`, `_magnitude2`
- `_fieldmap`, `_epi`

### BIDS Metadata (JSON Sidecar)

| Field | Description |
|-------|-------------|
| RepetitionTime | TR in seconds |
| EchoTime | TE in seconds |
| FlipAngle | Flip angle in degrees |
| SliceTiming | Slice acquisition times |
| PhaseEncodingDirection | Direction of phase encoding |
| EffectiveEchoSpacing | Effective echo spacing |
| TotalReadoutTime | Total readout duration |
| TaskName | Name of task for functional data |
| Manufacturer | Scanner manufacturer |
| ManufacturersModelName | Scanner model |
| MagneticFieldStrength | Field strength in Tesla |

---

## Image Acquisition Parameters

### MRI Parameters

| Parameter | Abbreviation | Units | Description |
|-----------|--------------|-------|-------------|
| Repetition Time | TR | ms | Time between RF excitation pulses |
| Echo Time | TE | ms | Time from excitation to signal measurement |
| Inversion Time | TI | ms | Time from 180° pulse to 90° pulse |
| Flip Angle | FA/α | degrees | RF pulse tip angle |
| Echo Train Length | ETL | count | Number of echoes per excitation |
| Bandwidth | BW | Hz/pixel | Receiver bandwidth per pixel |
| Number of Excitations | NEX/NSA | count | Signal averages |
| b-value | b | s/mm² | Diffusion weighting factor |

### Geometry Parameters

| Parameter | Description |
|-----------|-------------|
| **FOV** | Field of View - physical area imaged (mm) |
| **Matrix** | Acquisition matrix size (rows × columns) |
| **Slice Thickness** | Thickness of each 2D slice (mm) |
| **Slice Gap** | Space between slices (mm) |
| **Voxel Size** | 3D pixel dimensions (mm × mm × mm) |
| **Pixel Spacing** | In-plane pixel dimensions |
| **Number of Slices** | Total slices in acquisition |
| **Phase FOV** | FOV in phase-encoding direction |

### Derived Calculations

```
Pixel Size = FOV / Matrix
Voxel Volume = Pixel Size(x) × Pixel Size(y) × Slice Thickness
In-plane Resolution = FOV / Matrix dimension

Scan Time (2D SE) = TR × Phase Matrix × NEX
Scan Time (2D FSE) = TR × Phase Matrix × NEX / ETL
Scan Time (3D) = TR × Phase Matrix × NEX × Number of Slices
```

### CT Parameters

| Parameter | Units | Description |
|-----------|-------|-------------|
| kVp | kilovolt | Tube voltage |
| mA/mAs | milliampere | Tube current |
| Pitch | ratio | Table movement per rotation / collimation |
| Collimation | mm | X-ray beam width at detector |
| Slice Thickness | mm | Reconstructed slice thickness |
| Reconstruction Kernel | - | Filter (soft, standard, sharp, bone, lung) |
| Window/Level | HU | Display parameters |
| CTDIvol | mGy | Volume CT dose index |
| DLP | mGy·cm | Dose length product |

### Hounsfield Units (HU) Reference

| Material | HU Range |
|----------|----------|
| Air | -1000 |
| Lung | -500 to -900 |
| Fat | -100 to -50 |
| Water | 0 |
| Soft Tissue | +20 to +70 |
| Bone (cancellous) | +300 to +400 |
| Bone (cortical) | +1000 to +3000 |

---

## Coordinate Systems & Orientation

### Anatomical Planes

| Plane | Also Called | View |
|-------|-------------|------|
| Axial | Transverse | Top-down view |
| Sagittal | - | Side view |
| Coronal | Frontal | Front view |

### Anatomical Directions

| Direction | Opposite | Axis |
|-----------|----------|------|
| Left (L) | Right (R) | X |
| Anterior (A) | Posterior (P) | Y |
| Superior (S) | Inferior (I) | Z |

### Coordinate System Conventions

| Convention | X+ | Y+ | Z+ | Used By |
|------------|----|----|-----|---------|
| **RAS** | Right→Left | Anterior→Posterior | Inferior→Superior | NIfTI, FreeSurfer |
| **LAS** | Left→Right | Anterior→Posterior | Inferior→Superior | DICOM standard |
| **LPS** | Left→Right | Posterior→Anterior | Inferior→Superior | ITK, some DICOM |

### Orientation Strings

| Code | Meaning |
|------|---------|
| RAS | Right-Anterior-Superior (neurological) |
| LAS | Left-Anterior-Superior (radiological) |
| RPI | Right-Posterior-Inferior |
| AIL | Anterior-Inferior-Left |

### Affine Transformation Matrix

```
[x']   [r11 r12 r13 tx]   [i]
[y'] = [r21 r22 r23 ty] × [j]
[z']   [r31 r32 r33 tz]   [k]
[1 ]   [0   0   0   1 ]   [1]

Where:
- (i,j,k) = voxel indices
- (x',y',z') = real-world coordinates
- r = rotation/scaling components
- t = translation components
```

### RadiObject Core Terminology

| Term | Definition |
|------|------------|
| **RadiObject** | Top-level TileDB Group organizing subject metadata (ObsMeta) and multiple VolumeCollections |
| **RadiObjectView** | Immutable filtered view of a RadiObject supporting copy-on-write materialization via `to_radi_object()` |
| **obs_subject_id** | Unique string identifier for a subject (patient/participant). Primary key in RadiObject.obs_meta, foreign key in VolumeCollection.obs |
| **ObsMeta** | Subject-level observational metadata Dataframe in RadiObject, indexed by obs_subject_id |
| **VolumeCollection** | A TileDB Group organizing multiple Volumes with consistent X/Y/Z dimensions |
| **Volume** | A single 3D or 4D radiology acquisition backed by TileDB dense array |
| **obs_id** | Unique string identifier for a Volume, unique across entire RadiObject (stored in TileDB metadata) |
| **obs** | Volume-level observation dataframe containing per-volume metadata, indexed by obs_id with obs_subject_id as foreign key |
| **Dataframe** | A TileDB-backed 2D heterogeneous array for tabular data |
| **all_obs_ids** | RadiObject property returning all obs_ids across all collections (for uniqueness validation) |
| **get_volume()** | RadiObject method returning a Volume by obs_id from any collection |
| **rename_collection()** | RadiObject method to rename a collection within the TileDB group |

### RadiObject Indexing & Filtering Terminology

| Term | Definition |
|------|------------|
| **index** | Public property exposing the bidirectional Index for ID/position lookups (e.g., `radi.index.get_index("sub-01")`, `radi.index.get_key(0)`, `radi.index.keys`) |
| **Index** | Immutable dataclass providing bidirectional mapping between string IDs and integer positions. Methods: `get_index(key)`, `get_key(idx)`, `keys` property |
| **iloc** | Integer-location based indexer for selecting subjects (RadiObject) or volumes (VolumeCollection) by position |
| **loc** | Label-based indexer for selecting by obs_subject_id (RadiObject) or obs_id (VolumeCollection) |
| **__iter__ (VolumeCollection)** | Iterator yielding Volume objects in index order. Enables `for vol in collection:` syntax |
| **boolean mask indexing** | Filtering using a numpy boolean array (e.g., `radi.iloc[mask]` where mask is `np.ndarray[bool]`) |
| **filter()** | Filter subjects using TileDB QueryCondition expression on obs_meta (e.g., `radi.filter("age > 40")`) |
| **__getitem__** | Bracket indexing for RadiObject subjects by obs_subject_id. Alias for .loc[] (e.g., `radi["BraTS001"]`) |
| **describe()** | Return summary string showing subjects, collections, shapes, and label distributions |
| **head()** | Return view of first n subjects |
| **tail()** | Return view of last n subjects |
| **sample()** | Return view of n randomly sampled subjects with optional seed for reproducibility |
| **select_collections()** | Filter RadiObject or RadiObjectView to include only specified collections |
| **get_obs_row_by_obs_subject_id()** | Retrieve obs_meta row by subject ID (RadiObject) |
| **get_obs_row_by_obs_id()** | Retrieve obs row by volume ID (VolumeCollection) |

### RadiObject Query Builder (Pipeline Mode)

| Term | Definition |
|------|------------|
| **Query** | Lazy filter builder for RadiObject that accumulates filters without data access until explicit materialization |
| **CollectionQuery** | Lazy filter builder for VolumeCollection that accumulates filters on obs without data access |
| **query()** | Entry point method returning Query (RadiObject) or CollectionQuery (VolumeCollection) for pipeline-style filtering |
| **filter()** | Add TileDB QueryCondition on obs_meta (Query) or obs (CollectionQuery) |
| **filter_subjects()** | Add explicit subject ID filter to Query |
| **filter_collection()** | Add TileDB QueryCondition on a specific collection's obs within Query |
| **select_collections()** | Limit Query output to specific collections (does not affect filtering) |
| **to_radi_object()** | Materialize Query results as new RadiObject (supports streaming=True for memory efficiency) |
| **to_volume_collection()** | Materialize CollectionQuery results as new VolumeCollection |
| **iter_volumes()** | Streaming iterator yielding Volume objects matching the query |
| **iter_batches()** | Streaming iterator yielding VolumeBatch for ML training |
| **count()** | Materialize query to count subjects and volumes without loading volume data |
| **to_obs_meta()** | Materialize filtered obs_meta DataFrame |
| **to_obs()** | Materialize filtered obs DataFrame (CollectionQuery) |
| **to_numpy_stack()** | Load all matching volumes as stacked numpy array (N, X, Y, Z) |
| **map()** | Apply transform function to each volume during materialization. Composes with previous transforms |
| **TransformFn** | Type alias for volume transform: `Callable[[np.ndarray], np.ndarray]` (X, Y, Z) -> (X', Y', Z') |
| **VolumeBatch** | Dataclass containing stacked numpy arrays per collection and subject IDs for ML training |
| **QueryCount** | Dataclass containing n_subjects and per-collection volume counts |

### RadiObject Streaming Writers

| Term | Definition |
|------|------------|
| **StreamingWriter** | Context manager for incremental VolumeCollection writes without full memory load |
| **RadiObjectWriter** | Context manager for building RadiObject incrementally with multiple collections |
| **write_volume()** | Write single volume data to StreamingWriter with obs_id, obs_subject_id, and attributes |
| **write_batch()** | Write multiple volumes at once to StreamingWriter |
| **write_obs_meta()** | Write subject-level metadata to RadiObjectWriter |
| **add_collection()** | Add new collection to RadiObjectWriter, returning StreamingWriter for its volumes |
| **finalize()** | Complete writing and return the created RadiObject |

### RadiObject Append Operations

| Term | Definition |
|------|------------|
| **append()** | Atomic method to add new subjects and volumes to RadiObject or VolumeCollection |
| **atomic append** | Writing obs_meta and volumes together to maintain referential integrity |
| **cache invalidation** | Automatic clearing of cached properties (_index, _metadata, collection_names) after append |

### RadiObject Analysis Terminology

| Term | Definition |
|------|------------|
| **partial read** | Reading only a subset of volume data (slice, ROI) without loading entire volume into memory |
| **axial()** | Partial read returning X-Y plane at specified Z coordinate |
| **sagittal()** | Partial read returning Y-Z plane at specified X coordinate |
| **coronal()** | Partial read returning X-Z plane at specified Y coordinate |
| **slice()** | Partial read returning arbitrary 3D ROI specified by x, y, z slice objects |
| **to_numpy()** | Full volume read returning complete numpy array |
| **get_statistics()** | Compute descriptive statistics (mean, std, min, max, percentiles) for volume data |
| **compute_histogram()** | Compute intensity histogram of volume data |
| **to_nifti()** | Export volume to NIfTI file format |

### RadiObject Ingestion Terminology

| Term | Definition |
|------|------------|
| **from_niftis** | Factory method for bulk NIfTI ingestion with raw data storage (no preprocessing). Groups by modality or explicit collection_name |
| **from_dicoms** | Factory method for bulk DICOM ingestion with automatic metadata extraction and modality-based grouping |
| **collection_name** | Optional from_niftis param to place all volumes in a single named collection. If None, auto-groups by inferred modality |
| **NiftiMetadata** | Pydantic model capturing NIfTI header fields (dimensions, voxel spacing, orientation, scaling) for obs storage |
| **DicomMetadata** | Pydantic model capturing DICOM header fields (pixel spacing, modality, acquisition parameters) for obs storage |
| **infer_series_type** | Function that determines series type (T1w, FLAIR, etc.) from BIDS filename patterns and header description |
| **series_type** | BIDS-aligned identifier for imaging sequence type (T1w, T2w, FLAIR, bold, dwi, CT, etc.) |
| **voxel_spacing** | Physical voxel dimensions in mm (from NIfTI pixdim or DICOM PixelSpacing/SliceThickness) |
| **scl_slope/scl_inter** | NIfTI intensity scaling parameters: real_value = stored_value * scl_slope + scl_inter |
| **auto_grouping** | Automatic organization of input files into VolumeCollections based on (shape, series_type) tuple |
| **FK_constraint** | Foreign key validation ensuring obs_subject_ids in VolumeCollection.obs reference valid subjects in RadiObject.obs_meta |

### RadiObject Transformation Terminology

| Term | Definition |
|------|------------|
| **map()** | VolumeCollection method that applies a transform function to each volume during materialization. Returns CollectionQuery |
| **TransformFn** | Transform function signature: (np.ndarray) -> np.ndarray. Input is 3D volume, output can have different shape |
| **is_uniform** | VolumeCollection property returning True if all volumes have the same shape |
| **heterogeneous shapes** | Collection allowing volumes with different dimensions (supported after raw ingestion or when transform changes shape) |

### RadiObject Bulk Ingestion Terminology

| Term | Definition |
|------|------------|
| **NiftiSource** | Frozen dataclass representing NIfTI file with optional paired label (image_path, subject_id, label_path) |
| **IngestConfig** | Frozen dataclass configuring ingestion pipeline (resample_to, pad_to, derive_labels, batch_size) |
| **discover_nifti_pairs** | Function discovering NIfTI files in directory and optionally pairing with labels by filename |
| **compute_target_shape** | First-pass function computing target shape for padding (header-only, no data load) |
| **process_volumes_sequential** | Generator processing volumes one at a time with memory cleanup |
| **process_volumes_parallel** | Function processing volumes using ProcessPoolExecutor for CPU-bound resampling |
| **derive_labels** | from_niftis param mapping column names to functions that derive values from label masks |
| **has_any_nonzero** | Helper function checking if label mask has any non-zero voxels |
| **count_nonzero_voxels** | Helper function counting non-zero voxels in label mask |
| **unique_label_count** | Helper function counting unique non-zero labels in mask |
| **ProcessedVolume** | Dataclass containing processed volume data with derived labels and shape metadata |
| **image_dir** | from_niftis param specifying directory containing image NIfTIs |
| **label_dir** | from_niftis param specifying optional directory containing label NIfTIs |

### RadiObject ML Training Terminology

| Term | Definition |
|------|------------|
| **RadiObjectDataset** | PyTorch Dataset wrapping RadiObject for training, supports full_volume, patch, and slice_2d loading modes. Validates subject alignment when multiple modalities are specified |
| **VolumeReader** | Thread-safe wrapper for reading volumes from VolumeCollection with per-worker TileDB contexts |
| **LoadingMode** | Enum specifying data loading strategy: FULL_VOLUME, PATCH, or SLICE_2D |
| **DatasetConfig** | Pydantic model configuring dataset behavior (loading_mode, patch_size, modalities) |
| **worker_init_fn** | DataLoader worker initializer that creates per-worker TileDB contexts |
| **create_training_dataloader** | Factory function producing configured DataLoader with shuffle, pin_memory, persistent_workers |
| **create_distributed_dataloader** | Factory for DDP-compatible DataLoader with DistributedSampler |
| **patches_per_volume** | Number of random patches extracted per volume per epoch (default: 1) |
| **persistent_workers** | DataLoader setting to keep worker processes alive between epochs for faster iteration |
| **pin_memory** | DataLoader setting to copy tensors into CUDA pinned memory for faster GPU transfer |
| **RadiObjectSubjectsDataset** | TorchIO-compatible Dataset yielding tio.Subject objects for TorchIO Queue integration and patch-based training |
| **Compose** | Transform composition utility. Uses MONAI Compose if available, falls back to TorchIO Compose, then minimal fallback |
| **MONAI dict transforms** | MONAI transforms that operate on dict[str, Any] (e.g., NormalizeIntensityd, RandFlipd). Work directly with RadiObjectDataset output |
| **TorchIO Subject transforms** | TorchIO transforms that operate on tio.Subject objects. Require RadiObjectSubjectsDataset for integration |

### RadiObject Orientation Terminology

| Term | Definition |
|------|------------|
| **axcodes** | 3-character tuple indicating axis directions, e.g., ('R', 'A', 'S') |
| **canonical orientation** | Standard RAS orientation where X+ = Right, Y+ = Anterior, Z+ = Superior |
| **reorientation** | Transforming volume data and affine to match a target orientation |
| **OrientationInfo** | Pydantic model storing anatomical orientation metadata (axcodes, affine, source) |
| **tile_orientation** | Volume property returning SliceOrientation enum indicating tile chunking strategy used at creation |
| **orientation_affine** | TileDB metadata key storing 4x4 affine as JSON |
| **orientation_axcodes** | TileDB metadata key storing 3-char orientation string |
| **original_affine** | TileDB metadata preserving pre-reorientation affine for provenance |
| **orientation_source** | Origin of orientation data: nifti_sform, nifti_qform, dicom_iop, or identity |
| **orientation_confidence** | Trust level: header (from file), inferred (ML), or unknown |

### RadiObject Configuration Terminology

| Term | Definition |
|------|------------|
| **RadiObjectConfig** | Top-level configuration model with nested `write`, `read`, and `s3` settings |
| **WriteConfig** | Settings applied when creating new TileDB arrays (tile, compression, orientation). Immutable after array creation |
| **ReadConfig** | Settings for reading TileDB arrays (memory_budget_mb, concurrency, max_workers). Affects all reads |
| **TileConfig** | Tile chunking configuration (orientation, x/y/z extents). Part of WriteConfig |
| **CompressionConfig** | Compression settings (algorithm, level). Part of WriteConfig |
| **OrientationConfig** | Anatomical orientation settings (canonical_target, reorient_on_load). Part of WriteConfig |
| **S3Config** | Cloud storage settings (region, endpoint, credentials). Applies to both read and write |
| **configure()** | Global function to update configuration. Supports nested API (`write=WriteConfig(...)`) and deprecated flat API (`tile=TileConfig(...)`) |
| **get_config()** | Returns current global RadiObjectConfig instance |
| **ctx()** | Returns lazily-built TileDB context from current configuration |

---

## Clinical Workflow Systems

### Core Systems

| System | Full Name | Function |
|--------|-----------|----------|
| **PACS** | Picture Archiving and Communication System | Stores, retrieves, distributes medical images |
| **RIS** | Radiology Information System | Manages scheduling, reporting, billing |
| **HIS** | Hospital Information System | Enterprise-wide patient data management |
| **EHR/EMR** | Electronic Health/Medical Record | Comprehensive patient records |
| **VNA** | Vendor Neutral Archive | Long-term, vendor-agnostic image storage |
| **LIMS** | Laboratory Information Management System | Lab workflow management |

### PACS Components

| Component | Function |
|-----------|----------|
| Archive Server | Long-term image storage |
| Database Server | Metadata and index management |
| Web Server | Browser-based access |
| Workstation | Diagnostic viewing stations |
| Gateway | Protocol conversion, routing |
| Broker | Integration middleware |

### Radiology Workflow Steps

```
1. Order Entry (RIS/HIS) → HL7 ORM message
2. Patient Registration (RIS)
3. Modality Worklist Query (Modality → RIS)
4. Image Acquisition (Modality)
5. Image Storage (Modality → PACS) → DICOM C-STORE
6. Image Distribution (PACS → Workstations)
7. Interpretation (Radiologist)
8. Report Creation (RIS/Dictation)
9. Report Distribution (RIS → EHR) → HL7 ORU message
10. Billing (RIS)
```

### Worklist Terms

| Term | Description |
|------|-------------|
| **MWL** | Modality Worklist - scheduled procedures list |
| **MPPS** | Modality Performed Procedure Step - completion notification |
| **Accession Number** | Unique identifier for an imaging order |
| **MRN** | Medical Record Number - patient identifier |
| **Scheduled Procedure Step** | Planned imaging procedure |
| **Performed Procedure Step** | Completed imaging procedure |

---

## Interoperability Standards

### HL7 (Health Level Seven)

| Version | Description |
|---------|-------------|
| **HL7 v2.x** | Pipe-delimited message format, widely deployed |
| **HL7 v3** | XML-based, more complex |
| **HL7 FHIR** | RESTful APIs, JSON/XML, modern standard |
| **HL7 CDA** | Clinical Document Architecture for structured reports |

### Common HL7 v2 Messages

| Message | Type | Description |
|---------|------|-------------|
| ADT | Patient Admin | Admit/Discharge/Transfer |
| ORM | Order | New order request |
| ORU | Result | Observation/results |
| SIU | Scheduling | Appointment notifications |
| BAR | Billing | Financial transactions |
| MDM | Document | Document notifications |

### IHE (Integrating the Healthcare Enterprise) Profiles

| Profile | Description |
|---------|-------------|
| **SWF** | Scheduled Workflow - basic radiology workflow |
| **PIR** | Patient Information Reconciliation |
| **XDS-I.b** | Cross-enterprise Document Sharing for Imaging |
| **PDI** | Portable Data for Imaging (CD/DVD distribution) |
| **RWF** | Reporting Workflow |
| **IOCM** | Image Object Change Management |
| **ATNA** | Audit Trail and Node Authentication |

### FHIR Resources for Imaging

| Resource | Purpose |
|----------|---------|
| ImagingStudy | References to DICOM study |
| DiagnosticReport | Radiology report |
| Observation | Findings and measurements |
| ServiceRequest | Imaging order |
| Media | Non-DICOM images |
| Endpoint | DICOM/DICOMweb endpoints |

### DICOMweb Services

| Service | HTTP | Description |
|---------|------|-------------|
| **WADO-RS** | GET | Retrieve DICOM objects via REST |
| **STOW-RS** | POST | Store DICOM objects via REST |
| **QIDO-RS** | GET | Query DICOM objects via REST |
| **UPS-RS** | Various | Unified Procedure Step |

---

## AI/ML in Radiology

### Task Types

| Task | Description |
|------|-------------|
| **Classification** | Assign labels to images (benign/malignant, normal/abnormal) |
| **Detection** | Locate and identify lesions/structures with bounding boxes |
| **Segmentation** | Pixel-level labeling of structures |
| **Registration** | Align images from different times/modalities |
| **Reconstruction** | Generate images from raw sensor data |
| **Enhancement** | Improve image quality, denoise, super-resolution |

### Segmentation Types

| Type | Description |
|------|-------------|
| **Semantic** | Label each pixel with class (organ, lesion, background) |
| **Instance** | Distinguish individual objects of same class |
| **Panoptic** | Combined semantic + instance |

### Deep Learning Architectures

| Architecture | Use Case |
|--------------|----------|
| **CNN** | Convolutional Neural Network - image classification |
| **U-Net** | Medical image segmentation (encoder-decoder) |
| **ResNet** | Deep networks with skip connections |
| **DenseNet** | Dense connections for feature reuse |
| **VGG** | Simple deep architecture |
| **ViT** | Vision Transformer - attention-based |
| **YOLO** | Real-time object detection |
| **Mask R-CNN** | Instance segmentation |
| **GAN** | Generative Adversarial Network - synthesis |

### CAD Systems

| Type | Full Name | Function |
|------|-----------|----------|
| **CADe** | Computer-Aided Detection | Identifies potential abnormalities |
| **CADx** | Computer-Aided Diagnosis | Characterizes detected findings |
| **CADt** | Computer-Aided Triage | Prioritizes urgent cases |

### Performance Metrics

| Metric | Formula/Description |
|--------|---------------------|
| **Sensitivity** | TP / (TP + FN) - True positive rate |
| **Specificity** | TN / (TN + FP) - True negative rate |
| **PPV** | TP / (TP + FP) - Positive predictive value |
| **NPV** | TN / (TN + FN) - Negative predictive value |
| **AUC** | Area Under ROC Curve |
| **Dice** | 2×|A∩B| / (|A|+|B|) - Segmentation overlap |
| **IoU/Jaccard** | |A∩B| / |A∪B| - Intersection over Union |
| **Hausdorff** | Maximum boundary distance |

### AI/ML Terminology

| Term | Definition |
|------|------------|
| **Ground Truth** | Expert-annotated reference labels |
| **Inference** | Applying trained model to new data |
| **Epoch** | One complete pass through training data |
| **Batch Size** | Number of samples per gradient update |
| **Learning Rate** | Step size for weight updates |
| **Overfitting** | Model memorizes training data, poor generalization |
| **Data Augmentation** | Artificial training data expansion |
| **Transfer Learning** | Using pretrained weights from other tasks |
| **Fine-tuning** | Additional training on domain-specific data |
| **Explainability/XAI** | Interpretable AI decision-making |

---

## Radiomics & Texture Analysis

### Radiomics Workflow

```
1. Image Acquisition → 2. Segmentation (ROI) → 3. Preprocessing
       ↓
4. Feature Extraction → 5. Feature Selection → 6. Model Building
       ↓
7. Validation → 8. Clinical Application
```

### Feature Categories

#### First-Order (Histogram) Features

| Feature | Description |
|---------|-------------|
| Mean | Average intensity |
| Median | Middle value |
| Standard Deviation | Intensity spread |
| Skewness | Distribution asymmetry |
| Kurtosis | Distribution peakedness |
| Entropy | Intensity randomness |
| Energy | Sum of squared intensities |
| Range | Max - Min intensity |
| Percentiles | Intensity distribution quantiles |

#### Shape Features

| Feature | Description |
|---------|-------------|
| Volume | 3D size |
| Surface Area | Boundary extent |
| Sphericity | How sphere-like |
| Compactness | Volume/surface ratio |
| Elongation | Length/width ratio |
| Flatness | Major/minor axis ratio |
| Maximum 3D Diameter | Longest span |
| Mesh Volume | Volume from triangulated surface |

#### Texture Features (Second-Order)

**GLCM (Gray Level Co-occurrence Matrix):**
| Feature | Description |
|---------|-------------|
| Contrast | Local intensity variation |
| Correlation | Linear dependency |
| Energy (ASM) | Textural uniformity |
| Homogeneity | Local homogeneity |
| Entropy | Randomness |
| Dissimilarity | Intensity difference |
| Cluster Shade | Skewness and uniformity |
| Cluster Prominence | Asymmetry |

**GLRLM (Gray Level Run Length Matrix):**
| Feature | Description |
|---------|-------------|
| SRE | Short Run Emphasis |
| LRE | Long Run Emphasis |
| GLN | Gray Level Non-uniformity |
| RLN | Run Length Non-uniformity |
| RP | Run Percentage |
| LGLRE | Low Gray Level Run Emphasis |
| HGLRE | High Gray Level Run Emphasis |

**GLSZM (Gray Level Size Zone Matrix):**
| Feature | Description |
|---------|-------------|
| SAE | Small Area Emphasis |
| LAE | Large Area Emphasis |
| LGLZE | Low Gray Level Zone Emphasis |
| HGLZE | High Gray Level Zone Emphasis |
| SZN | Size Zone Non-uniformity |
| ZP | Zone Percentage |

**NGTDM (Neighborhood Gray Tone Difference Matrix):**
| Feature | Description |
|---------|-------------|
| Coarseness | Spatial rate of change |
| Contrast | Dynamic range and spatial frequency |
| Busyness | Spatial frequency of intensity changes |
| Complexity | Non-uniformity and rapid changes |
| Strength | Primitive size and intensity |

### Higher-Order Features

| Type | Description |
|------|-------------|
| **Wavelet** | Multi-resolution analysis features |
| **Laplacian of Gaussian** | Edge detection at multiple scales |
| **Gabor** | Texture and orientation analysis |
| **Fractal** | Self-similarity at different scales |
| **LBP** | Local Binary Patterns |

### Radiomics Software Tools

| Tool | Type | Language |
|------|------|----------|
| **PyRadiomics** | Open source | Python |
| **IBEX** | Open source | MATLAB |
| **LIFEx** | Open source | Java |
| **3D Slicer** | Platform | C++/Python |
| **TexRAD** | Commercial | - |

### IBSI (Image Biomarker Standardization Initiative)

Provides standardized definitions for radiomic features ensuring reproducibility across software implementations.

---

## Clinical Terminology

### Report Structure Terms

| Term | Description |
|------|-------------|
| **Clinical History** | Patient symptoms, relevant medical history |
| **Indication** | Reason for examination |
| **Technique** | Imaging protocol used |
| **Comparison** | Prior studies referenced |
| **Findings** | Objective observations |
| **Impression** | Summary and differential diagnoses |
| **Recommendation** | Suggested follow-up actions |

### Anatomical Descriptors

| Term | Meaning |
|------|---------|
| **Proximal** | Closer to trunk/origin |
| **Distal** | Further from trunk/origin |
| **Medial** | Toward midline |
| **Lateral** | Away from midline |
| **Superficial** | Closer to surface |
| **Deep** | Further from surface |
| **Ipsilateral** | Same side |
| **Contralateral** | Opposite side |

### Lesion Descriptors

| Term | Description |
|------|-------------|
| **Hypodense** | Lower density than surroundings (CT) |
| **Hyperdense** | Higher density than surroundings (CT) |
| **Hypointense** | Lower signal than surroundings (MRI) |
| **Hyperintense** | Higher signal than surroundings (MRI) |
| **Isodense/Isointense** | Same as surroundings |
| **Enhancing** | Increases signal with contrast |
| **Non-enhancing** | No change with contrast |
| **Homogeneous** | Uniform appearance |
| **Heterogeneous** | Variable appearance |

### Size and Shape

| Term | Description |
|------|-------------|
| **Nodule** | Small, rounded density |
| **Mass** | Larger (usually >3cm) lesion |
| **Cyst** | Fluid-filled structure |
| **Calcification** | Calcium deposit |
| **Spiculated** | Star-shaped margins |
| **Lobulated** | Multi-lobed margins |
| **Circumscribed** | Well-defined margins |
| **Ill-defined** | Indistinct margins |

### Reporting Standardization Systems

| System | Domain |
|--------|--------|
| **BI-RADS** | Breast Imaging |
| **Lung-RADS** | Lung CT screening |
| **LI-RADS** | Liver imaging |
| **PI-RADS** | Prostate MRI |
| **TI-RADS** | Thyroid ultrasound |
| **O-RADS** | Ovarian/adnexal |
| **Fleischner** | Lung nodule follow-up |

### Controlled Vocabularies

| Vocabulary | Purpose |
|------------|---------|
| **RadLex** | Radiology-specific lexicon |
| **SNOMED CT** | Clinical terminology |
| **ICD-10** | Diagnosis coding |
| **CPT** | Procedure coding |
| **LOINC** | Laboratory/clinical observations |

---

## Data Processing Concepts

### Preprocessing Operations

| Operation | Description |
|-----------|-------------|
| **Resampling** | Changing voxel dimensions |
| **Registration** | Aligning images to common space |
| **Normalization** | Standardizing intensity ranges |
| **Skull Stripping** | Removing non-brain tissue |
| **Bias Field Correction** | Removing intensity inhomogeneity |
| **Denoising** | Reducing image noise |
| **Windowing/Leveling** | Adjusting display contrast |

### Registration Types

| Type | Description |
|------|-------------|
| **Rigid** | Translation + rotation only (6 DOF) |
| **Affine** | Rigid + scaling + shearing (12 DOF) |
| **Non-rigid/Deformable** | Local warping, many DOF |
| **Intrasubject** | Same patient, different times |
| **Intersubject** | Different patients |

### Standard Spaces

| Space | Description |
|-------|-------------|
| **MNI-152** | Montreal Neurological Institute template |
| **Talairach** | Historical brain coordinate system |
| **Native** | Original scanner coordinates |
| **Scanner** | Physical scanner coordinate system |

### Interpolation Methods

| Method | Quality | Speed |
|--------|---------|-------|
| Nearest Neighbor | Lowest | Fastest |
| Linear/Trilinear | Medium | Fast |
| Cubic/Tricubic | High | Medium |
| B-spline | Highest | Slowest |
| Sinc | Highest | Slowest |

### File Size Estimation

```
Uncompressed size (bytes) = 
  X × Y × Z × T × (bits_per_voxel / 8)

Example: 256 × 256 × 180 × 1 × (16/8) = 23.6 MB
```

---

## Quality & Validation

### Image Quality Metrics

| Metric | Description |
|--------|-------------|
| **SNR** | Signal-to-Noise Ratio |
| **CNR** | Contrast-to-Noise Ratio |
| **MTF** | Modulation Transfer Function |
| **FWHM** | Full Width at Half Maximum |
| **Ghosting** | Motion artifact measure |
| **Geometric Distortion** | Spatial accuracy |

### Data Quality Checks

| Check | Description |
|-------|-------------|
| **Header Validation** | DICOM conformance, required tags |
| **Pixel Data Integrity** | No corruption, correct dimensions |
| **Orientation Consistency** | Correct patient position |
| **Series Completeness** | All expected images present |
| **De-identification** | PHI removal verification |

### Common Artifacts

| Artifact | Cause | Modality |
|----------|-------|----------|
| Motion blur | Patient movement | All |
| Aliasing/Wrap | FOV too small | MRI |
| Chemical shift | Fat/water interface | MRI |
| Susceptibility | Metal, air-tissue | MRI |
| Beam hardening | Dense materials | CT |
| Ring artifacts | Detector malfunction | CT |
| Streak artifacts | Metal implants | CT |

### Anonymization/De-identification

| Level | Removed |
|-------|---------|
| **Basic** | Patient name, ID, dates |
| **HIPAA Safe Harbor** | 18 identifiers |
| **Full** | All identifying information including burned-in pixels |

| DICOM Tags to Remove |
|---------------------|
| (0010,0010) Patient Name |
| (0010,0020) Patient ID |
| (0010,0030) Birth Date |
| (0008,0050) Accession Number |
| (0008,0080) Institution Name |
| (0008,1030) Study Description |
| (0008,0090) Referring Physician |

---

## Quick Reference: Common Python Libraries

| Library | Purpose |
|---------|---------|
| **pydicom** | DICOM file I/O |
| **nibabel** | NIfTI/BIDS file I/O |
| **SimpleITK** | Image processing, registration |
| **nilearn** | Neuroimaging ML |
| **PyRadiomics** | Radiomics feature extraction |
| **TorchIO** | Medical imaging data augmentation |
| **MONAI** | Medical imaging deep learning |
| **highdicom** | Advanced DICOM objects (SR, SEG) |
| **dcm2niix** | DICOM to NIfTI conversion |
| **ANTsPy** | Registration, segmentation |
| **FSL/FreeSurfer** | Neuroimaging pipelines |

---

## Quick Reference: Command Line Tools

```bash
# DICOM to NIfTI conversion
dcm2niix -o output_dir -f %p_%s input_dicom_folder

# View NIfTI header
fslhd image.nii.gz
nib-ls image.nii.gz

# Reorient to standard
fslreorient2std input.nii.gz output.nii.gz

# BIDS validation
bids-validator /path/to/bids/dataset

# DICOM dump
dcmdump file.dcm
```
