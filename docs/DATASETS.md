# Open-source radiology datasets for programmatic download

The best options for immediate programmatic access to DICOM/NIfTI radiology data are **Medical Segmentation Decathlon** (via AWS S3, no auth), **TCIA public collections** (via REST API or `tcia_utils`), and **OpenNeuro/IXI** for NIfTI brain imaging. These sources require no approval process, support command-line downloads, and offer permissive licenses suitable for development and testing.

For testing a new radiology data structure, **start with Medical Segmentation Decathlon** for NIfTI across 10 modality/organ combinations (single `aws s3 sync` command, ~42GB), then **TCIA's LIDC-IDRI** for diverse DICOM CT data via Python. The IXI dataset provides the simplest NIfTI brain MRI access via direct wget URLs.

## Instant-access datasets requiring zero registration

These datasets can be downloaded immediately with a single command and no account creation.

**Medical Segmentation Decathlon** offers the fastest path to varied NIfTI data. Hosted on AWS S3, it includes **10 segmentation tasks** spanning brain tumors, liver, lung, pancreas, cardiac, prostate, hepatic vessels, spleen, colon, and liver tumors—covering both CT and MRI modalities in preprocessed NIfTI format. Total size is approximately **42GB**, licensed under CC-BY-SA 4.0.

```bash
# Download everything (~42GB)
aws s3 sync --no-sign-request s3://msd-for-monai/ ./msd-data/

# Download specific task (brain tumors only)
aws s3 sync --no-sign-request s3://msd-for-monai/Task01_BrainTumour/ ./brain/
```

**IXI Dataset** provides the simplest direct download for healthy brain MRI. Approximately 600 subjects scanned at three UK sites offer T1, T2, PD-weighted images, MRA, and DTI data—all in NIfTI format under CC BY-SA 3.0 license:

```bash
wget http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar
wget http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T2.tar
wget http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-DTI.tar
```

**NCI Imaging Data Commons** aggregates over **85TB** of harmonized DICOM cancer imaging from TCIA, TCGA, and other NCI projects on AWS S3. More than 95% carries CC-BY licensing:

```bash
aws s3 ls --no-sign-request s3://idc-open-data/
aws s3 cp --no-sign-request s3://idc-open-data/[path] ./
```

## TCIA offers the richest DICOM radiology collection

The Cancer Imaging Archive provides the most extensive programmatic access to DICOM radiology data through its **REST API (v4)** and the official `tcia_utils` Python package. Public collections require no authentication.

**Quick setup and download workflow:**

```python
pip install tcia_utils
from tcia_utils import nbia

# List all available collections
collections = nbia.getCollections()

# Get series metadata for a specific collection
series = nbia.getSeries(collection="LIDC-IDRI", modality="CT")

# Download (saves to ./tciaDownload)
nbia.downloadSeries(series, path="./downloads")
```

| Dataset | Modality | Subjects | Size | License | Best for |
|---------|----------|----------|------|---------|----------|
| **LIDC-IDRI** | CT (lung) | 1,018 | ~125GB | CC BY 3.0 | Lung nodule analysis |
| **CBIS-DDSM** | Mammography | 1,566 | 164GB | CC BY 3.0 | Breast imaging |
| **CT-ORG** | CT | 140 | ~15GB | CC BY 3.0 | Multi-organ segmentation |
| **RIDER Lung PET-CT** | PET/CT | 63 | ~5GB | CC BY 3.0 | Multimodal testing |
| **UPENN-GBM** | MRI | 630 | ~80GB | CC BY 4.0 | Brain MRI with NIfTI |

The **NBIA Data Retriever CLI** handles bulk downloads via manifest files (`.tcia`), useful for downloading entire collections:

```bash
# Linux installation
sudo dpkg -i nbia-data-retriever_4.4.3-1_amd64.deb

# Download from manifest
/opt/nbia-data-retriever/bin/nbia-data-retriever --cli manifest.tcia -d /output -v
```

**UPENN-GBM** is particularly valuable as it provides both DICOM source data and preprocessed NIfTI volumes with segmentation labels—ideal for testing both formats simultaneously.

## Neuroimaging repositories excel at NIfTI and BIDS formats

**OpenNeuro** hosts thousands of BIDS-formatted brain imaging datasets under CC0 license. The platform supports multiple programmatic access methods:

```bash
# DataLad (recommended - clone metadata, then selectively download)
datalad clone https://github.com/OpenNeuroDatasets/ds000001.git
cd ds000001
datalad get sub-01/anat/*.nii.gz

# OpenNeuro CLI (Deno-based)
deno install -A --global jsr:@openneuro/cli -n openneuro
openneuro download ds000001 ./output

# Python
pip install openneuro-py
import openneuro as on
on.download(dataset='ds000246', target_dir='./bids')
```

**DataLad** provides the most flexible neuroimaging access, supporting version control, selective file retrieval, and easy updates without full re-downloads.

**OASIS brain datasets** require free registration but offer excellent longitudinal Alzheimer's imaging data. OASIS-3 includes **1,378 participants** with T1w, T2w, FLAIR, diffusion MRI, and PET scans in NIfTI format (ODC-BY license). Access via NITRC-IR at nitrc.org after accepting data use terms.

## NIH Clinical Center datasets cover X-ray and CT

**NIH ChestX-ray14** (112,120 images, 14 disease labels) exists in both PNG and Google-converted DICOM formats:

```bash
# PNG via Google Cloud Storage (requester pays)
gsutil -u YOUR_PROJECT cp -r gs://gcs-public-data--healthcare-nih-chest-xray/png/ ./

# DICOM version
gsutil -u YOUR_PROJECT cp -r gs://gcs-public-data--healthcare-nih-chest-xray/dicom/ ./
```

**NIH DeepLesion** provides **32,735 lesions** across diverse anatomical locations in CT format (~220GB). Available via NIH Box (https://nihcc.app.box.com/v/DeepLesion) or Academic Torrents for bulk download.

## Kaggle competitions offer annotated challenge datasets

RSNA challenges provide high-quality labeled DICOM data. The Kaggle API enables programmatic downloads after one-time setup:

```bash
pip install kaggle
# Place API token in ~/.kaggle/kaggle.json

# Download RSNA pneumonia detection (~30,000 chest X-rays, DICOM)
kaggle competitions download -c rsna-pneumonia-detection-challenge

# Intracranial hemorrhage (~750,000 head CT images, DICOM)
kaggle competitions download -c rsna-intracranial-hemorrhage-detection
```

Note that competition datasets often carry non-commercial research licenses, so verify terms before production use.

## Converting between DICOM and NIfTI formats

Since TCIA and clinical sources primarily provide DICOM while neuroimaging repositories favor NIfTI, conversion tools bridge the gap:

```bash
# dcm2niix (fastest, most reliable)
pip install dcm2niix
dcm2niix -o output_dir -f %p_%s input_dicom_folder

# Python alternative
pip install dicom2nifti
python -c "import dicom2nifti; dicom2nifti.convert_directory('dicom_dir', 'output')"
```

Several TCIA "Analysis Results" provide pre-converted NIfTI data, including BraTS brain tumor segmentations and UPENN-GBM.

## Practical recommendations for data structure testing

For comprehensive testing across formats and modalities, this acquisition sequence maximizes coverage with minimal friction:

1. **Start immediately** with Medical Segmentation Decathlon via AWS (`aws s3 sync --no-sign-request`) for NIfTI across 10 tasks
2. **Add DICOM variety** using `tcia_utils` to download LIDC-IDRI (lung CT), CT-ORG (multi-organ), and CBIS-DDSM (mammography)
3. **Include brain imaging** via IXI direct wget for healthy subjects, OpenNeuro for task/resting fMRI
4. **Test edge cases** with RIDER PET-CT for multimodal DICOM and NIH ChestX-ray14 for 2D formats

| Use Case | Recommended Dataset | Access Method |
|----------|-------------------|---------------|
| Quick NIfTI test | Medical Segmentation Decathlon | `aws s3 sync --no-sign-request` |
| DICOM parsing | LIDC-IDRI via TCIA | `tcia_utils` Python |
| BIDS structure | OpenNeuro ds000001 | DataLad clone |
| Mixed formats | UPENN-GBM | TCIA (has both) |
| Large-scale stress test | NCI IDC | AWS S3 buckets |

All recommended datasets permit development/testing use. CC-BY variants require attribution in publications; CC0 (OpenNeuro) has no restrictions. Avoid UK Biobank and ADNI for quick testing—both require multi-week approval processes.