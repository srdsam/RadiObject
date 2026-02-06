# Datasets

## Download Commands

```bash
# List available datasets
python scripts/download_dataset.py --list

# Download for tests (NIfTI + DICOM)
python scripts/download_dataset.py --all-tests

# Download specific dataset
python scripts/download_dataset.py msd-brain-tumour
python scripts/download_dataset.py msd-lung
python scripts/download_dataset.py nsclc-radiomics

# Download without AWS credentials (public sources only)
python scripts/download_dataset.py msd-brain-tumour --public
```

## Available Datasets

| Dataset | Format | Size | Samples | Use Case |
|---------|--------|------|---------|----------|
| `msd-brain-tumour` | NIfTI | ~1.5GB | 10 | Tests, tutorials 1-5 |
| `msd-lung` | NIfTI | ~8.5GB | 63 | ML tutorials 6-7 |
| `nsclc-radiomics` | DICOM | ~500MB | 10 | DICOM parsing tests |

## External Data Sources

For additional radiology datasets, these sources offer programmatic access:

| Source | Format | Access |
|--------|--------|--------|
| [Medical Segmentation Decathlon](http://medicaldecathlon.com/) | NIfTI | `aws s3 sync --no-sign-request s3://msd-for-monai/ ./` |
| [TCIA](https://www.cancerimagingarchive.net/) | DICOM | `pip install tcia_utils` |
| [OpenNeuro](https://openneuro.org/) | NIfTI/BIDS | `pip install openneuro-py` |
| [IXI Dataset](https://brain-development.org/ixi-dataset/) | NIfTI | Direct wget |
