"""Notebook configuration - edit URIs to switch between local and S3 storage."""

# BraTS tutorial data (notebooks 00-04)
# Run `python scripts/download_tutorial_data.py` first for local data
# Options:
#   Local: "./data/brats-tutorial" (default - run download script first)
#   S3:    "s3://souzy-scratch/radiobject/brats-tutorial"
BRATS_URI = "./data/brats-tutorial"

# MSD Lung data (notebooks 05-06)
# Options:
#   Local: "./data/msd-lung"
#   S3:    "s3://souzy-scratch/msd-lung/radiobject-2mm"
MSD_LUNG_URI = "s3://souzy-scratch/msd-lung/radiobject-2mm"

# S3 region (only needed if using S3 URIs above)
# Note: The souzy-scratch bucket is in us-east-2
S3_REGION = "us-east-2"
