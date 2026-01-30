"""Notebook configuration - edit URIs to switch between S3 and local storage."""

# BraTS tutorial data (notebooks 01-04)
# Options:
#   S3:    "s3://souzy-scratch/radiobject/brats-tutorial"
#   Local: "./data/brats-tutorial"
BRATS_URI = "s3://souzy-scratch/radiobject/brats-tutorial"

# MSD Lung data (notebooks 05-06)
# Options:
#   S3:    "s3://souzy-scratch/msd-lung/radiobject-2mm"
#   Local: "./data/msd-lung"
MSD_LUNG_URI = "s3://souzy-scratch/msd-lung/radiobject-2mm"

# S3 region (ignored for local paths)
# Note: The souzy-scratch bucket is in us-east-2
S3_REGION = "us-east-2"
