#!/usr/bin/env python3
"""Download BraTS sample data for RadiObject tutorials.

Downloads a subset of the BraTS 2021 dataset from the Medical Segmentation Decathlon (MSD).
This provides enough data to run notebooks 00-04 without S3 access.

Usage:
    python scripts/download_tutorial_data.py

The script will:
    1. Download Task01_BrainTumour.tar from MSD (~1.5GB)
    2. Extract 5 sample subjects
    3. Place NIfTI files in ./data/brats-tutorial/
    4. Clean up the full archive to save space

Requirements:
    pip install radiobject[download]  # or: pip install requests
"""

import hashlib
import shutil
import sys
import tarfile
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: requests package required. Install with: pip install radiobject[download]")
    sys.exit(1)

MSD_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
EXPECTED_MD5 = None  # MSD doesn't publish checksums; skip verification
N_SAMPLES = 5
DATA_DIR = Path(__file__).parent.parent / "data" / "brats-tutorial"
TEMP_DIR = Path(__file__).parent.parent / "data" / ".tmp_download"


def download_with_progress(url: str, dest: Path) -> None:
    """Download file with progress bar."""
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192
    downloaded = 0

    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=block_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                pct = (downloaded / total_size) * 100
                bar = "=" * int(pct // 2) + ">" + " " * (50 - int(pct // 2))
                print(f"\rDownloading: [{bar}] {pct:.1f}%", end="", flush=True)
    print()


def extract_samples(tar_path: Path, n_samples: int) -> list[str]:
    """Extract N sample subjects from MSD archive."""
    subjects = []

    with tarfile.open(tar_path, "r") as tar:
        # Find unique subject IDs from training images
        members = tar.getnames()
        training_files = [m for m in members if "imagesTr/" in m and m.endswith(".nii.gz")]

        # Extract subject IDs (format: BRATS_001.nii.gz)
        subject_ids = sorted(set(Path(f).stem.replace(".nii", "") for f in training_files))
        selected = subject_ids[:n_samples]

        print(f"Extracting {n_samples} subjects: {', '.join(selected)}")

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        for subject_id in selected:
            # Extract training image
            img_name = f"Task01_BrainTumour/imagesTr/{subject_id}.nii.gz"
            seg_name = f"Task01_BrainTumour/labelsTr/{subject_id}.nii.gz"

            for name, dest_subdir in [(img_name, "images"), (seg_name, "labels")]:
                try:
                    member = tar.getmember(name)
                    dest_dir = DATA_DIR / dest_subdir
                    dest_dir.mkdir(parents=True, exist_ok=True)

                    # Extract to temp then move (tarfile extracts with full path)
                    tar.extract(member, TEMP_DIR)
                    src = TEMP_DIR / name
                    dst = dest_dir / f"{subject_id}.nii.gz"
                    shutil.move(str(src), str(dst))
                    print(f"  Extracted: {dst.relative_to(DATA_DIR.parent.parent)}")
                except KeyError:
                    print(f"  Warning: {name} not found in archive")

            subjects.append(subject_id)

    return subjects


def verify_download(tar_path: Path) -> bool:
    """Verify download integrity."""
    if EXPECTED_MD5 is None:
        return True

    print("Verifying checksum...", end=" ", flush=True)
    md5 = hashlib.md5()
    with open(tar_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)

    if md5.hexdigest() == EXPECTED_MD5:
        print("OK")
        return True
    else:
        print("FAILED")
        return False


def cleanup(tar_path: Path) -> None:
    """Remove temporary files."""
    if tar_path.exists():
        tar_path.unlink()
        print(f"Cleaned up: {tar_path}")

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)


def main() -> None:
    """Download and extract BraTS tutorial data."""
    print("=" * 60)
    print("RadiObject Tutorial Data Downloader")
    print("=" * 60)
    print()

    # Check if data already exists
    if DATA_DIR.exists() and any(DATA_DIR.glob("images/*.nii.gz")):
        existing = list(DATA_DIR.glob("images/*.nii.gz"))
        print(f"Data already exists at {DATA_DIR}")
        print(f"Found {len(existing)} images. Delete the directory to re-download.")
        return

    tar_path = TEMP_DIR / "Task01_BrainTumour.tar"

    try:
        # Download
        print("Downloading MSD BrainTumour dataset (~1.5GB)...")
        print(f"Source: {MSD_URL}")
        print()
        download_with_progress(MSD_URL, tar_path)

        # Verify
        if not verify_download(tar_path):
            print("Download verification failed. Please retry.")
            cleanup(tar_path)
            sys.exit(1)

        # Extract samples
        print()
        subjects = extract_samples(tar_path, N_SAMPLES)

        # Cleanup
        print()
        cleanup(tar_path)

        # Summary
        print()
        print("=" * 60)
        print("Download complete!")
        print("=" * 60)
        print()
        print(f"Location: {DATA_DIR}")
        print(f"Subjects: {len(subjects)}")
        print()
        print("Next steps:")
        print("  1. cd notebooks")
        print("  2. jupyter notebook 00_ingest_brats.ipynb")
        print()

    except KeyboardInterrupt:
        print("\n\nDownload cancelled.")
        cleanup(tar_path)
        sys.exit(1)
    except requests.RequestException as e:
        print(f"\nDownload failed: {e}")
        cleanup(tar_path)
        sys.exit(1)


if __name__ == "__main__":
    main()
