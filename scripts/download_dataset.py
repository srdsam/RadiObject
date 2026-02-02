#!/usr/bin/env python3
"""CLI for downloading RadiObject datasets from public sources or S3."""

from __future__ import annotations

import argparse
import sys


def list_datasets() -> None:
    """Print available datasets in table format."""
    from radiobject.data.registry import DATASETS

    print("=" * 70)
    print("Available Datasets")
    print("=" * 70)
    print()

    for name, info in DATASETS.items():
        public = "Yes" if info.public_source else "TCIA API"
        size = f"{info.size_mb}MB" if info.size_mb else "Unknown"

        print(f"  {name}")
        print(f"    Name:        {info.name}")
        print(f"    Format:      {info.format.value.upper()}")
        print(f"    Size:        ~{size}")
        print(f"    Samples:     {info.n_samples or 'Unknown'}")
        print(f"    Public:      {public}")
        print(f"    Description: {info.description}")
        print()

    print("Usage:")
    print("  python scripts/download_dataset.py <dataset-name>")
    print("  python scripts/download_dataset.py --all-tests")
    print()


def download_dataset(name: str, force: bool = False, prefer_public: bool = False) -> None:
    """Download a single dataset by name."""
    from radiobject.data import get_dataset

    print(f"Downloading dataset: {name}")
    print()

    prefer_s3 = not prefer_public
    path = get_dataset(name, prefer_s3=prefer_s3, force_download=force)

    print()
    print(f"Dataset available at: {path}")


def download_all_tests(force: bool = False, prefer_public: bool = False) -> None:
    """Download all datasets required for test suite."""
    test_datasets = ["msd-brain-tumour", "nsclc-radiomics"]

    print("=" * 70)
    print("Downloading All Test Datasets")
    print("=" * 70)
    print()

    for name in test_datasets:
        print(f"[{test_datasets.index(name) + 1}/{len(test_datasets)}] {name}")
        print("-" * 40)
        try:
            download_dataset(name, force=force, prefer_public=prefer_public)
        except Exception as e:
            print(f"  Warning: Failed to download {name}: {e}")
            print("  Skipping...")
        print()

    print("=" * 70)
    print("Download complete!")
    print()
    print("Run tests with:")
    print("  uv run pytest test/ -v")
    print()


def main() -> None:
    """CLI entry point for dataset downloads."""
    parser = argparse.ArgumentParser(
        description="Download RadiObject datasets from public sources or S3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                    List available datasets
  %(prog)s msd-brain-tumour          Download BraTS brain tumor data
  %(prog)s nsclc-radiomics           Download NSCLC DICOM data
  %(prog)s --all-tests               Download all test datasets
  %(prog)s msd-lung --public         Download MSD Lung (public source)
        """,
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        help="Dataset name to download (use --list to see available)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available datasets",
    )
    parser.add_argument(
        "--all-tests",
        action="store_true",
        help="Download all datasets needed for tests",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download even if cached",
    )
    parser.add_argument(
        "--public",
        "-p",
        action="store_true",
        help="Prefer public sources over S3",
    )

    args = parser.parse_args()

    # Handle --list
    if args.list:
        list_datasets()
        return

    # Handle --all-tests
    if args.all_tests:
        download_all_tests(force=args.force, prefer_public=args.public)
        return

    # Handle specific dataset
    if args.dataset:
        from radiobject.data.registry import DATASETS

        if args.dataset not in DATASETS:
            print(f"Error: Unknown dataset '{args.dataset}'")
            print()
            print("Available datasets:")
            for name in DATASETS:
                print(f"  - {name}")
            print()
            print("Use --list for details.")
            sys.exit(1)

        try:
            download_dataset(args.dataset, force=args.force, prefer_public=args.public)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        return

    # No arguments provided
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
