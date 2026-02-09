"""Quick verification of obs_meta schema after migration."""

import json

from radiobject import RadiObject, S3Config, configure

configure(s3=S3Config(region="us-east-2"))

for name, uri in [
    ("MSD Lung", "s3://souzy-scratch/msd-lung/radiobject-2mm"),
    ("BraTS", "s3://souzy-scratch/radiobject/brats-tutorial"),
]:
    try:
        radi = RadiObject(uri)
    except Exception as e:
        print(f"\n=== {name} ({uri}) ===")
        print(f"  NOT AVAILABLE: {e}")
        continue

    df = radi.obs_meta.read()
    print(f"\n=== {name} ({uri}) ===")
    print(f"  obs_meta columns: {list(df.columns)}")
    print(f"  obs_meta shape: {df.shape}")
    print(f"  index_columns: {radi.obs_meta.index_columns}")
    print(f"  'obs_id' in columns: {'obs_id' in df.columns}")
    print(f"  'obs_ids' in columns: {'obs_ids' in df.columns}")

    if "obs_ids" in df.columns:
        row = df.iloc[0]
        sid = row["obs_subject_id"]
        obs_ids = json.loads(row["obs_ids"])
        print(f"  Sample obs_ids for {sid}: {obs_ids}")

    print(f"  Collections: {radi.collection_names}")
    print(f"  Subjects: {len(radi)}")

    # Validate
    try:
        radi.validate()
        print("  validate(): PASSED")
    except Exception as e:
        print(f"  validate(): FAILED - {e}")
