# S3 Storage (Optional)

RadiObject works locally by default. Cloud storage via S3 is an optional feature for scaling to larger datasets.

## Setup

RadiObject inherits AWS credentials from boto3's credential chain:

**Environment variables** (recommended for CI/containers):
```bash
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-2
```

**AWS CLI** (recommended for local development):
```bash
aws configure
# Enter your credentials when prompted
```

**Named profiles**:
```bash
# In ~/.aws/credentials
[myprofile]
aws_access_key_id = AKIA...
aws_secret_access_key = ...

# Then set the profile
export AWS_PROFILE=myprofile
```

## Usage

Use S3 URIs anywhere you'd use a local path:

```python
from radiobject import RadiObject

# Read from S3
radi = RadiObject("s3://your-bucket/your-dataset")

# Write to S3
RadiObject.from_niftis(
    "s3://your-bucket/new-dataset",
    images_dir="./local/images",
    metadata_df=df
)
```

## Configuration

Configure S3 region and other settings:

```python
from radiobject import configure
from radiobject.ctx import S3Config

configure(s3=S3Config(region="us-west-2"))
```

## Performance Tips

1. **Use the same region**: Place your S3 bucket in the same region as your compute
2. **Partial reads shine on S3**: RadiObject's tile-based access is 5-16x faster than NIfTI for partial reads
3. **Full volume reads are slower**: S3 has higher latency than local disk; batch multiple operations
4. **Consider S3 Transfer Acceleration** for cross-region access

See [Benchmarks](BENCHMARKS.md) for detailed S3 vs local performance comparisons.
