# Cloud Setup

RadiObject works locally by default. Cloud storage via S3 is optional for scaling to larger datasets.

## AWS S3 Credentials

RadiObject inherits AWS credentials from boto3's credential chain.

**Environment variables** (recommended for CI/containers):

```bash
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
```

**AWS CLI** (recommended for local development):

```bash
aws configure
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
RadiObject.from_images(
    "s3://your-bucket/new-dataset",
    images={"CT": "./local/images"},
    obs_meta=df,
)
```

## S3Config

Configure S3 region and performance settings:

```python
from radiobject import configure, S3Config

configure(s3=S3Config(
    region="us-west-2",
    max_parallel_ops=16,          # Concurrent S3 operations (default: 8)
    multipart_part_size_mb=100,   # Multipart upload chunk size (default: 50)
))
```

For custom endpoints (MinIO, LocalStack):

```python
configure(s3=S3Config(endpoint="http://localhost:9000"))
```

For all S3Config options, see [Configuration: S3Config](../reference/configuration.md#s3config).

## Performance Tips

1. **Same-region access**: Place your S3 bucket in the same region as your compute
2. **Partial reads shine on S3**: Tile-based access is 35x faster than full-volume reads from S3
3. **Parallel uploads**: Configure `max_parallel_ops` for write-heavy workloads
4. **Batch operations**: Use streaming writers for large dataset ingestion to S3
