import os
from pathlib import Path

import boto3
import pandas as pd

s3_uri = "s3://sports-footage-autotrim-bucket/feature_extractions/vq3CZAx3GnM_019_features.parquet"

bucket = s3_uri.split("/")[2]
key = "/".join(s3_uri.split("/")[3:])

env_path = Path(".env")
if env_path.is_file():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k and k not in os.environ:
            os.environ[k] = v

out = Path("/tmp/vq3CZAx3GnM_019_features.parquet")

s3 = boto3.client(
    "s3",
    region_name=os.environ.get("AWS_REGION", "us-west-2"),
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
)

s3.download_file(bucket, key, str(out))

print(f"Downloaded to {out}")

df = pd.read_parquet(out)
print("Columns:", list(df.columns))
print("dtypes:\n", df.dtypes)
print("head:\n", df.head(3))
