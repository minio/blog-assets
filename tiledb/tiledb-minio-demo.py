import tiledb
import numpy as np

# MinIO keys
minio_key = "minioadmin"
minio_secret = "minioadmin"

# The configuration object with MinIO keys
config = tiledb.Config()
config["vfs.s3.aws_access_key_id"] = minio_key
config["vfs.s3.aws_secret_access_key"] = minio_secret
config["vfs.s3.scheme"] = "http"
config["vfs.s3.region"] = ""
config["vfs.s3.endpoint_override"] = "play.min.io:9000"
config["vfs.s3.use_virtual_addressing"] = "false"

# Create TileDB config context
ctx = tiledb.Ctx(config)

# The MinIO bucket URI path of tiledb demo
array_minio = "s3://tiledb-demo/tiledb_minio_demo/"

tiledb.from_numpy(array_minio, np.array([1.0, 2.0, 3.0]), ctx=tiledb.Ctx(config))

with tiledb.open(array_minio, ctx=tiledb.Ctx(config)) as A:
    print(A[:])