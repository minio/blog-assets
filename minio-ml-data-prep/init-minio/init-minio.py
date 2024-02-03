from minio import Minio
from minio.error import S3Error

minio_client = Minio(
    "minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

def create_bucket_if_not_exists(bucket_name):
    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            print(f"Bucket '{bucket_name}' created successfully.")
        else:
            print(f"Bucket '{bucket_name}' already exists.")
    except S3Error as e:
        print(f"Error: {e}")

# Create raw and clean buckets if they don't exist
create_bucket_if_not_exists("raw")
create_bucket_if_not_exists("clean")
