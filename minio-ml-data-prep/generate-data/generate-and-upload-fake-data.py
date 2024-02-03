# generate_and_upload_to_minio.py

from faker import Faker
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import random
import string
import os
from minio import Minio
from minio.error import S3Error

fake = Faker()

# MinIO configuration
minio_client = Minio(
    "minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)


def generate_fake_data():
    data = {
        "name": fake.name(),
        "email": fake.email(),
        "address": fake.address(),
        "phone_number": fake.phone_number(),
        "ssn": fake.ssn(),  # PII field
        "random_string": ''.join(random.choices(string.ascii_letters + string.digits, k=10)),
        "random_number": random.randint(1, 100),
        "employee_details": {
            "position": fake.job(),
            "department": fake.job(),
            "salary": round(random.uniform(40000, 100000), 2),
            "hire_date": str(fake.date_this_decade()),  # Convert to string
        },
    }
    return json.dumps(data)


def save_as_parquet(fake_data, file_path):
    df = pd.DataFrame([json.loads(fake_data)])
    table = pa.Table.from_pandas(df)

    # Create the 'data' directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Save the Parquet file in the 'data' directory
    parquet_file_path = os.path.join("data", file_path)

    pq.write_table(table, parquet_file_path)
    print(f"Saved Parquet file to {parquet_file_path}")
    return parquet_file_path


def upload_to_minio(bucket_name, local_file_path, object_name):
    file_size = os.path.getsize(local_file_path)
    with open(local_file_path, 'rb') as data:
        try:
            minio_client.put_object(bucket_name, object_name, data, file_size, content_type="application/octet-stream")
            print(f"Uploaded {object_name} to MinIO bucket: {bucket_name}")
        except S3Error as e:
            print(f"Error uploading {object_name} to MinIO bucket: {e}")


def generate_and_upload_to_minio(bucket_name, num_files=10):
    for i in range(num_files):
        object_name = f"data_{i + 1}.parquet"
        fake_data = generate_fake_data()
        local_file_path = save_as_parquet(fake_data, object_name)
        upload_to_minio(bucket_name, local_file_path, object_name)


# Specify the MinIO raw bucket name
raw_bucket_name = "raw"

# Generate and upload fake data as Parquet files to the MinIO raw bucket
generate_and_upload_to_minio(raw_bucket_name)
