# scrub_and_upload_to_minio.py

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from minio import Minio
from minio.error import S3Error

# MinIO configuration
minio_client = Minio(
    "minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

def scrub_pii(data):
    # Remove personally identifiable information (PII)
    data["name"] = "Scrubbed Name"
    data["email"] = "scrubbed_email@example.com"
    data["address"] = "Scrubbed Address"
    data["phone_number"] = "Scrubbed Phone"
    data["ssn"] = "Scrubbed SSN"
    # Additional PII fields can be scrubbed as needed
    return data

def scrub_and_upload_to_minio(input_bucket, output_bucket, num_files=10):
    for i in range(num_files):
        input_object_name = f"data_{i + 1}.parquet"
        output_object_name = f"clean_{i + 1}.parquet"

        # Download Parquet file from raw bucket
        local_file_path = f"data/{input_object_name}"
        minio_client.fget_object(input_bucket, input_object_name, local_file_path)

        # Read Parquet file into a pandas DataFrame
        table = pq.read_table(local_file_path)
        df = table.to_pandas()

        # Scrub personally identifiable information
        df = scrub_pii(df)

        # Save the scrubbed data as Parquet
        scrubbed_table = pa.Table.from_pandas(df)
        scrubbed_file_path = os.path.join("data", output_object_name)
        pq.write_table(scrubbed_table, scrubbed_file_path)

        # Upload scrubbed data to the clean bucket
        with open(scrubbed_file_path, 'rb') as scrubbed_data:
            try:
                minio_client.put_object(output_bucket, output_object_name, scrubbed_data, os.path.getsize(scrubbed_file_path), content_type="application/octet-stream")
                print(f"Uploaded {output_object_name} to MinIO bucket: {output_bucket}")
            except S3Error as e:
                print(f"Error uploading {output_object_name} to MinIO bucket: {e}")

# Specify the MinIO raw and clean bucket names
raw_bucket_name = "raw"
clean_bucket_name = "clean"

# Scrub and upload data to the clean MinIO bucket
scrub_and_upload_to_minio(raw_bucket_name, clean_bucket_name)