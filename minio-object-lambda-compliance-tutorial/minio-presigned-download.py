from minio import Minio
from datetime import timedelta
import requests

# Set your Minio server information
minio_endpoint = '127.0.0.1:9000'
minio_access_key = 'minioadmin'
minio_secret_key = 'minioadmin'

# Initialize a Minio client
s3Client = Minio(minio_endpoint, access_key=minio_access_key, secret_key=minio_secret_key, secure=False)

# Set lambda function target via `lambdaArn`
lambda_arn = 'arn:minio:s3-object-lambda::function:webhook'

# Generate presigned GET URL with lambda function
bucket_name = 'anon-commerce-data'
object_name = 'fake_pos_transactions.json'
expiration = timedelta(seconds=1000)  # Expiration time in seconds

req_params = {'lambdaArn': lambda_arn}
presigned_url = s3Client.presigned_get_object(bucket_name, object_name, expires=expiration, response_headers=req_params)
print(presigned_url)

# Use the presigned URL to retrieve the data using requests
response = requests.get(presigned_url)

if response.status_code == 200:
    content = response.content
    print("Transformed data:\n", content)
else:
    print("Failed to download the data. Status code:", response.status_code, "Reason:", response.reason)
