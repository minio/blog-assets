import json
import random
from minio import Minio
from minio.error import S3Error

# Set your Minio server information
minio_endpoint = '127.0.0.1:9000'
minio_access_key = 'minioadmin'
minio_secret_key = 'minioadmin'

# Initialize a Minio client
s3Client = Minio(minio_endpoint, access_key=minio_access_key, secret_key=minio_secret_key, secure=False)

# Define the length of the fake credit card numbers
credit_card_length = 16  # The same length as provided examples

# Generate fake POS transactions
pos_transactions = []
for _ in range(10):  # Generate 10 fake transactions
    # Generate a random credit card number with the specified length
    credit_card_number = ''.join(random.choice('0123456789') for _ in range(credit_card_length))

    transaction = {
        "transaction_id": random.randint(1000, 9999),
        "amount": round(random.uniform(10.0, 100.0), 2),
        "currency": "USD",
        "credit_card_number": credit_card_number,
        "timestamp": "2023-11-08T12:00:00"  # You can use the actual timestamp
    }
    pos_transactions.append(transaction)

# Save the POS transactions to a JSON file
json_file_path = "fake_pos_transactions.json"
with open(json_file_path, "w") as json_file:
    json.dump(pos_transactions, json_file, indent=4)

print(f"Fake POS transactions with random credit card numbers have been generated and saved to {json_file_path}.")

# Define the bucket name
bucket_name = 'anon-commerce-data'

# Create the bucket if it doesn't exist
try:
    s3Client.make_bucket(bucket_name, location="us-east-1")
    print(f"Bucket '{bucket_name}' created successfully.")
except S3Error as e:
    if e.code == 'BucketAlreadyOwnedByYou':
        print(f"Bucket '{bucket_name}' already exists.")
    else:
        print(f"Error creating bucket: {e}")

# Upload the file to the bucket
try:
    s3Client.fput_object(bucket_name, json_file_path, json_file_path)
    print(f"File '{json_file_path}' uploaded successfully to '{bucket_name}' bucket.")
except S3Error as e:
    print(f"Error uploading file to bucket: {e}")