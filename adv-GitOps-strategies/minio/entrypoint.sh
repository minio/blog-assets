#!/bin/sh

set -e

# Start MinIO in the background
minio server /data --console-address ":9001" &

# Wait for MinIO to start
sleep 5

# Set up alias and create bucket
mc alias set myminio http://minio:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD}

if ! mc ls myminio/weaviate-backups; then
  mc mb myminio/weaviate-backups
fi

if ! mc ls myminio/cda-datasets; then
  mc mb myminio/cda-datasets
fi


# Keep the script running to prevent the container from exiting
tail -f /dev/null
