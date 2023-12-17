#!/bin/sh
# Start MinIO
minio server /data &
MINIO_PID=$!

# Setup MinIO Client (mc) and configure webhook
mc alias set myminio http://localhost:9000 minio minio123
mc admin config set myminio notify_webhook:1 endpoint="http://flaskapp:5000/event" queue_limit="10" comment="Webhook for Flask app"
mc admin service restart myminio

# Keep the container running
wait $MINIO_PID
