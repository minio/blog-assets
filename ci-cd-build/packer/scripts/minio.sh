#!/bin/bash -eu

MINIO_VERSION=${MINIO_VERSION}
MINIO_SERVICE_USER=${MINIO_SERVICE_USER:-minio-user}
MINIO_SERVICE_GROUP=${MINIO_SERVICE_GROUP:-minio-user}

echo "==> Downloading MinIO version ${MINIO_VERSION}"
wget https://dl.min.io/server/minio/release/linux-amd64/archive/minio_${MINIO_VERSION}_amd64.deb -O minio.deb
dpkg -i minio.deb

echo "==> Creating ${MINIO_SERVICE_GROUP} group and ${MINIO_SERVICE_USER} user"
groupadd -r ${MINIO_SERVICE_GROUP}
useradd -M -r -g ${MINIO_SERVICE_GROUP} ${MINIO_SERVICE_USER}

# Clean up
systemctl disable minio.service
rm -f minio.deb