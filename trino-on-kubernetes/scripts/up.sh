#!/bin/sh

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${BASE_DIR}/.."
(
cd ${REPO_DIR}
kubectl create namespace trino --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic redis-table-definition --from-file=redis/test.json -n trino || true

# Adding Helm repos and ignoring errors if the repo already exists
helm repo add bitnami https://charts.bitnami.com/bitnami || true
helm repo add trino https://trinodb.github.io/charts/ || true

kubectl minio init -n trino
kubectl minio tenant create tenant-1 --servers 4 --volumes 4 --capacity 4Gi -n trino
helm upgrade --install hive-metastore-postgresql bitnami/postgresql -n trino -f hive-metastore-postgresql/values.yaml
helm upgrade --install my-hive-metastore -n trino -f hive-metastore/values.yaml ./charts/hive-metastore
helm upgrade --install my-redis bitnami/redis -n trino -f redis/values.yaml
# helm upgrade --install my-trino trino/trino --version 0.7.0 --namespace trino -f trino/values.yaml

)
