#!/bin/bash
# push_config.sh
# Sync configuration to the cluster:
#   coordinator/ → coordinator:/etc/starburst   (excludes catalog and node.properties)
#   worker/      → each worker:/etc/starburst   (excludes catalog and node.properties)
#   catalog/     → all nodes:/etc/starburst/catalog

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=starburst_hosts
source "${SCRIPT_DIR}/starburst_hosts"

WORK=$(mktemp -d)
trap "rm -rf ${WORK}" EXIT

# ── Build tarballs locally ────────────────────────────────────────────────────
# Collect basenames into arrays (maxdepth 1 excludes subdirs; node.properties excluded for config)
coord_files=()
while IFS= read -r -d '' f; do
    coord_files+=("$(basename "$f")")
done < <(find "${SCRIPT_DIR}/coordinator" -maxdepth 1 -type f ! -name 'node.properties' -print0)

worker_files=()
while IFS= read -r -d '' f; do
    worker_files+=("$(basename "$f")")
done < <(find "${SCRIPT_DIR}/worker" -maxdepth 1 -type f ! -name 'node.properties' -print0)

catalog_files=()
while IFS= read -r -d '' f; do
    catalog_files+=("$(basename "$f")")
done < <(find "${SCRIPT_DIR}/catalog" -maxdepth 1 -type f -print0)

# COPYFILE_DISABLE=1 suppresses macOS xattr metadata in the archives
COPYFILE_DISABLE=1 tar -czf "${WORK}/coord.tar.gz"   -C "${SCRIPT_DIR}/coordinator" "${coord_files[@]}"
COPYFILE_DISABLE=1 tar -czf "${WORK}/worker.tar.gz"  -C "${SCRIPT_DIR}/worker"      "${worker_files[@]}"
COPYFILE_DISABLE=1 tar -czf "${WORK}/catalog.tar.gz" -C "${SCRIPT_DIR}/catalog"     "${catalog_files[@]}"

# ── Coordinator config ────────────────────────────────────────────────────────
echo "=== Syncing coordinator config → ${COORDINATOR}:/etc/starburst ==="
clush -w "${COORDINATOR}" --copy "${WORK}/coord.tar.gz" --dest /tmp/
clush -w "${COORDINATOR}" "sudo find /etc/starburst -maxdepth 1 -type f -not -name 'node.properties' -delete \
    && sudo tar --warning=no-unknown-keyword -xzf /tmp/coord.tar.gz -C /etc/starburst/ \
    && rm /tmp/coord.tar.gz"

# ── Worker config ─────────────────────────────────────────────────────────────
echo ""
echo "=== Syncing worker config → ${WORKERS}:/etc/starburst ==="
clush -w "${WORKERS}" --copy "${WORK}/worker.tar.gz" --dest /tmp/
clush -w "${WORKERS}" "sudo find /etc/starburst -maxdepth 1 -type f -not -name 'node.properties' -delete \
    && sudo tar --warning=no-unknown-keyword -xzf /tmp/worker.tar.gz -C /etc/starburst/ \
    && rm /tmp/worker.tar.gz"

# ── Catalog ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Syncing catalog → ${ALL_NODES}:/etc/starburst/catalog ==="
clush -w "${ALL_NODES}" --copy "${WORK}/catalog.tar.gz" --dest /tmp/
clush -w "${ALL_NODES}" "sudo find /etc/starburst/catalog -maxdepth 1 -type f -delete \
    && sudo tar --warning=no-unknown-keyword -xzf /tmp/catalog.tar.gz -C /etc/starburst/catalog/ \
    && rm /tmp/catalog.tar.gz"

echo ""
echo "=== Config sync complete ==="
