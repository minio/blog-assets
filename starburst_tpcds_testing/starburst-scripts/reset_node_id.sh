#!/bin/bash
# reset_node_id.sh
# Regenerate a unique node.id in /etc/starburst/node.properties on all nodes.
# Sets node.internal-address to each node's 192.168.11.x hostname (loadgen-NNa).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=starburst_hosts
source "${SCRIPT_DIR}/starburst_hosts"

NODES=(${ALL_NODES//,/ })

for node in "${NODES[@]}"; do
    ssh "${node}" bash << EOF
uuid=\$(uuidgen)
sed -i "s/\(node\.id=\).*/\1\${uuid}/" /etc/starburst/node.properties
if grep -q '^node\.internal-address=' /etc/starburst/node.properties; then
    sed -i 's|^node\.internal-address=.*|node.internal-address=${node}|' /etc/starburst/node.properties
else
    echo 'node.internal-address=${node}' >> /etc/starburst/node.properties
fi
echo "  \$(hostname): node.id=\${uuid}, internal-address=${node}"
EOF
done
