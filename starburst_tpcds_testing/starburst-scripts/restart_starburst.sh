#!/bin/bash
# restart_starburst.sh
# Restart the starburst service on all cluster nodes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=starburst_hosts
source "${SCRIPT_DIR}/starburst_hosts"

echo "=== Restarting starburst on ${ALL_NODES} ==="
clush -w "${ALL_NODES}" "sudo systemctl restart starburst"

echo ""
echo "=== Checking status ==="
sleep 5
clush -w "${ALL_NODES}" "sudo systemctl status starburst --no-pager | grep -E 'Active'"
