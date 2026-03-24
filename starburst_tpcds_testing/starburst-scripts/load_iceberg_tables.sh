#!/bin/bash
# load_iceberg_tables.sh
# Loads all TPC-DS sf10000 tables into aistor.scale10000 via Starburst REST API.
# Runs tables sequentially, skips tables that already exist.

set -euo pipefail

COORDINATOR="http://loadgen-09a:8080"
CATALOG="aistor"
SCHEMA="scale10000"
SOURCE="tpcds.sf10000"
LOG="/tmp/load_iceberg_tables.log"

# Tables ordered small → large
TABLES=(
    time_dim item customer customer_address customer_demographics
    household_demographics income_band promotion reason ship_mode
    warehouse call_center store web_page web_site dbgen_version
    inventory store_returns catalog_returns web_returns
    web_sales catalog_sales store_sales
)

run_query() {
    local sql="$1"
    local user="admin"

    # Submit query
    local response
    response=$(curl -s -X POST "${COORDINATOR}/v1/statement" \
        -H "X-Trino-User: ${user}" \
        -H "X-Trino-Catalog: ${CATALOG}" \
        -H "Content-Type: text/plain" \
        --data "${sql}")

    local next_uri
    next_uri=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('nextUri',''))" 2>/dev/null)

    if [[ -z "$next_uri" ]]; then
        echo "ERROR: Failed to submit query. Response: $response"
        return 1
    fi

    # Poll until done
    while [[ -n "$next_uri" ]]; do
        sleep 2
        response=$(curl -s -X GET "$next_uri" -H "X-Trino-User: ${user}")
        next_uri=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('nextUri',''))" 2>/dev/null)
        local state
        state=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('stats',{}).get('state',''))" 2>/dev/null)
        local rows
        rows=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('stats',{}).get('processedRows',0))" 2>/dev/null)
        echo -ne "  state: ${state}, rows: ${rows}\r"

        if echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if 'error' not in d else 1)" 2>/dev/null; then
            true
        else
            local err
            err=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['error']['message'])" 2>/dev/null)
            echo ""
            echo "  ERROR: $err"
            return 1
        fi
    done
    echo ""
}

echo "=== TPC-DS sf10000 → aistor.scale10000 ===" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

for table in "${TABLES[@]}"; do
    echo -n "[$(date +%H:%M:%S)] $table ... " | tee -a "$LOG"

    # Check if table already exists
    check=$(curl -s -X POST "${COORDINATOR}/v1/statement" \
        -H "X-Trino-User: admin" \
        -H "Content-Type: text/plain" \
        --data "SELECT count(*) FROM ${CATALOG}.${SCHEMA}.${table} LIMIT 1")
    if ! echo "$check" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(1 if 'error' in d else 0)" 2>/dev/null; then
        echo "already exists, skipping" | tee -a "$LOG"
        continue
    fi

    start_time=$SECONDS
    if run_query "CREATE TABLE ${CATALOG}.${SCHEMA}.${table} AS SELECT * FROM ${SOURCE}.${table}" 2>&1 | tee -a "$LOG"; then
        elapsed=$((SECONDS - start_time))
        echo "  DONE in ${elapsed}s" | tee -a "$LOG"
    else
        echo "  FAILED" | tee -a "$LOG"
    fi
done

echo "" | tee -a "$LOG"
echo "=== Completed: $(date) ===" | tee -a "$LOG"
