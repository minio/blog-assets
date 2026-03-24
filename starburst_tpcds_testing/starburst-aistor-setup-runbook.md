# Starburst + MinIO AIStor Setup Runbook
Date: 2026-03-20
Cluster: loadgen1-9 (Starburst), sds01-08 (MinIO AIStor)

## Cluster Layout
- **Coordinator**: loadgen9 (loadgen-09a = 192.168.11.19)
- **Workers**: loadgen1–8 (loadgen-01a–loadgen-08a = 192.168.11.11–18)
- **MinIO AIStor**: sds01–08 (sds-01a–sds-08a = 192.168.11.1–8), HTTPS on port 9000
- **Deployment scripts**: root@loadgen1:/root/starburst/

---

## Problem: Workers Not Appearing in Coordinator UI

### Root Cause
`worker/config.properties` had `node.internal-address=trino-internal` pushed to all 8 workers.
On the coordinator, `trino-internal` resolved to its own IP (192.168.11.19), so it
could never reach any worker — they all appeared to register at the coordinator's own address.

### Fix Applied (2026-03-20)

#### Files backed up
```
/root/starburst/reset_node_id.sh            → reset_node_id.sh.20260319
/root/starburst/worker/config.properties    → worker/config.properties.20260319
/root/starburst/coordinator/config.properties → coordinator/config.properties.20260319
```

#### 1. `worker/config.properties` — removed shared node.internal-address
```diff
- node.internal-address=trino-internal
```
Each worker's address now lives in its own `node.properties` (excluded from push_config.sh).

#### 2. `coordinator/config.properties` — uncommented bind-host
```diff
- #http-server.http.bind-host=trino-internal
+ http-server.http.bind-host=loadgen-09a
```

#### 3. `reset_node_id.sh` — changed from clush to per-node for loop
Previously used `clush` with a single heredoc, making every node get `trino-internal`.
Rewrote to use a `for node in "${NODES[@]}"` loop so each node gets its own `loadgen-NNa`
address (e.g. loadgen-01a, loadgen-02a, ...) written into `node.properties`.

#### Steps run to fix live cluster
```bash
cd /root/starburst
bash reset_node_id.sh      # updates node.properties on all 9 nodes with unique node.id + correct node.internal-address
bash push_config.sh        # pushes updated worker/coordinator config.properties + catalogs
source starburst_hosts && clush -w "${ALL_NODES}" sudo systemctl restart starburst
```

#### Result
All 9 nodes came up active. Each node.properties now contains:
- Unique `node.id` (UUID)
- Correct `node.internal-address=loadgen-NNa` (per-node 192.168.11.x hostname)

---

## Setting Up AIStor Iceberg Catalog in Starburst

### Overview
Configure Starburst to query TPC-DS data stored as Iceberg tables in MinIO AIStor.
- Iceberg REST catalog endpoint: `https://sds-01a:9000/_iceberg`
- Warehouse: `tpcds-iceberg`
- Namespaces: `scale1000`, `scale3000`, `scale5000`

### Step 1 — Create catalog properties file

Based on `catalog/hold/lab1temp.properties` template.
Created `catalog/aistor.properties`:

```ini
connector.name=iceberg
iceberg.catalog.type=rest
iceberg.rest-catalog.uri=https://sds-01a:9000/_iceberg
iceberg.rest-catalog.warehouse=tpcds-iceberg
iceberg.rest-catalog.security=SIGV4
iceberg.rest-catalog.signing-name=s3tables
fs.native-s3.enabled=true
s3.endpoint=https://sds-01a:9000
s3.region=us-east-1
s3.path-style-access=true
s3.aws-access-key=minioadmin
s3.aws-secret-key=minioadmin
```

### Step 2 — Import MinIO CA cert into Starburst's bundled JVM

Starburst uses its own JVM at `/usr/lib/starburst/jvm/` with its own truststore,
separate from the system Java. The MinIO self-signed CA cert must be imported
into `/usr/lib/starburst/jvm/lib/security/cacerts` on **all 9 loadgen nodes**.

MinIO CA cert location on sds01: `/opt/minio/certs/public.crt`

```bash
# Run from loadgen1
source /root/starburst/starburst_hosts
# Copy cert from sds01 to local
scp root@sds01:/opt/minio/certs/public.crt /tmp/minio-ca.crt

# Import into Starburst JVM truststore on all nodes
for node in ${ALL_NODES//,/ }; do
    scp /tmp/minio-ca.crt ${node}:/tmp/minio-ca.crt
    ssh ${node} "keytool -import -trustcacerts -noprompt \
        -alias minio-ca \
        -file /tmp/minio-ca.crt \
        -keystore /usr/lib/starburst/jvm/lib/security/cacerts \
        -storepass changeit && rm /tmp/minio-ca.crt"
    echo "${node}: cert imported"
done
```

### Step 3 — Push catalog and restart

```bash
cd /root/starburst
bash push_config.sh
source starburst_hosts && clush -w "${ALL_NODES}" sudo systemctl restart starburst
```

### Step 4 — Verify

```sql
-- Check all workers registered (should be 9 rows, all http_uri on 192.168.11.x)
SELECT node_id, http_uri, state FROM system.runtime.nodes ORDER BY http_uri;

-- Check AIStor catalog schemas
SHOW SCHEMAS IN aistor;

-- Expected: scale1000, scale3000, scale5000

-- Quick row count test
SELECT count(*) FROM aistor.scale1000.store_sales;
```

---

## JVM Memory Tuning (2026-03-20)

### Backups
```
worker/jvm.config      → worker/jvm.config.20260320
coordinator/jvm.config → coordinator/jvm.config.20260320
```

### Changes made
- `jvm.config` (both worker and coordinator): `-Xmx3500m` → `-Xmx400g`
- `config.properties` (both): added `query.max-memory-per-node=280GB`

### Why 280GB not 350GB
Starburst reserves 30% of Xmx as heap headroom (~120GB on 400GB heap).
Formula: `query.max-memory-per-node + heap_headroom ≤ Xmx`
→ `280 + 120 = 400GB` ✓

---

## Loading TPC-DS sf10000 Data into Iceberg (2026-03-20)

### Script created
`/root/starburst/load_iceberg_tables.sh` — custom script (not part of original Starburst deployment).
Submits CTAS queries to the Starburst REST API via curl, polling until each completes.
Skips tables that already exist so it is safe to re-run if interrupted.

### What it does
For each of the 25 TPC-DS tables:
1. Checks if the table already exists in `aistor.scale10000` (skips if so)
2. Submits: `CREATE TABLE aistor.scale10000.<table> AS SELECT * FROM tpcds.sf10000.<table>`
3. Polls every 2 seconds printing state + row count
4. Logs to `/tmp/load_iceberg_tables.log`

Tables run in order: small dimensions first, large fact tables last.

### Pre-requisite
`date_dim` was already loaded manually as a test before running the script.

### Run command
```bash
ssh root@loadgen1
cd /root/starburst
nohup bash load_iceberg_tables.sh > /tmp/load_iceberg_tables.log 2>&1 &
```

### Monitor
```bash
# Follow live progress
ssh root@loadgen1 'tail -f /tmp/load_iceberg_tables.log'

# Check how many tables loaded so far
ssh root@sds01 'mc table ls myminio tpcds-iceberg scale10000'

# Check from Starburst UI
SHOW TABLES IN aistor.scale10000;
```

### Time estimate
| Phase | Tables | Estimated time |
|-------|--------|---------------|
| Small dimensions (16 tables) | time_dim, item, customer, etc. | 5–15 minutes total |
| Medium facts | inventory, store_returns, catalog_returns, web_returns | 1–2 hours each |
| Large facts | web_sales, catalog_sales, store_sales | 2–4 hours each |
| **Total** | 24 remaining tables | **8–14 hours** |

The bottleneck is generating and writing ~10TB across 8 workers to MinIO over 400Gbps.
`store_sales` is the largest table at ~2.88 billion rows and will take the longest.

---

## Notes
- MinIO alias: myminio → https://192.168.11.1:9000
- MinIO CA cert: /opt/minio/certs/public.crt (on sds01)
- Iceberg warehouse created via: `mc table warehouse create myminio tpcds-iceberg`
- Starburst version: 479-e.4
- discovery.uri uses loadgen-09a (192.168.11.19) — the coordinator's 400Gbps address
- All inter-node communication routes through 192.168.11.x (400Gbps interfaces)
