# Workshop 01 — OpenCode Context

You are helping a developer work with **MinIO AIStor Tables**, an enterprise
Iceberg REST catalog built into MinIO. Read this entire file before generating
any code — AIStor has specific requirements that differ from generic Iceberg setups.

All patterns below are verified working against the stack defined in `docker-compose.yaml`.

## What `make up` sets up automatically

Running `make up` performs all base setup — the student never needs to do this manually:

| Step | What is created |
|------|----------------|
| Warehouse | `saleswarehouse` |
| Namespace | `retail` |
| Table | `retail.sales` (10 columns) |
| Seed data | 15 rows — 4 regions · 3 categories · 4 reps · Jan–Mar 2025 |
| Trino | Restarted and connected to the warehouse |

When writing scripts or prompts, assume this infrastructure already exists.
Connect to it — don't recreate it.

## Running generated scripts

`make up` creates a `.venv` with all packages installed. Always tell the student
to run scripts with:

```bash
.venv/bin/python3 script.py
```

---

## Stack

| Service | Internal host | External port |
|---------|---------------|---------------|
| MinIO AIStor (S3 + Iceberg REST) | `minio:9000` | `localhost:9000` |
| MinIO Console | `minio:9001` | `localhost:9001` |
| Trino (SQL engine) | `trino:8080` | `localhost:8090` |

Credentials: `minioadmin` / `minioadmin`

---

## Critical AIStor-specific rules

### Rule 1 — SigV4 service name MUST be `s3tables`

```python
# CORRECT
SigV4Auth(session.get_credentials(), "s3tables", "local").add_auth(request)

# WRONG — returns 403
SigV4Auth(session.get_credentials(), "s3", "local").add_auth(request)
```

### Rule 2 — Warehouse creation is a signed REST POST, not an S3 bucket create

```python
warehouse_url = f"{CATALOG_URL}/v1/warehouses"
payload = json.dumps({"name": warehouse_name})
# Must be signed with SigV4 — see full pattern below
```

### Rule 3 — Warehouse name must not contain underscores or special characters

```python
# CORRECT
WAREHOUSE = "saleswarehouse"

# WRONG — returns 400 "invalid characters"
WAREHOUSE = "sales_warehouse"
```

### Rule 4 — PyIceberg RestCatalog requires these exact settings

`rest.signing-region` IS valid for PyIceberg (unlike Trino — see Rule 6).

```python
catalog = RestCatalog(
    name="aistor",
    uri=CATALOG_URL,           # "http://localhost:9000/_iceberg"
    warehouse=WAREHOUSE,       # "saleswarehouse"
    **{
        "rest.sigv4-enabled":       "true",
        "rest.signing-name":        "s3tables",   # MUST be s3tables
        "rest.signing-region":      "local",
        "client.access-key-id":     ACCESS_KEY,
        "client.secret-access-key": SECRET_KEY,
        "client.region":            "local",
        "s3.endpoint":              MINIO_ENDPOINT,
        "s3.path-style-access":     "true",
        "s3.access-key-id":         ACCESS_KEY,
        "s3.secret-access-key":     SECRET_KEY,
    },
)
```

### Rule 5 — MinIO image must be the EDGE build — only EDGE activates the Tables REST catalog

```yaml
# CORRECT — /_iceberg endpoint is active
image: quay.io/minio/aistor/minio:EDGE.2026-01-14T20-23-54Z

# WRONG — /_iceberg returns "unsupported API call"
image: quay.io/minio/aistor/minio:RELEASE.2025-12-20T04-58-37Z
```

License must be passed via `--license` CLI flag, not `MINIO_LICENSE` env var:

```yaml
command: server /data --console-address ":9001" --license /etc/minio/minio.license
volumes:
  - ./license/license:/etc/minio/minio.license
```

### Rule 6 — Trino `signing-region` property does NOT exist in Trino 480+

```properties
# CORRECT — region is read from AWS_REGION env var set in docker-compose
iceberg.rest-catalog.security=SIGV4
iceberg.rest-catalog.signing-name=s3tables

# WRONG — Trino will refuse to start
iceberg.rest-catalog.signing-region=local
```

### Rule 7 — Trino catalog name is `iceberg`, not the warehouse name

```sql
-- CORRECT
SELECT * FROM iceberg.retail.sales

-- WRONG
SELECT * FROM saleswarehouse.retail.sales
```

### Rule 8 — MinIO data lives in a Docker named volume, not a host directory

The `minio-data` Docker volume is managed by Docker itself. This avoids stale
`.minio.sys` state that would otherwise cause "warehouse does not exist" errors
after a `make clean`.

```bash
make clean          # runs: docker compose down -v  ← removes the volume completely
make up             # fresh volume created automatically on next start
```

Never bind-mount a host directory for MinIO data. Leftover `.minio.sys/catalog/`
entries from a previous run will cause `/_iceberg/v1/config?warehouse=X` to return 404
even when `POST /v1/warehouses` appears to succeed.

### Rule 9 — Trino must be restarted after the warehouse is created on a fresh stack

Trino connects to the REST catalog at startup. If the warehouse didn't exist yet,
restart Trino once after creating it:

```bash
docker compose restart trino   # or: make restart-trino
# wait ~20s for Trino to reconnect, then run your SQL
```

---

## Project variables

```python
MINIO_ENDPOINT = "http://localhost:9000"
CATALOG_URL    = f"{MINIO_ENDPOINT}/_iceberg"
WAREHOUSE      = "saleswarehouse"
NAMESPACE      = "retail"
TABLE_NAME     = "sales"
ACCESS_KEY     = "minioadmin"
SECRET_KEY     = "minioadmin"
TRINO_HOST     = "localhost"
TRINO_PORT     = 8090
```

---

## Verified warehouse creation pattern

```python
import hashlib, json, requests, boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

def create_warehouse(warehouse_name):
    """Create an AIStor Tables warehouse (idempotent)."""
    session = boto3.Session(
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name="local",
    )
    url     = f"{CATALOG_URL}/v1/warehouses"
    payload = json.dumps({"name": warehouse_name})
    headers = {
        "content-type":         "application/json",
        "content-length":       str(len(payload)),
        "x-amz-content-sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "Host":                 MINIO_ENDPOINT.replace("http://", "").replace("https://", ""),
    }
    req  = AWSRequest(method="POST", url=url, data=payload, headers=headers)
    SigV4Auth(session.get_credentials(), "s3tables", "local").add_auth(req)
    resp = requests.post(url, data=payload, headers=req.headers)
    if resp.status_code in (200, 201):
        print(f"✅ Warehouse '{warehouse_name}' created: {resp.json()}")
    elif resp.status_code == 409:
        print(f"ℹ️  Warehouse '{warehouse_name}' already exists")
    else:
        resp.raise_for_status()
```

---

## Verified catalog connection pattern

```python
import os
from pyiceberg.catalog.rest import RestCatalog

os.environ["AWS_ACCESS_KEY_ID"]     = ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = SECRET_KEY
os.environ["AWS_REGION"]            = "local"

catalog = RestCatalog(
    name="aistor",
    uri=CATALOG_URL,
    warehouse=WAREHOUSE,
    **{
        "rest.sigv4-enabled":       "true",
        "rest.signing-name":        "s3tables",
        "rest.signing-region":      "local",
        "client.access-key-id":     ACCESS_KEY,
        "client.secret-access-key": SECRET_KEY,
        "client.region":            "local",
        "s3.endpoint":              MINIO_ENDPOINT,
        "s3.path-style-access":     "true",
        "s3.access-key-id":         ACCESS_KEY,
        "s3.secret-access-key":     SECRET_KEY,
    },
)
```

---

## Verified namespace creation pattern

Use the specific PyIceberg exception — more reliable than string matching.

```python
from pyiceberg.exceptions import NamespaceAlreadyExistsError

try:
    catalog.create_namespace((NAMESPACE,))
    print(f"✅ Namespace '{NAMESPACE}' created")
except NamespaceAlreadyExistsError:
    print(f"ℹ️  Namespace '{NAMESPACE}' already exists")
```

---

## Schema definition — two styles, both work

### Option A: PyArrow schema (simpler)

```python
import pyarrow as pa

SALES_SCHEMA = pa.schema([
    pa.field("order_id",      pa.int32()),
    pa.field("customer_name", pa.string()),
    pa.field("product",       pa.string()),
    pa.field("category",      pa.string()),   # Electronics | Furniture | Office Supplies
    pa.field("quantity",      pa.int32()),
    pa.field("unit_price",    pa.float64()),
    pa.field("total_amount",  pa.float64()),
    pa.field("region",        pa.string()),   # East | West | North | South
    pa.field("sale_date",     pa.string()),   # YYYY-MM-DD
    pa.field("rep_name",      pa.string()),
])
```

### Option B: PyIceberg native schema (more control — use when nullable fields matter)

Pandas DataFrames produce nullable columns so `required=False` is safest.

```python
from pyiceberg.schema import Schema
from pyiceberg.types import NestedField, StringType, DoubleType, IntegerType

SALES_SCHEMA = Schema(
    NestedField(1,  "order_id",      IntegerType(), required=False),
    NestedField(2,  "customer_name", StringType(),  required=False),
    NestedField(3,  "product",       StringType(),  required=False),
    NestedField(4,  "category",      StringType(),  required=False),
    NestedField(5,  "quantity",      IntegerType(), required=False),
    NestedField(6,  "unit_price",    DoubleType(),  required=False),
    NestedField(7,  "total_amount",  DoubleType(),  required=False),
    NestedField(8,  "region",        StringType(),  required=False),
    NestedField(9,  "sale_date",     StringType(),  required=False),
    NestedField(10, "rep_name",      StringType(),  required=False),
)
```

---

## Verified table creation pattern (idempotent)

```python
from pyiceberg.exceptions import TableAlreadyExistsError

try:
    table = catalog.create_table(
        identifier=(NAMESPACE, TABLE_NAME),
        schema=SALES_SCHEMA,
    )
    print(f"✅ Table '{NAMESPACE}.{TABLE_NAME}' created")
except TableAlreadyExistsError:
    table = catalog.load_table((NAMESPACE, TABLE_NAME))
    print(f"ℹ️  Table loaded (already exists)")
```

---

## Verified data insertion pattern

Always pass the schema explicitly to `from_pandas` to avoid type inference mismatches.

```python
import pandas as pd
import pyarrow as pa

arrow_table = pa.Table.from_pandas(df, schema=SALES_SCHEMA)
table.append(arrow_table)
```

Idempotent version (only insert if table is empty):

```python
if len(table.scan().to_pandas()) == 0:
    table.append(pa.Table.from_pandas(df, schema=SALES_SCHEMA))
```

---

## Trino connection helper

```python
import trino, pandas as pd

conn = trino.dbapi.connect(
    host=TRINO_HOST,
    port=TRINO_PORT,
    user="admin",
    catalog="iceberg",
    schema=NAMESPACE,
)

def trino_query(sql):
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    return pd.DataFrame(rows, columns=cols)
```

---

## Code style conventions

- Use **single-quoted** Python strings
- Use `NamespaceAlreadyExistsError` and `TableAlreadyExistsError` from `pyiceberg.exceptions` — not bare `Exception`
- Use `'''...'''` triple single-quotes for multi-line SQL strings
- All code must be **idempotent** — safe to re-run without duplicating data
- Prefer PyArrow schema for simple tables; use `Schema+NestedField` when nullable control is needed
