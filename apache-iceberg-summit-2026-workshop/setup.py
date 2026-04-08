"""
AIStor Tables — Workshop 01 base setup
Runs automatically via: make up

Creates warehouse → namespace → table → inserts seed data.
All steps are idempotent — safe to re-run.
"""

import hashlib, json, os, requests, boto3
import pyarrow as pa
import pandas as pd
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from pyiceberg.catalog.rest import RestCatalog
from pyiceberg.exceptions import NamespaceAlreadyExistsError, TableAlreadyExistsError

MINIO_ENDPOINT = "http://localhost:9000"
CATALOG_URL    = f"{MINIO_ENDPOINT}/_iceberg"
WAREHOUSE      = "saleswarehouse"
NAMESPACE      = "retail"
TABLE_NAME     = "sales"
ACCESS_KEY     = "minioadmin"
SECRET_KEY     = "minioadmin"

os.environ["AWS_ACCESS_KEY_ID"]     = ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = SECRET_KEY
os.environ["AWS_REGION"]            = "local"

# ── helpers ───────────────────────────────────────────────────────────────────

def _sign(method, url, body="", extra=None):
    session = boto3.Session(
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name="local",
    )
    hdrs = extra or {}
    hdrs["x-amz-content-sha256"] = hashlib.sha256(body.encode()).hexdigest()
    hdrs["Host"] = MINIO_ENDPOINT.replace("http://","").replace("https://","")
    req = AWSRequest(method=method, url=url, data=body, headers=hdrs)
    SigV4Auth(session.get_credentials(), "s3tables", "local").add_auth(req)
    return req

# ── Step 1: Warehouse ─────────────────────────────────────────────────────────
print("  [1/5] Creating warehouse...")
url     = f"{CATALOG_URL}/v1/warehouses"
payload = json.dumps({"name": WAREHOUSE})
signed  = _sign("POST", url, payload, {
    "content-type": "application/json", "content-length": str(len(payload))
})
resp = requests.post(url, data=payload, headers=signed.headers)
if resp.status_code in (200, 201):
    print(f"       Warehouse '{WAREHOUSE}' created  uuid={resp.json().get('uuid','?')}")
elif resp.status_code == 409:
    print(f"       Warehouse '{WAREHOUSE}' already exists")
else:
    resp.raise_for_status()

# ── Step 2: Catalog connection ────────────────────────────────────────────────
print("  [2/5] Connecting to catalog...")
catalog = RestCatalog(
    name="aistor", uri=CATALOG_URL, warehouse=WAREHOUSE,
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
print(f"       Connected to {CATALOG_URL}")

# ── Step 3: Namespace ─────────────────────────────────────────────────────────
print("  [3/5] Creating namespace...")
try:
    catalog.create_namespace((NAMESPACE,))
    print(f"       Namespace '{NAMESPACE}' created")
except NamespaceAlreadyExistsError:
    print(f"       Namespace '{NAMESPACE}' already exists")

# ── Step 4: Table ─────────────────────────────────────────────────────────────
print("  [4/5] Creating table...")
SALES_SCHEMA = pa.schema([
    pa.field("order_id",      pa.int32()),
    pa.field("customer_name", pa.string()),
    pa.field("product",       pa.string()),
    pa.field("category",      pa.string()),
    pa.field("quantity",      pa.int32()),
    pa.field("unit_price",    pa.float64()),
    pa.field("total_amount",  pa.float64()),
    pa.field("region",        pa.string()),
    pa.field("sale_date",     pa.string()),
    pa.field("rep_name",      pa.string()),
])
try:
    table = catalog.create_table(identifier=(NAMESPACE, TABLE_NAME), schema=SALES_SCHEMA)
    print(f"       Table '{NAMESPACE}.{TABLE_NAME}' created")
except TableAlreadyExistsError:
    table = catalog.load_table((NAMESPACE, TABLE_NAME))
    print(f"       Table '{NAMESPACE}.{TABLE_NAME}' loaded")

# ── Step 5: Seed data ─────────────────────────────────────────────────────────
print("  [5/5] Inserting seed data...")
ROWS = [
    (1001,"Acme Corp",    "Laptop Pro 15",      "Electronics",      2,1299.99,2599.98,"East", "2025-01-05","Sarah Chen"),
    (1002,"Global Inc",   "Standing Desk",       "Furniture",        4, 649.00,2596.00,"West", "2025-01-08","Mike Torres"),
    (1003,"TechStart",    "Wireless Headset",    "Electronics",     10, 189.99,1899.90,"North","2025-01-12","Sarah Chen"),
    (1004,"Retail Plus",  "Ergonomic Chair",     "Furniture",        6, 399.00,2394.00,"South","2025-01-15","James Wilson"),
    (1005,"DataCo",       "Monitor 4K 27in",     "Electronics",      3, 799.99,2399.97,"East", "2025-01-18","Anna Park"),
    (1006,"MegaStore",    "Printer Laser",       "Office Supplies",  2, 499.99, 999.98,"West", "2025-01-22","Mike Torres"),
    (1007,"SunriseMedia", "Keyboard Mechanical", "Electronics",      8, 149.99,1199.92,"East", "2025-02-03","Anna Park"),
    (1008,"BizHub",       "Filing Cabinet",      "Furniture",        3, 279.00, 837.00,"South","2025-02-07","James Wilson"),
    (1009,"CloudNet",     "Webcam HD Pro",       "Electronics",     15,  79.99,1199.85,"North","2025-02-11","Sarah Chen"),
    (1010,"InnovateCo",   "Paper Ream Case",     "Office Supplies", 20,  42.99, 859.80,"East", "2025-02-14","Anna Park"),
    (1011,"PrimeRetail",  "Tablet 10-inch",      "Electronics",      5, 449.99,2249.95,"West", "2025-02-18","Mike Torres"),
    (1012,"FastTrack",    "Conference Table",    "Furniture",        1,1599.00,1599.00,"North","2025-02-21","Sarah Chen"),
    (1013,"GreenEnergy",  "USB-C Hub 7in1",      "Electronics",     12,  59.99, 719.88,"South","2025-02-25","James Wilson"),
    (1014,"TechGiant",    "Ink Cartridges Set",  "Office Supplies",  8,  89.99, 719.92,"East", "2025-03-04","Anna Park"),
    (1015,"SmartOffice",  "Smart TV 55in",       "Electronics",      2,1099.99,2199.98,"West", "2025-03-07","Mike Torres"),
]
COLS = ["order_id","customer_name","product","category",
        "quantity","unit_price","total_amount","region","sale_date","rep_name"]
df = pd.DataFrame(ROWS, columns=COLS)

current = len(table.scan().to_pandas())
if current == 0:
    table.append(pa.Table.from_pandas(df, schema=SALES_SCHEMA))
    print(f"       Inserted {len(df)} rows into {NAMESPACE}.{TABLE_NAME}")
else:
    print(f"       Table already has {current} rows — skipping insert")

print()
print(f"  ✅  Setup complete")
print(f"      warehouse : {WAREHOUSE}")
print(f"      table     : {NAMESPACE}.{TABLE_NAME}  ({max(current,len(df))} rows)")
print(f"      location  : {table.location()}")
