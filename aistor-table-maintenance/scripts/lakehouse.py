"""Shared helpers: connect to AIStor Tables and measure a table's health.

Used by make_mess.py, inspect_table.py and the notebook.
"""
import os
import time
from dataclasses import dataclass, asdict

import duckdb
import pyarrow as pa
from pyarrow import fs as pafs
from pyiceberg.catalog.rest import RestCatalog
from pyiceberg.exceptions import NamespaceAlreadyExistsError
from pyiceberg.schema import Schema
from pyiceberg.types import (DoubleType, LongType, NestedField, StringType,
                             TimestamptzType)

AISTOR_URL = os.environ.get("AISTOR_URL", "http://localhost:9000")
WAREHOUSE = os.environ.get("WAREHOUSE", "lakehouse")
ACCESS_KEY = os.environ.get("ACCESS_KEY", "ingest-svc")
SECRET_KEY = os.environ.get("SECRET_KEY", "ingest-svc-secret")
REGION = "us-east-1"

NAMESPACE = "iot"
TABLE = "sensor_readings"
IDENTIFIER = f"{NAMESPACE}.{TABLE}"

SCHEMA = Schema(
    NestedField(1, "reading_id", LongType(), required=True),
    NestedField(2, "device_id", StringType(), required=True),
    NestedField(3, "site", StringType(), required=True),
    NestedField(4, "ts", TimestamptzType(), required=True),
    NestedField(5, "temperature_c", DoubleType(), required=True),
    NestedField(6, "humidity_pct", DoubleType(), required=True),
)


def catalog() -> RestCatalog:
    """AIStor Tables catalog at /_iceberg, signed with SigV4 (service s3tables)."""
    return RestCatalog(
        name="aistor",
        uri=f"{AISTOR_URL}/_iceberg",
        warehouse=WAREHOUSE,
        **{
            "rest.sigv4-enabled": "true",
            "rest.signing-name": "s3tables",
            "rest.signing-region": REGION,
            "client.access-key-id": ACCESS_KEY,
            "client.secret-access-key": SECRET_KEY,
            "client.region": REGION,
            "s3.endpoint": AISTOR_URL,
            "s3.access-key-id": ACCESS_KEY,
            "s3.secret-access-key": SECRET_KEY,
            "s3.region": REGION,
            "s3.path-style-access": "true",
        },
    )


def ensure_table(cat: RestCatalog):
    try:
        cat.create_namespace(NAMESPACE)
    except NamespaceAlreadyExistsError:
        pass
    return cat.create_table_if_not_exists(IDENTIFIER, schema=SCHEMA)


def arrow_schema() -> pa.Schema:
    return SCHEMA.as_arrow()


@dataclass
class Health:
    rows: int
    snapshots: int
    data_files: int          # files the current snapshot reads
    data_mb: float
    avg_file_kb: float
    objects_on_storage: int  # every object under the table's location
    storage_mb: float
    query_ms: float          # median of a few runs of an aggregate query

    def as_dict(self):
        return asdict(self)


def _s3fs() -> pafs.S3FileSystem:
    host = AISTOR_URL.split("://", 1)[1]
    return pafs.S3FileSystem(
        access_key=ACCESS_KEY, secret_key=SECRET_KEY, region=REGION,
        endpoint_override=host, scheme=AISTOR_URL.split("://", 1)[0],
    )


def storage_footprint(table) -> tuple[int, float]:
    """Count every object under the table location, referenced or not."""
    loc = table.location().split("://", 1)[1].rstrip("/")
    infos = _s3fs().get_file_info(pafs.FileSelector(loc, recursive=True))
    files = [i for i in infos if i.type == pafs.FileType.File]
    return len(files), sum(i.size for i in files) / 1e6


def duckdb_conn() -> duckdb.DuckDBPyConnection:
    """DuckDB attached to the same AIStor Tables catalog (SigV4)."""
    host = AISTOR_URL.split("://", 1)[1]
    con = duckdb.connect()
    con.execute("INSTALL iceberg; LOAD iceberg; INSTALL httpfs; LOAD httpfs;")
    con.execute(f"""
        CREATE SECRET aistor_s3 (TYPE S3, KEY_ID '{ACCESS_KEY}', SECRET '{SECRET_KEY}',
          ENDPOINT '{host}', URL_STYLE 'path', USE_SSL 0, REGION '{REGION}')""")
    con.execute(f"""
        ATTACH '{WAREHOUSE}' AS aistor (TYPE ICEBERG,
          ENDPOINT '{AISTOR_URL}/_iceberg', AUTHORIZATION_TYPE 'sigv4',
          SIGV4_SERVICE 's3tables', SIGV4_REGION '{REGION}', SECRET aistor_s3)""")
    return con


QUERY = f"""
    SELECT site, count(*) AS readings, round(avg(temperature_c), 2) AS avg_temp_c
    FROM aistor.{NAMESPACE}.{TABLE}
    GROUP BY site ORDER BY site
"""


def time_query(runs: int = 5) -> tuple[float, list]:
    """Median wall time of QUERY in DuckDB, with a fresh connection per run."""
    times, result = [], None
    for _ in range(runs):
        con = duckdb_conn()
        start = time.perf_counter()
        result = con.execute(QUERY).fetchall()
        times.append((time.perf_counter() - start) * 1000)
        con.close()
    times.sort()
    return times[len(times) // 2], result


def health(cat: RestCatalog | None = None, query_runs: int = 5) -> Health:
    cat = cat or catalog()
    table = cat.load_table(IDENTIFIER)
    files = table.inspect.files().select(["record_count", "file_size_in_bytes", "content"])
    data = files.filter(pa.compute.equal(files["content"], 0))
    n = data.num_rows
    size = pa.compute.sum(data["file_size_in_bytes"]).as_py() or 0
    rows = pa.compute.sum(data["record_count"]).as_py() or 0
    objects, storage_mb = storage_footprint(table)
    query_ms, _ = time_query(query_runs) if query_runs else (0.0, None)
    return Health(
        rows=rows,
        snapshots=len(table.snapshots()),
        data_files=n,
        data_mb=round(size / 1e6, 2),
        avg_file_kb=round(size / n / 1e3, 1) if n else 0.0,
        objects_on_storage=objects,
        storage_mb=round(storage_mb, 2),
        query_ms=round(query_ms, 1),
    )


def print_health(h: Health, title: str = "") -> None:
    if title:
        print(f"\n{title}")
    print(f"  rows                  {h.rows:>10,}")
    print(f"  snapshots             {h.snapshots:>10,}")
    print(f"  data files (current)  {h.data_files:>10,}   avg {h.avg_file_kb:,.1f} KB, total {h.data_mb:,.2f} MB")
    print(f"  objects on storage    {h.objects_on_storage:>10,}   total {h.storage_mb:,.2f} MB (data + metadata, incl. unreferenced)")
    if h.query_ms:
        print(f"  DuckDB query (median) {h.query_ms:>10,.1f} ms")
