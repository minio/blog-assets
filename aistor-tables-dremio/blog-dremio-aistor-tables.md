# Run SQL Analytics on AIStor Tables with Dremio

Apache Iceberg has become the default table format for analytics on object storage. The appeal is straightforward: store data as Parquet files on S3-compatible storage, manage it through a catalog, and query it from any engine that speaks Iceberg. No vendor lock-in on the query side, no proprietary formats on the storage side.

The harder question has always been the catalog. Iceberg tables need something to track metadata -- which Parquet files belong to which table, what the schema looks like, where the latest snapshot lives. Historically, that meant running a Hive Metastore backed by PostgreSQL, or adopting a managed catalog service like AWS Glue. Either way, it's another system to deploy, secure, and maintain alongside your actual storage.

MinIO AIStor Tables takes a different approach. The Iceberg REST Catalog is built directly into the storage server. There is no external metastore. When you write a table, the catalog metadata and the Parquet data files live on the same MinIO cluster, managed by the same process. This eliminates an entire tier of infrastructure from the lakehouse stack.

This post walks through connecting Dremio -- a SQL analytics engine with native Iceberg support -- to AIStor Tables. We will create tables with PyIceberg, then query them from Dremio using standard SQL. Everything runs in Docker Compose, and a companion repository has the full working setup.

## Why This Combination

AIStor Tables and Dremio serve different layers of the analytics stack, and the integration between them is worth understanding.

**AIStor Tables** is the storage and catalog layer. It exposes two APIs from a single endpoint:

- The **Iceberg REST Catalog API** at `/_iceberg` for metadata operations -- creating warehouses, namespaces, tables, and views. PyIceberg, Spark, Trino, and Dremio all speak this protocol.
- The **S3 API** for reading and writing the actual Parquet data files. Any S3-compatible client works here.

Because both APIs run on the same server, there is no network hop between catalog lookups and data reads. The catalog knows exactly where the data lives because it manages the storage directly.

**Dremio** is the query engine layer. It connects to the Iceberg REST Catalog to discover tables, then reads the Parquet files over S3 to execute queries. Dremio supports the `RESTCATALOG` source type, which is purpose-built for this pattern. You point it at a catalog endpoint and an S3 endpoint, and it handles the rest -- predicate pushdown into Parquet, partition pruning via Iceberg metadata, and columnar reads for only the columns your query touches.

The practical result is a two-component analytics stack: MinIO for storage and catalog, Dremio for SQL. No Hive Metastore, no PostgreSQL for catalog state, no Glue dependency. Data engineers write tables with PyIceberg or Spark, analysts query them with SQL in Dremio.

## Architecture

The data flow has two paths:

```
┌─────────────────┐       ┌──────────────────────────────────┐
│     Dremio      │       │         MinIO AIStor              │
│  (Query Engine) │       │                                   │
│                 ├──────►│  /_iceberg  (REST Catalog API)    │
│                 │  (1)  │    Table metadata, schemas,       │
│                 │       │    snapshot tracking               │
│                 │       │                                   │
│                 ├──────►│  S3 API    (Data Access)          │
│                 │  (2)  │    Parquet data files             │
└─────────────────┘       └──────────────────────────────────┘
```

1. Dremio calls the Iceberg REST Catalog to list namespaces, load table metadata, and resolve file locations.
2. Dremio reads Parquet data files over the S3 API, applying predicate pushdown and column pruning.

Both paths terminate at the same MinIO server. Authentication uses SigV4 signing for catalog requests and standard S3 credentials for data access.

## Setting Up the Environment

The companion repository provides a Docker Compose stack with three services: MinIO AIStor, Dremio Enterprise, and Jupyter.

### Prerequisites

- Docker and Docker Compose
- An AIStor enterprise license
- A Dremio Enterprise license or trial

### Licensing

Both AIStor Tables and Dremio Enterprise require licenses to run. Here is how to obtain each one.

**AIStor**: The Tables feature requires an AIStor enterprise license. Go to [min.io/download](https://www.min.io/download) to request a license key. This is a single string that you set as an environment variable in the `.env` file. MinIO picks it up automatically on startup -- no UI step required.

**Dremio Enterprise**: The `RESTCATALOG` source type is only available in Dremio Enterprise -- it is not included in the Community (OSS) edition. Register at [dremio.com/get-started](https://www.dremio.com/get-started/) for a 30-day free trial. The process is fully automated -- no sales call required. After registering, you receive an email with:

1. **Container registry credentials** -- a robot account username and token for pulling the Enterprise Docker image from `quay.io`
2. **A license key** -- entered through the Dremio UI on first launch (or included in a `values-overrides.yaml` for Kubernetes deployments)

The trial gives full access to all Enterprise features for 30 days.

### Starting the Stack

Clone the repository:

```bash
git clone https://github.com/miniohq/aistor-tables-dremio.git
cd aistor-tables-dremio
```

Open `.env` and set your AIStor license:

```
MINIO_LICENSE=your-actual-license-key
```

Log in to the Dremio Enterprise container registry using the credentials from your trial signup:

```bash
docker login quay.io -u '<trial-robot-account>' -p '<trial-token>'
```

Then start everything:

```bash
./scripts/run.sh
```

The script waits for each service to become healthy before printing access URLs:

| Service | URL | Credentials |
|---------|-----|-------------|
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |
| Dremio UI | http://localhost:9047 | Created on first login |
| Jupyter | http://localhost:8888 | No password |

On first launch, open the Dremio UI at http://localhost:9047 to complete the initial setup:

1. **Accept the EULA**
2. **Create an admin account** -- choose a username and password (you will need these for the notebook)
3. **Enter your Dremio license key** -- if prompted, paste the license key from your trial registration

The AIStor license is handled automatically through the `MINIO_LICENSE` environment variable in `.env` -- no manual UI step is needed for MinIO.

## Creating Tables with PyIceberg

The included Jupyter notebook handles the full workflow. Here is what it does and why.

### Warehouse and Namespace

AIStor Tables organizes data into warehouses and namespaces. A warehouse is the top-level container (analogous to a database server), and namespaces organize tables within it (analogous to schemas).

```python
# Create warehouse
resp = make_request("POST", "/v1/warehouses", json.dumps({
    "name": "dremio-warehouse"
}))
```

```python
# Create namespace
resp = make_request("POST", "/v1/dremio-warehouse/namespaces", json.dumps({
    "namespace": ["sales"]
}))
```

These are SigV4-signed HTTP calls to the Iceberg REST API. The signing uses `s3tables` as the service name, which is the convention for Iceberg REST endpoints on AIStor.

### Writing Tables

PyIceberg connects to the same REST Catalog API:

```python
catalog = load_catalog(
    "aistor",
    type="rest",
    uri="http://minio:9000/_iceberg",
    warehouse="dremio-warehouse",
    **{
        "rest.sigv4-enabled": "true",
        "rest.signing-name": "s3tables",
        "rest.signing-region": "dummy",
        "s3.access-key-id": "minioadmin",
        "s3.secret-access-key": "minioadmin",
        "s3.endpoint": "http://minio:9000",
        "s3.path-style-access": "true",
    }
)
```

From here, creating and populating tables uses standard PyIceberg:

```python
customers_table = catalog.create_table("sales.customers", schema=customers_schema)
customers_table.append(customers_data)
```

The notebook creates three related tables -- `customers`, `products`, and `orders` -- to demonstrate multi-table joins later in Dremio.

### Iceberg Views

The notebook also creates an Iceberg View through the REST API. Views are stored in the catalog alongside tables and are visible to any query engine that connects:

```python
view_payload = {
    "name": "revenue_by_region",
    "schema": { ... },
    "view-version": {
        "representations": [{
            "type": "sql",
            "sql": "SELECT region, SUM(total_amount) as total_revenue ...",
            "dialect": "spark"
        }]
    }
}
```

## Connecting Dremio to AIStor Tables

This is where the configuration details matter. Dremio's `RESTCATALOG` source type needs two things: the catalog endpoint for metadata and the S3 endpoint for data files. Both point to the same MinIO server, but they use different URL formats.

### The Source Configuration

```python
source_config = {
    "entityType": "source",
    "name": "aistor_tables",
    "type": "RESTCATALOG",
    "config": {
        "restEndpointUri": "http://minio:9000/_iceberg",
        "propertyList": [
            {"name": "warehouse", "value": "dremio-warehouse"},
            {"name": "rest.sigv4-enabled", "value": "true"},
            {"name": "rest.signing-name", "value": "s3tables"},
            {"name": "rest.signing-region", "value": "dummy"},
            {"name": "rest.access-key-id", "value": "minioadmin"},
            {"name": "rest.secret-access-key", "value": "minioadmin"},
            {"name": "fs.s3a.endpoint", "value": "minio:9000"},
            {"name": "fs.s3a.path.style.access", "value": "true"},
            {"name": "fs.s3a.connection.ssl.enabled", "value": "false"},
            {"name": "fs.s3a.aws.credentials.provider",
             "value": "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider"},
            {"name": "dremio.s3.compat", "value": "true"}
        ],
        "secretPropertyList": [
            {"name": "fs.s3a.access.key", "value": "minioadmin"},
            {"name": "fs.s3a.secret.key", "value": "minioadmin"}
        ]
    }
}
```

A few things to note here, since they are common sources of configuration errors:

**`restEndpointUri` vs `fs.s3a.endpoint`**: The REST endpoint keeps the `http://` scheme because Dremio's HTTP client needs it. The S3A endpoint must be **just the host and port** -- `minio:9000`, not `http://minio:9000`. The Hadoop S3A library treats a scheme prefix as a hostname, which leads to `UnknownHostException: http: Name or service not known`.

**`dremio.s3.compat`**: This must be `true` for S3-compatible object storage. Without it, Dremio may attempt AWS-specific behaviors that MinIO does not support.

**`fs.s3a.connection.ssl.enabled`**: Must be `false` for non-TLS connections. If `fs.s3a.endpoint` has no scheme (as required) and this flag is not explicitly set to `false`, S3A defaults to HTTPS and the connection fails.

**REST credentials in `propertyList`**: The `rest.access-key-id` and `rest.secret-access-key` properties go in the regular `propertyList`, not in `secretPropertyList`. The S3 data credentials (`fs.s3a.access.key` / `fs.s3a.secret.key`) go in `secretPropertyList`.

## Querying from Dremio

Once the source is connected, tables are immediately available in Dremio's SQL editor. The naming convention is `source.namespace.table`:

```sql
SELECT * FROM aistor_tables.sales.customers;
```

Multi-table joins work as expected:

```sql
SELECT
    c.name AS customer,
    p.product_name,
    o.quantity,
    o.total_amount,
    o.order_date
FROM aistor_tables.sales.orders o
JOIN aistor_tables.sales.customers c ON o.customer_id = c.customer_id
JOIN aistor_tables.sales.products p ON o.product_id = p.product_id
ORDER BY o.total_amount DESC
LIMIT 20;
```

Dremio pushes predicates and projections down into the Parquet reads, so queries that filter on specific columns or date ranges benefit from Iceberg's metadata-driven pruning without any manual optimization.

Aggregations work directly:

```sql
SELECT
    region,
    COUNT(*) AS order_count,
    SUM(total_amount) AS total_revenue,
    ROUND(AVG(total_amount), 2) AS avg_order_value
FROM aistor_tables.sales.orders
WHERE status = 'Completed'
GROUP BY region
ORDER BY total_revenue DESC;
```

The notebook automates these queries via the Dremio REST API and displays results in pandas DataFrames, but the same SQL works in the Dremio UI at http://localhost:9047.

## What This Looks Like in Production

The Docker Compose setup in this tutorial uses a single MinIO node and a single Dremio node. A production deployment looks different in scale but not in architecture:

- **AIStor** runs as a distributed cluster with erasure coding across multiple nodes. The Iceberg REST Catalog scales with the storage cluster -- there is no separate catalog database to size or replicate.
- **Dremio** runs as a coordinator-executor cluster. Multiple executors read Parquet files in parallel from MinIO, and the coordinator merges results.
- The **connection configuration** is the same. The `RESTCATALOG` source properties are identical whether MinIO is a single Docker container or a 32-node production cluster. The only things that change are the endpoint URLs and credentials.

This is the operational advantage of having the catalog built into the storage layer. There is no Hive Metastore to upgrade, no PostgreSQL replica set for catalog state, no consistency concerns between the catalog and the data it describes. The catalog and the data are managed by the same system.

## Try It

The full working setup is in the companion repository:

```bash
git clone https://github.com/miniohq/aistor-tables-dremio.git
cd aistor-tables-dremio
# Set your license in .env
./scripts/run.sh
```

Open Jupyter at http://localhost:8888, run the `DremioAIStorTables.ipynb` notebook, and you will have tables on AIStor queryable from Dremio in a few minutes.
