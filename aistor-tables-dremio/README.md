# Dremio + AIStor Tables

[![MinIO](https://raw.githubusercontent.com/minio/minio/master/.github/logo.svg?sanitize=true)](https://min.io)

Query Apache Iceberg tables on MinIO AIStor using Dremio Enterprise. This tutorial demonstrates how to connect Dremio to AIStor's built-in Iceberg REST Catalog, create tables with PyIceberg, and run SQL analytics.

> **Enterprise Licenses Required**: Both AIStor Tables and Dremio Enterprise require valid licenses.
> - **AIStor**: Contact [enterprise@min.io](mailto:enterprise@min.io) or visit [SUBNET](https://subnet.min.io)
> - **Dremio Enterprise**: The RESTCATALOG source type is **not available** in Dremio Community Edition. Get a [30-day trial](https://www.dremio.com/get-started/).

## Architecture

```
┌─────────────────┐     ┌──────────────────────────────┐
│     Dremio      │     │      MinIO AIStor             │
│  (Query Engine) │     │                               │
│                 ├────►│  /_iceberg  (REST Catalog API) │
│  SQL Analytics  │     │      Table metadata, schemas  │
│                 │     │                               │
│                 ├────►│  S3 API    (Data Access)       │
│                 │     │      Parquet data files       │
└─────────────────┘     └──────────────────────────────┘
        ▲
        │
┌───────┴─────────┐
│    Jupyter       │
│  (PyIceberg)     │
│  Table creation  │
│  & data loading  │
└─────────────────┘
```

## Prerequisites

- Docker and Docker Compose
- AIStor enterprise license (for Tables feature)
- Dremio Enterprise trial or license (for RESTCATALOG source type)

## Quick Start

```bash
git clone <this-repo>
cd aistor-tables-dremio
```

1. **Set your AIStor license** in `.env`:
   ```bash
   MINIO_LICENSE=your-license-key-here
   ```

2. **Login to Dremio Enterprise registry** (required to pull the Enterprise image):
   ```bash
   docker login quay.io -u '<trial-robot-account>' -p '<trial-token>'
   ```
   You'll receive these credentials when you sign up for the [Dremio Enterprise trial](https://www.dremio.com/get-started/).

3. **Start all services:**
   ```bash
   ./scripts/run.sh
   ```

4. **Complete Dremio first-time setup** at http://localhost:9047:
   - Accept the EULA
   - Create your admin account (username and password)

5. **Open Jupyter** at http://localhost:8888 and run `notebooks/DremioAIStorTables.ipynb`

## Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |
| Dremio UI | http://localhost:9047 | Create on first login |
| Jupyter Notebook | http://localhost:8888 | No password |

## What the Notebook Does

The `DremioAIStorTables.ipynb` notebook walks through:

1. **Creates a warehouse and namespace** on AIStor via the Iceberg REST API
2. **Creates three Iceberg tables** using PyIceberg:
   - `sales.customers` (20 rows)
   - `sales.products` (15 rows)
   - `sales.orders` (50 rows)
3. **Creates an Iceberg View** (`sales.revenue_by_region`)
4. **Configures the Dremio source** via API (RESTCATALOG type)
5. **Runs SQL queries** through Dremio, including multi-table joins

## Dremio Source Configuration

The notebook configures the Dremio source automatically. If you want to add it manually via the Dremio UI, use these settings:

**Source Type:** Iceberg REST Catalog (`RESTCATALOG`)

### General Settings

| Setting | Value |
|---------|-------|
| Name | `aistor_tables` |
| REST Endpoint URI | `http://minio:9000/_iceberg` |

### Properties (propertyList)

| Property | Value |
|----------|-------|
| `warehouse` | `dremio-warehouse` |
| `rest.sigv4-enabled` | `true` |
| `rest.signing-name` | `s3tables` |
| `rest.signing-region` | `dummy` |
| `rest.access-key-id` | `minioadmin` |
| `rest.secret-access-key` | `minioadmin` |
| `fs.s3a.endpoint` | `minio:9000` |
| `fs.s3a.path.style.access` | `true` |
| `fs.s3a.connection.ssl.enabled` | `false` |
| `fs.s3a.aws.credentials.provider` | `org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider` |
| `dremio.s3.compat` | `true` |

### Secret Properties (secretPropertyList)

| Property | Value |
|----------|-------|
| `fs.s3a.access.key` | `minioadmin` |
| `fs.s3a.secret.key` | `minioadmin` |

> **Note:** `rest.access-key-id` and `rest.secret-access-key` go in **propertyList**, not secretPropertyList. The `dremio.s3.compat` property is required for MinIO and other S3-compatible storage.

## Example SQL Queries

Once the source is configured and tables are populated, try these in the Dremio SQL editor:

```sql
-- Customer list
SELECT * FROM aistor_tables.sales.customers;

-- Top orders with customer and product details
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

-- Revenue by region
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

## Troubleshooting

### RESTCATALOG source type not available

The RESTCATALOG source type is only available in **Dremio Enterprise**, not the Community (OSS) edition. If you don't see it in the source type dropdown, verify you're running the Enterprise image (`quay.io/dremio/dremio-enterprise`).

### Source status is "good" but queries fail

The most common issue is the `fs.s3a.endpoint` property. It must be **just the host:port** without the `http://` scheme:

| Setting | Correct | Wrong |
|---------|---------|-------|
| `fs.s3a.endpoint` | `minio:9000` | `http://minio:9000` |

The `http://` prefix causes S3A to treat "http" as a hostname, resulting in `UnknownHostException: http: Name or service not known`.

You must also set:
- `fs.s3a.connection.ssl.enabled` to `false` for non-TLS connections
- `dremio.s3.compat` to `true` for MinIO/S3-compatible storage

### Dremio takes a long time to start

Dremio typically needs 60-90 seconds to fully initialize. The startup script waits for this automatically.

### Dremio first-time setup

On first launch, Dremio requires you to create an admin user account. You can do this via the UI at http://localhost:9047, or automate it with Python (recommended — avoids shell escaping issues with special characters in passwords):

```python
import requests

requests.put("http://localhost:9047/apiv2/bootstrap/firstuser",
    json={
        "userName": "admin",
        "firstName": "Admin",
        "lastName": "User",
        "email": "admin@example.com",
        "password": "your-password-here",
        "role": "admin"
    }
)
```

> **Note:** Avoid using `curl -d` with passwords containing `!` or other shell-special characters — bash history expansion can silently corrupt the value. Python bypasses this entirely.

After creating your account, update `DREMIO_USER` and `DREMIO_PASS` in notebook cell 7 before running the Dremio sections.

### Warehouse not found errors

Make sure you run the notebook cells in order. The warehouse must be created before PyIceberg can connect, and before Dremio can query tables.

### Connection refused from Dremio

Inside Docker Compose, services communicate using container names. Dremio reaches MinIO at `minio:9000`, not `localhost:9000`.

### Dremio API authentication

Dremio uses a non-standard authorization header format: `_dremio{TOKEN}` (not `Bearer {TOKEN}`). The notebook handles this automatically.

## Stop Services

```bash
./scripts/run.sh --stop
```

## License

- MinIO AIStor Tables requires a valid enterprise license. Contact [enterprise@min.io](mailto:enterprise@min.io) for licensing.
- Dremio Enterprise requires a valid license. Get a [30-day trial](https://www.dremio.com/get-started/).
