# Workshop 01 — AIStor Tables: Create, Query, and Analyse with Trino

A hands-on lab that builds a full Iceberg data pipeline on MinIO AIStor Tables,
then queries it with both PyIceberg and Trino SQL — all code generated with
**OpenCode** prompts.

## What you build

```
MinIO AIStor (saleswarehouse)
  └── retail  (namespace)
        └── sales  (Iceberg table — 25 sales transactions)
              ├── Parquet data files
              └── Iceberg metadata / snapshots

Trino  ──► REST catalog ──► same warehouse
PyIceberg ──► same REST catalog
```

## Stack

| Component | Version | Role |
|-----------|---------|------|
| MinIO AIStor | RELEASE.2025-12-20 | S3 + Iceberg REST catalog |
| Trino | latest | Distributed SQL engine |
| PyIceberg | ≥ 0.7 | Python Iceberg client |
| PyArrow | ≥ 14 | In-memory columnar format |
| Jupyter Lab | latest | Notebook runtime |
| OpenCode | any | AI code generation (terminal) |

## Quick start

### 1. Prerequisites

- Docker + Docker Compose
- An AIStor license file (copy it to `license/minio.license`)
- OpenCode CLI installed: https://opencode.ai

### 2. Start the stack

```bash
cd training/courses/aistor-tables/workshop-01
docker compose up -d
```

Wait ~30 s for all services to become healthy, then open:

| URL | Service |
|-----|---------|
| http://localhost:9001 | MinIO Console (minioadmin / minioadmin) |
| http://localhost:8090 | Trino Web UI |
| http://localhost:8888 | Jupyter Lab |

### 3. Open the notebook

Navigate to **http://localhost:8888**, open `work/Workshop-01.ipynb`,
and run cells top-to-bottom.

Each **💡 OpenCode Prompt** box shows the exact terminal command you would
use to generate the cell's code with OpenCode. The code cell beneath it
is the output — ready to run.

### 4. Tear down

```bash
docker compose down
```

To also remove all stored data:

```bash
docker compose down -v
rm -rf minio-data/*
```

## Directory layout

```
workshop-01/
├── docker-compose.yaml          # MinIO + Trino + Jupyter
├── trino-config/
│   ├── iceberg.properties       # Trino REST catalog config for AIStor
│   └── jvm.config               # JVM flags (SigV4 + heap)
├── notebooks/
│   └── Workshop-01.ipynb        # The lab notebook
├── minio-data/                  # Bind-mounted MinIO data volume
├── license/
│   └── minio.license            # ← put your AIStor license here
└── README.md
```

## Notebook structure

| Part | What you do |
|------|-------------|
| 0 | Environment config (auto-detects Docker vs local) |
| 1 | OpenCode → create `saleswarehouse` via SigV4 REST API |
| 2 | OpenCode → connect PyIceberg `RestCatalog` |
| 3 | OpenCode → create `retail` namespace |
| 4 | OpenCode → create `retail.sales` table (10 columns) |
| 5 | OpenCode → insert 25 sample sales rows |
| 6 | OpenCode → PyIceberg scans + row filters |
| 7 | OpenCode → Trino SQL: counts, totals, rankings, time trends |
| 8 | OpenCode → inspect Iceberg snapshot metadata |

## Key concepts demonstrated

- **Warehouse creation** — AIStor-specific SigV4 `POST /_iceberg/v1/warehouses`
- **SigV4 signing** — service name must be `s3tables` (not `s3`)
- **PyIceberg RestCatalog** — `rest.sigv4-enabled`, `rest.signing-name`
- **Trino REST catalog** — `iceberg.rest-catalog.security=SIGV4`
- **OpenCode workflow** — prompt → generate → run → iterate

## Relationship to other labs

This workshop builds on the seed patterns in `intro/Pyiceberg.ipynb` and
adds Trino + OpenCode on top. After completing this lab, explore:

- `etl-pipeline/` — batch ETL with real NYC taxi data
- `kafka-pipeline/` — real-time streaming from OpenSky flight data
