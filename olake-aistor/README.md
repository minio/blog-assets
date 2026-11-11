# OLake + AIStor Tables: Postgres CDC to Iceberg

> [!NOTE]
> This repo is a self-contained demo environment. To run it in production, use
> your own TLS, secrets and scoped users in place of the demo defaults (see
> [Moving to production](#moving-to-production)).

A local, end-to-end test of [OLake](https://olake.io) replicating Postgres into
Apache Iceberg tables in MinIO AIStor. OLake reads the Postgres write-ahead log
(WAL) directly and commits each sync through **AIStor Tables**, the Iceberg REST
catalog built into AIStor at `/_iceberg`. You don't need Kafka, Debezium, Spark
or a separate catalog server.

```
Postgres 16 (wal_level=logical)
   │  slot olake_slot (pgoutput) + publication olake_pub
   ▼
OLake (olakego/source-postgres)
   │  1st sync: full backfill     later syncs: inserts/updates/deletes from the WAL
   │
   ├── Parquet data + positional delete files ──(S3 API)──▶ AIStor warehouse "analytics"
   └── snapshot commit ──(SigV4, service s3tables)──▶ AIStor Tables  http://minio:9000/_iceberg
                                                            │
                         test: catalog API (OAuth) + DuckDB (SigV4) read the same tables
```

Tested 2026-11-11 with AIStor `RELEASE.2026-09-19T17-05-25Z.hotfix.175a5c7f2f85`,
OLake `v1.1.2`, Postgres 16 and DuckDB 1.5.5 (24/24 checks across both
syncs). OLake's own
[Postgres to AIStor guide](https://olake.io/docs/integrations/postgres-to-aistor/)
was tested with OLake v1.1.1 against the same AIStor release.

## Prerequisites

- Docker with Compose v2
- An AIStor license. AIStor requires one; get one from
  [SUBNET](https://subnet.min.io).

## Quick start

```bash
cp .env.example .env        # set MINIO_LICENSE
./scripts/run.sh            # start everything, sync twice, verify each sync
./scripts/run.sh test       # re-run the verification
./scripts/run.sh sync       # run another CDC sync after changing Postgres
./scripts/run.sh down       # tear down, remove volumes and OLake output
```

`run.sh` expects a fresh stack. If an earlier run exists, it stops and tells
you to run `test` to re-verify it, or `down` to start over.

The demo pins the AIStor release it was tested with. To try a newer one,
set the image for a single run, or change `MINIO_TEST_IMAGE` in `.env`:

```bash
./scripts/run.sh down
MINIO_TEST_IMAGE=quay.io/minio/aistor/minio:<tag> ./scripts/run.sh
```

Releases are listed on [quay.io](https://quay.io/repository/minio/aistor/minio?tab=tags).
The `latest` tag can trail the newest hotfix, so name the exact tag.

`run.sh` does the following:

1. Starts Postgres and AIStor, then creates the `analytics` warehouse and the
   `olake-svc` user.
2. Runs `olake check` against the source (connection, `wal_level`, slot,
   publication) and the destination (writes and drops a test table).
3. Runs `olake discover`, then sets every stream to `cdc` with positional
   deletes ([scripts/select_streams.py](scripts/select_streams.py)).
4. **First sync**: backfills 8 customers and 40 orders, then verifies them.
5. Applies [postgres/changes.sql](postgres/changes.sql): 7 inserts and 5
   updates.
6. **Second sync**: replays the 12 changes from the WAL, continuing from
   `olake/state.json`, then verifies the result.

## OLake configuration

All OLake files live in [olake/](olake/) and are mounted at `/mnt/config` in
the connector container.

**Source** ([olake/source.json](olake/source.json)): a `replication_slot` in
`update_method` turns on CDC. OLake reads the plugin from the slot. With
`pgoutput`, a publication is required.

**Destination** ([olake/destination.json](olake/destination.json)): AIStor is
a *Generic REST* catalog with SigV4:

```json
{
  "type": "ICEBERG",
  "writer": {
    "catalog_type": "rest",
    "rest_catalog_url": "http://minio:9000/_iceberg",
    "iceberg_s3_path": "analytics",
    "rest_auth_type": "sigv4",
    "rest_signing_v_4": true,
    "rest_signing_name": "s3tables",
    "rest_signing_region": "us-east-1",
    "aws_region": "us-east-1",
    "aws_access_key": "olake-svc",
    "aws_secret_key": "olake-svc-secret",
    "s3_endpoint": "http://minio:9000",
    "s3_path_style": true,
    "s3_use_ssl": false
  }
}
```

[olake/destination-oauth.json](olake/destination-oauth.json) is an alternative
that authenticates with OAuth2 `client_credentials` against AIStor's token
endpoint (`/_iceberg/v1/oauth/tokens`) instead of SigV4. The client ID and
secret are an AIStor access key and secret key, and the scope is `catalog`.
`olake check --destination` passes with it: OLake creates, writes to and drops
a test table through the catalog. The full sync test uses SigV4. To use OAuth,
pass `--destination /mnt/config/destination-oauth.json` to `olake check` and
`olake sync`.

OLake logs a deprecation warning about `rest.sigv4-enabled` with either file.
The warning is harmless.

### Gotchas

- **`rest_catalog_url` doesn't include `/v1`.** The Iceberg client adds
  `v1/...` itself. The OAuth token URI is the exception: it's the full
  `/_iceberg/v1/oauth/tokens`.
- **`iceberg_s3_path` is the warehouse name** (`analytics`), not an `s3://`
  path. AIStor decides where the files go.
- **The key is `rest_signing_v_4`**, with an underscore before the 4. OLake
  silently ignores a misspelled key. S3 writes then succeed, but AIStor rejects
  the catalog calls.
- **Use `catalog_type: "rest"`, not `"s3tables"`.** The `s3tables` preset is
  for AWS S3 Tables.
- **The destination file needs the wrapper**
  `{"type": "ICEBERG", "writer": {...}}`. Without it, OLake fails with
  `invalid destination type`.
- **Namespace naming.** OLake names the namespace
  `<prefix>_<database>_<schema>`. `run.sh` passes
  `--destination-database-prefix olake`, which gives `olake_shop_public`.

## What the test checks

[scripts/test_olake.py](scripts/test_olake.py) checks the tables two
independent ways after each sync.

**AIStor catalog API (OAuth token from AIStor):**

| Check |
|-------|
| AIStor issues a catalog token for `olake-svc` |
| OLake created the `olake_shop_public` namespace |
| `customers` and `orders` are registered |
| Each table has a current snapshot (2 or more after the change sync) |

**DuckDB, attached to the same catalog with SigV4:**

| Check | After first sync | After change sync |
|-------|------------------|-------------------|
| `customers` rows, one per id | 8 | 10 |
| `orders` rows, one per id | 40 | 45 |
| `orders.amount` total | 6345.00 | 8188.39 |
| OLake columns `_olake_id`, `_op_type`, `_cdc_lsn` present | ✓ | ✓ |
| Liam Carter is `gold`, Zara Ali is in Abu Dhabi | | ✓ |
| Orders 1, 4, 7 are `delivered` | | ✓ |
| `_op_type` shows 2 inserts (`c`) and 2 updates (`u`) on `customers` | | ✓ |

## Components

| Service | URL | Login (demo values) |
|---------|-----|-------|
| AIStor console | http://localhost:9001 | minioadmin / minioadmin |
| AIStor S3 + Iceberg REST | http://localhost:9000, `/_iceberg` | olake-svc / olake-svc-secret |
| Postgres | localhost:5432, db `shop` | postgres / postgres (OLake uses olake / olake) |

The AIStor user `olake-svc` has the [olake-tables](policies/olake-tables.json)
policy. The policy grants catalog access to warehouse `analytics` (including
namespace create and delete, which `olake check` uses for its test table) and
object read, write and delete on the warehouse's data.

## Run OLake by hand

```bash
docker compose run --rm olake spec                                    # source spec
docker compose run --rm olake check --config /mnt/config/source.json
docker compose run --rm olake check --destination /mnt/config/destination.json
docker compose run --rm olake discover --config /mnt/config/source.json --destination-database-prefix olake
docker compose run --rm olake sync \
  --config /mnt/config/source.json \
  --destination /mnt/config/destination.json \
  --available-streams /mnt/config/available_streams.json \
  --selected-streams /mnt/config/selected_streams.json \
  --destination-database-prefix olake \
  --state /mnt/config/state.json      # omit on the very first sync
```

OLake writes logs to `olake/sync_<timestamp>/olake.log` and writes
`state.json` and `stats.json` to `olake/`.

## Query from your machine with DuckDB

```sql
INSTALL iceberg; LOAD iceberg;
CREATE SECRET aistor_s3 (TYPE S3, KEY_ID 'olake-svc', SECRET 'olake-svc-secret',
  ENDPOINT 'localhost:9000', URL_STYLE 'path', USE_SSL 0, REGION 'us-east-1');
ATTACH 'analytics' AS aistor (TYPE ICEBERG, ENDPOINT 'http://localhost:9000/_iceberg',
  AUTHORIZATION_TYPE 'sigv4', SIGV4_SERVICE 's3tables', SIGV4_REGION 'us-east-1',
  SECRET aistor_s3);
SELECT id, name, city, tier, _op_type FROM aistor.olake_shop_public.customers ORDER BY id;
```

## Moving to production

- Serve AIStor and Postgres over TLS (`s3_use_ssl: true`, `ssl.mode` of
  `verify-full`), and replace every demo secret.
- Run syncs on a schedule. The replication slot pins WAL on Postgres until
  OLake reads it, so a paused job makes the WAL grow. Alert on
  `pg_replication_slots` lag.
- Use one slot and one publication per OLake job.
- Set `REPLICA IDENTITY FULL` on replicated tables (the demo does). Without
  it, OLake warns that unchanged TOAST values can be lost on CDC updates.
- OLake doesn't compact tables. Plan table maintenance separately.
- OLake writes Postgres `numeric` columns as Iceberg `double`.

## Trademarks

OLake is a project of Datazip, Inc. This project isn't affiliated with or
endorsed by Datazip, Inc.
