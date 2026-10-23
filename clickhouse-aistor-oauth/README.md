# ClickHouse + AIStor OAuth Test

> [!NOTE]
> This repo is a self-contained demo environment. The integration uses
> standard OIDC, AIStor STS and the Iceberg REST catalog with OAuth. To run it
> in production, use your own identity provider, TLS and secrets in place of
> the demo defaults (see [Moving to production](#moving-to-production)).

A local test of ClickHouse reading and writing MinIO AIStor with OAuth-issued
credentials, so no long-lived keys are passed per query. It covers two paths:

1. **Plain objects (Parquet).** A Keycloak token goes to AIStor STS and comes
   back as temporary S3 credentials for ClickHouse's `s3()` table function.
2. **AIStor Tables (Iceberg).** ClickHouse's `DataLakeCatalog` engine connects
   to Iceberg REST catalogs with OAuth, which AIStor Tables supports. ClickHouse
   creates, writes and reads Iceberg tables through the AIStor catalog.

Tested with AIStor `RELEASE.2026-09-19T17-05-25Z.hotfix.4ef74f03d6f2` and
ClickHouse `26.9`.

## Prerequisites

- Docker with Compose v2
- An AIStor license. AIStor requires one; get one from
  [SUBNET](https://subnet.min.io).

## Quick start

```bash
cp .env.example .env        # set MINIO_LICENSE
./scripts/run.sh            # start everything and run the test
./scripts/run.sh test       # re-run the test
./scripts/run.sh down       # tear down and remove volumes
```

Expected result: `25/25 checks passed`.

## Path 1: Parquet via STS

```
Keycloak (OIDC IdP)
   │  1. client_credentials grant → JWT access token (claims: aud=aistor, policy=...)
   ▼
AIStor STS  (AssumeRoleWithWebIdentity)
   │  2. validates JWT against Keycloak JWKS, maps the `policy` claim to an AIStor policy
   │     → temporary AccessKeyId / SecretAccessKey / SessionToken
   ▼
ClickHouse  s3(url, key, secret, session_token, 'Parquet')
      3. reads/writes AIStor with the temporary credentials
```

## Path 2: AIStor Tables via the Iceberg REST catalog

```
ClickHouse DataLakeCatalog ──(client_credentials)──▶ token endpoint
        │                       either AIStor:   /_iceberg/v1/oauth/tokens
        │                       or Keycloak:     /realms/aistor/protocol/openid-connect/token
        │
        ├── Bearer token ──▶ AIStor Iceberg REST catalog (/_iceberg/v1, warehouse "analytics")
        │                    create table, commit snapshots, load metadata
        └── S3 keys ───────▶ AIStor data/metadata files
```

```sql
SET allow_database_iceberg = 1, allow_insert_into_iceberg = 1;

CREATE DATABASE aistor_tables
ENGINE = DataLakeCatalog('http://minio:9000/_iceberg/v1',
                         'clickhouse-svc', 'clickhouse-svc-secret')   -- storage keys
SETTINGS catalog_type       = 'rest',
         warehouse          = 'analytics',
         catalog_credential = 'clickhouse-svc:clickhouse-svc-secret', -- OAuth client
         oauth_server_uri   = 'http://minio:9000/_iceberg/v1/oauth/tokens',
         auth_scope         = 'catalog',
         storage_endpoint   = 'http://minio:9000';

CREATE TABLE aistor_tables.`demo.trips` (id Int64, day Date, fare Float64)
ENGINE = IcebergS3('http://minio:9000/analytics/demo/trips/',
                   'clickhouse-svc', 'clickhouse-svc-secret');

INSERT INTO aistor_tables.`demo.trips`
SELECT number, toDate('2026-01-01') + (number % 30), round(randCanonical() * 100, 2)
FROM numbers(10000);

SELECT count() FROM aistor_tables.`demo.trips`;
```

### Ingesting into ClickHouse

Paths 1 and 2 query data in place on AIStor. To keep hot data local to
ClickHouse, copy it into a MergeTree table:

```sql
CREATE TABLE default.trips_hot (id Int64, day Date, fare Float64)
ENGINE = MergeTree ORDER BY (day, id);

INSERT INTO default.trips_hot
SELECT id, day, fare FROM aistor_tables.`demo.trips`;
```

The demo uses open-source MergeTree. On ClickHouse Cloud the same statements
create SharedMergeTree tables.

To use Keycloak instead, set `oauth_server_uri` to
`http://keycloak:8080/realms/aistor/protocol/openid-connect/token`,
`catalog_credential` to `clickhouse-iceberg:clickhouse-iceberg-secret`, and
`auth_scope` to `openid`. AIStor validates the Keycloak JWT and applies the
`clickhouse-tables` policy named in its `policy` claim.

## Compatibility notes

Observed with the versions listed above. Newer releases may behave differently.

- **ClickHouse version.** Creating and inserting into Iceberg tables through
  a REST catalog worked on ClickHouse 26.9. On 25.8, `CREATE TABLE` in a
  `DataLakeCatalog` database didn't register the table in the catalog.
- **AIStor OAuth token endpoint.** `/_iceberg/v1/oauth/tokens` accepts the
  `client_credentials` grant. The client ID and secret are an AIStor access
  key and secret key.
- **Storage credentials.** In this setup the catalog doesn't return vended
  credentials on `loadTable`. ClickHouse therefore gets S3 keys through the
  `DataLakeCatalog(...)` engine arguments.
- **Dropping tables.** ClickHouse doesn't add the catalog prefix to drop
  requests on prefixed REST catalogs (tracked in
  [ClickHouse#112234](https://github.com/ClickHouse/ClickHouse/issues/112234)).
  Until that's resolved, the test drops tables through the catalog API.
- **Table path.** ClickHouse writes an initial `v1.metadata.json` under the
  path given to `IcebergS3(...)`. The catalog assigns the table's actual
  location, and data and commits land there. ClickHouse treats an existing
  metadata file at that path as an existing table, so the test uses a unique
  path per run.

## What the test checks

Sections 1–4 cover Parquet via STS:

| # | Check |
|---|-------|
| 1 | Keycloak issues a token for client `clickhouse` |
| 2 | AIStor STS exchanges it for temporary credentials |
| 3 | A tampered token is rejected (`InvalidIdentityToken`) |
| 4 | ClickHouse writes 10k rows of Parquet to `s3://clickhouse-oauth/trips/` |
| 5 | ClickHouse reads them back with a glob |
| 6 | The same key/secret **without** the session token is denied |
| 7 | Client `clickhouse-ro` gets credentials with policy `clickhouse-readonly` |
| 8 | Those readonly credentials can read |
| 9 | Those readonly credentials cannot write |

Sections 5–6 cover AIStor Tables. Each runs once with AIStor's token endpoint
and once with Keycloak:

| Check |
|-------|
| ClickHouse attaches the catalog and lists tables |
| ClickHouse creates `demo.trips` (or `demo.trips_kc`) |
| ClickHouse inserts 10k rows and reads them back |
| The AIStor catalog API shows the committed snapshot |
| A wrong OAuth client secret is rejected (AIStor endpoint only) |

Section 7 covers ingest:

| Check |
|-------|
| ClickHouse creates MergeTree table `default.trips_hot` |
| ClickHouse copies `aistor_tables.demo.trips` into it |
| Row count, fare total and date range match the Iceberg source |

## Components

| Service | URL | Login (demo values) |
|---------|-----|-------|
| Keycloak | http://localhost:8080 | admin / admin |
| AIStor console | http://localhost:9001 | minioadmin / minioadmin |
| ClickHouse HTTP | http://localhost:8123 | default / clickhouse |

Keycloak clients (realm `aistor`):

| Client | Secret | `policy` claim |
|--------|--------|----------------|
| `clickhouse` | `clickhouse-secret` | `clickhouse-readwrite` |
| `clickhouse-ro` | `clickhouse-ro-secret` | `clickhouse-readonly` |
| `clickhouse-iceberg` | `clickhouse-iceberg-secret` | `clickhouse-tables` |

The AIStor user `clickhouse-svc` / `clickhouse-svc-secret` has the
`clickhouse-tables` policy. It's the OAuth client for AIStor's token endpoint
and also supplies the storage keys.

## Policies

Each policy is scoped to a single bucket or warehouse and grants only what the
test uses. Use them as a starting point:

| Policy | Grants |
|--------|--------|
| [clickhouse-readwrite](policies/clickhouse-readwrite.json) | `ListBucket`, `GetObject`, `PutObject` on `clickhouse-oauth` |
| [clickhouse-readonly](policies/clickhouse-readonly.json) | `ListBucket`, `GetObject` on `clickhouse-oauth` |
| [clickhouse-tables](policies/clickhouse-tables.json) | Catalog read, create, commit and drop on warehouse `analytics`, plus object read, write and delete on its data |

ClickHouse needs `s3:ListBucket` to expand globs such as `trips/*.parquet`.
AIStor's built-in `readonly` policy doesn't include it. If ClickHouse should
not drop tables, remove `s3tables:DeleteTable` and `s3:DeleteObject` from
`clickhouse-tables`. The test uses both to clean up between runs.

## Moving to production

The demo defaults are for a single machine. In production:

- Run Keycloak (or your IdP) in production mode with TLS and a real database.
- Serve AIStor and ClickHouse over TLS, and replace every demo secret.
- Store client secrets in a secrets manager, not in SQL or compose files.
  ClickHouse masks secrets in query logs, but they still sit in the
  `CREATE DATABASE` statement.
- Plan for expiry. STS credentials last `DurationSeconds` (3600 in the test),
  and OAuth tokens expire too.

## Configuration notes

- **AIStor OIDC** is set with environment variables in
  [docker-compose.yaml](docker-compose.yaml): `MINIO_IDENTITY_OPENID_CONFIG_URL`,
  `MINIO_IDENTITY_OPENID_CLIENT_ID=aistor` and `MINIO_IDENTITY_OPENID_CLAIM_NAME=policy`.
- **Audience.** Every Keycloak client adds `aistor` to the token `aud`. That
  lets one AIStor OIDC config accept tokens from several clients.
- **Issuer.** Keycloak's hostname is pinned to `http://keycloak:8080` so the
  token `iss` matches what AIStor discovers. Tokens therefore have to be
  requested inside the Docker network (the `tester` container does this). To
  request them from your machine, add `127.0.0.1 keycloak` to `/etc/hosts`.
- **Apple Silicon.** Keycloak runs with `-XX:UseSVE=0` to avoid a JVM SIGILL
  crash under Docker on newer M-series chips.

## Manual query example

```bash
docker compose run --rm --entrypoint python tester -c "
import sys; sys.path.insert(0, '/scripts'); import test_oauth as t
_, c = t.assume_role(t.get_token(*t.RW_CLIENT))
print(c)"
```

Then in ClickHouse:

```sql
SELECT count()
FROM s3('http://minio:9000/clickhouse-oauth/trips/*.parquet',
        '<AccessKeyId>', '<SecretAccessKey>', '<SessionToken>', 'Parquet');
```

## Trademarks

ClickHouse is a trademark of ClickHouse, Inc. This project isn't affiliated
with or endorsed by ClickHouse, Inc.
