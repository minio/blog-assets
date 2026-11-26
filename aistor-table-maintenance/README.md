# AIStor Tables: automated table maintenance

> [!NOTE]
> This repo is a self-contained demo environment. It uses demo credentials and
> undocumented settings that make maintenance run within minutes. Don't copy
> those settings to production (see [Moving to production](#moving-to-production)).

Streaming ingest and change data capture (CDC) jobs commit small batches
often. Each commit to an Apache Iceberg table adds a data file, manifest files
and a snapshot, so a streaming table soon has:

- **many small files**, which slow down every query that has to open them
- **a long trail of snapshots**, so table metadata grows without limit
- **unreferenced files** that nothing points to anymore but that still take up storage

The usual fix is a set of scheduled Spark jobs (`rewrite_data_files`,
`expire_snapshots`, `remove_orphan_files`) that someone has to write, schedule
and monitor. **AIStor Tables**, the Iceberg-native tables with a built-in REST
catalog in MinIO AIStor, runs this maintenance itself. You set one policy per
warehouse (or per table), and AIStor runs three background jobs:

| Job | `--type` | What it does |
|---|---|---|
| Compaction | `icebergCompaction` | Rewrites small data files into fewer, larger files |
| Snapshot expiration | `icebergSnapshotManagement` | Expires snapshots older than a maximum age, always keeping a minimum number |
| Unreferenced file removal | `icebergUnreferencedFileRemoval` | Marks files that no snapshot references as noncurrent, then deletes them after a retention period |

```
 writer (PyIceberg, ingest-svc)                       AIStor admin (mc)
   │  200 small appends = 200 commits                    │  mc table warehouse maintenance set ...
   ▼                                                     ▼
 AIStor Tables  /_iceberg ─── warehouse "lakehouse" ─── maintenance policy
   │                             └─ iot.sensor_readings
   │                                  200 files · 200 snapshots · 801 objects
   │       background jobs: compaction → snapshot expiration → unreferenced file removal
   ▼
 iot.sensor_readings   1 file · 5 snapshots · 510 objects   (same 100,000 rows)
   ▲
 DuckDB reads through the same catalog, before and after
```

Tested 2026-11-16 with AIStor `RELEASE.2026-09-19T17-05-25Z.hotfix.175a5c7f2f85`,
PyIceberg 0.10.0 and DuckDB 1.5.5. The scripted run takes about 2.5 minutes,
including the image build on first run.

## Prerequisites

- Docker with Compose v2
- An AIStor license. AIStor requires one; get one from
  [SUBNET](https://subnet.min.io).

## Quick start

```bash
cp .env.example .env        # set MINIO_LICENSE
./scripts/run.sh            # start AIStor, make a messy table, turn on maintenance, watch
./scripts/run.sh status     # policy, job status, metrics and table health
./scripts/run.sh down       # tear down and remove volumes
```

`./scripts/run.sh` does the following:

1. Starts AIStor, then creates the `lakehouse` warehouse and the `ingest-svc`
   user.
2. Streams 200 micro-batches of 500 rows into `iot.sensor_readings`, one Iceberg
   commit per batch ([scripts/make_mess.py](scripts/make_mess.py)).
3. Records the table's health ([scripts/inspect_table.py](scripts/inspect_table.py)):
   snapshots, data files, every object under the table's location, and the
   median time of a DuckDB aggregate query.
4. Sets a maintenance policy on the warehouse as the AIStor admin.
5. Polls the table until all three jobs have visibly changed it, then prints
   before and after.

A typical run:

```
==> Watching AIStor clean up the table
  04:36:06  files  200  snapshots  200  objects   801   [ ] compaction  [ ] snapshot expiration  [ ] unreferenced file removal
  04:36:21  files  200  snapshots    5  objects   703   [ ] compaction  [x] snapshot expiration  [ ] unreferenced file removal
  04:36:36  files    1  snapshots    6  objects   708   [x] compaction  [x] snapshot expiration  [ ] unreferenced file removal
  ...
  04:37:21  files    1  snapshots    5  objects   510   [x] compaction  [x] snapshot expiration  [x] unreferenced file removal

                                before       after
  rows                         100,000     100,000
  snapshots                        200           5
  data files (current)             200           1
  avg data file KB                 9.4     1,046.5
  objects on storage               801         510
  storage MB                      19.0       14.81
  DuckDB query ms                 90.2        16.1
```

These numbers come from one local run on a laptop. Treat them as an
illustration of what maintenance does, not as a benchmark.

## Present from the notebook

[notebooks/table-maintenance.ipynb](notebooks/table-maintenance.ipynb) walks
through the same story one step at a time, with a before-and-after chart, the
table's snapshot history and a time-travel query.

```bash
./scripts/run.sh down       # start from a clean stack
./scripts/run.sh start      # AIStor + Jupyter at http://localhost:8888
```

The notebook runs as `ingest-svc`. When it reaches step 3, switch to a terminal
and run `./scripts/run.sh policy` as the admin. That pause is part of the
story: the writer can read the maintenance policy but can't change it.

## The maintenance policy

[scripts/run.sh](scripts/run.sh) `policy` sets one policy on the warehouse. Every
table in `lakehouse` inherits it:

```bash
mc table warehouse maintenance set aistor/lakehouse --type icebergCompaction \
   --target-file-size-mb 128 --interval 1
mc table warehouse maintenance set aistor/lakehouse --type icebergSnapshotManagement \
   --max-age-hours 0 --min-snapshots 5 --interval 1
mc table warehouse maintenance set aistor/lakehouse --type icebergUnreferencedFileRemoval \
   --unreferenced-days 0 --noncurrent-days 1 --interval 1
```

`--interval` is in minutes. Other useful commands:

```bash
docker compose run --rm mc table warehouse maintenance info aistor/lakehouse
docker compose run --rm mc table maintenance info   aistor/lakehouse/iot/sensor_readings
docker compose run --rm mc table maintenance status aistor/lakehouse/iot/sensor_readings
# Per-table override, for example to exclude one table from compaction:
docker compose run --rm mc table maintenance update aistor/lakehouse/iot/sensor_readings \
   --type icebergCompaction --status disabled
```

You can also set the policy in the AIStor console at http://localhost:9001.

### What stays on storage, and why

After the run, the table still has about 500 objects. That's expected:

- **The original 200 small files.** `--min-snapshots 5` keeps four snapshots
  from before compaction, and those still point at the small files. AIStor
  won't delete a file that a kept snapshot can still read, so time travel to
  those snapshots keeps working. The files go once those snapshots expire.
- **About 100 `metadata.json` files.** These are Iceberg's own metadata log,
  which the writer controls (`write.metadata.previous-versions-max`, default
  100, and `write.metadata.delete-after-commit.enabled`).

Compaction shows up in the table history as an Iceberg `replace` snapshot
committed by AIStor, with the row count unchanged.

## How the demo makes maintenance run in minutes

Out of the box, AIStor checks for due maintenance work once a day, and its
retention settings have floors of 1 hour (snapshot age) and 1 day (unreferenced
and noncurrent files). The demo changes this in
[docker-compose.yaml](docker-compose.yaml):

| Setting | Demo | Default | Documented |
|---|---|---|---|
| `MINIO_TABLES_MAINTENANCE_POLL_INTERVAL` | `1m` | `24h` | Yes, `mc admin config get <alias> tables` |
| `MINIO_TABLES_COMPACTION_INTERVAL` | `1m` | `24h` | Yes |
| `MINIO_TABLES_SNAPSHOT_EXPIRY_INTERVAL` | `1m` | `24h` | Yes |
| `MINIO_TABLES_UNREFERENCED_FILE_REMOVAL_INTERVAL` | `1m` | `24h` | Yes |
| `_MINIO_TABLE_MAINTENANCE_STARTUP_DELAY` | `10s` | longer than 15 minutes in testing | **No** |
| `_MINIO_TABLE_MAINTENANCE_MIN_SNAPSHOT_AGE_HOURS` | `0` | `1` | **No** |
| `_MINIO_TABLE_UFR_MIN_UNREFERENCED_DAYS` | `0` | `1` | **No** |

The three underscore settings are internal to AIStor and undocumented. They let
`--max-age-hours 0` and `--unreferenced-days 0` through, and start the
maintenance loops right after boot. Without the startup delay setting,
compaction stayed at `Not_Yet_Run` for more than 15 minutes in testing, even
though snapshot expiration ran within a minute. `--noncurrent-days` still has a
floor of 1 day, so files marked noncurrent in the demo are deleted permanently
a day later.

> [!WARNING]
> The `--interval` on a policy can't run more often than the server's
> `maintenance_poll_interval`. `mc` warns about this when you set a short
> interval.

## Permissions

The writer, `ingest-svc`, has [policies/lakehouse-ingest.json](policies/lakehouse-ingest.json):
catalog and data access to `lakehouse`, plus read-only maintenance actions. Only
the admin can set policies.

| Action | Who has it here |
|---|---|
| `s3tables:GetWarehouseMaintenanceConfiguration` | admin |
| `s3tables:PutWarehouseMaintenanceConfiguration` | admin |
| `s3tables:GetTableMaintenanceConfiguration` | admin, `ingest-svc` |
| `s3tables:PutTableMaintenanceConfiguration` | admin (also removes overrides) |
| `s3tables:GetTableMaintenanceJobStatus` | admin, `ingest-svc` |

## Monitoring

AIStor exports maintenance metrics at `/minio/metrics/v3/tables`. The demo
makes them public (`MINIO_PROMETHEUS_AUTH_TYPE=public`); in production, scrape
with a token.

```bash
curl -s localhost:9000/minio/metrics/v3/tables | grep -E '^minio_tables_(compaction|maintenance)'
```

| Metric | Meaning |
|---|---|
| `minio_tables_compaction_files_rewritten_total` | Small files rewritten by compaction |
| `minio_tables_compaction_files_added_total` | Compacted files written |
| `minio_tables_maintenance_snapshots_expired_total` | Snapshots expired |
| `minio_tables_maintenance_orphan_files_marked_total` | Unreferenced files marked noncurrent |
| `minio_tables_maintenance_cycle_total{type=...}` | Maintenance cycles per job type |
| `minio_tables_maintenance_errors_total{type=...}` | Failed runs per job type |

## Components

| Service | URL | Login (demo values) |
|---|---|---|
| AIStor console | http://localhost:9001 | minioadmin / minioadmin |
| AIStor S3 API + Iceberg REST catalog | http://localhost:9000, `/_iceberg` | ingest-svc / ingest-svc-secret |
| Jupyter (`./scripts/run.sh start` or `lab`) | http://localhost:8888 | no token |

| File | Purpose |
|---|---|
| [scripts/run.sh](scripts/run.sh) | `up`, `start`, `mess [N]`, `policy`, `status`, `lab`, `down` |
| [scripts/lakehouse.py](scripts/lakehouse.py) | Catalog connection (SigV4, service `s3tables`) and table health measures |
| [scripts/make_mess.py](scripts/make_mess.py) | Streams micro-batches, one commit each |
| [scripts/inspect_table.py](scripts/inspect_table.py) | Health report, save and compare, watch mode |
| [lab/Dockerfile](lab/Dockerfile) | Python 3.12 with PyIceberg, DuckDB and JupyterLab |

## Query from your machine with DuckDB

```sql
INSTALL iceberg; LOAD iceberg;
CREATE SECRET aistor_s3 (TYPE S3, KEY_ID 'ingest-svc', SECRET 'ingest-svc-secret',
  ENDPOINT 'localhost:9000', URL_STYLE 'path', USE_SSL 0, REGION 'us-east-1');
ATTACH 'lakehouse' AS aistor (TYPE ICEBERG, ENDPOINT 'http://localhost:9000/_iceberg',
  AUTHORIZATION_TYPE 'sigv4', SIGV4_SERVICE 's3tables', SIGV4_REGION 'us-east-1',
  SECRET aistor_s3);
SELECT site, count(*), avg(temperature_c) FROM aistor.iot.sensor_readings GROUP BY site;
```

## Moving to production

- **Remove all three underscore settings** and keep the server intervals at
  their defaults, or set them to what your operations team agrees to.
- **Use retention values that fit your recovery needs.** For example,
  `--max-age-hours 168 --min-snapshots 5` keeps a week of time travel, and
  `--unreferenced-days 7 --noncurrent-days 7` gives two weeks to recover from an
  accidental snapshot expiration before files are deleted permanently.
- **Use per-table overrides** for tables that need something different, for
  example disabling compaction on a table whose sort order compaction can't
  reproduce.
- Unreferenced file removal runs on the primary site only when site
  replication is configured.
- Serve AIStor over TLS, replace every demo secret, and scrape metrics with a
  bearer token.

## Notes from testing

- The [table maintenance docs](https://docs.min.io/aistor/administration/aistor-tables/table-maintenance/)
  say compaction can't be configured at the warehouse level. In the tested
  release, `mc table warehouse maintenance set --type icebergCompaction` works
  and takes `--target-file-size-mb` and `--interval`. This demo uses it.
- With no maintenance policy set, nothing runs. `mc table maintenance status`
  returns `MaintenanceConfigurationNotFound` until a policy exists.
