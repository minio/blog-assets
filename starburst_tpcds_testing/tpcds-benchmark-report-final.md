# TPC-DS Benchmark Report
## Starburst Enterprise on MinIO AIStor

**Date:** March 26, 2026
**Prepared by:** sendhil mandala
**Scale Factors:** sf1000, sf3000, sf5000, sf10000

---

## Table of Contents

1. Executive Summary
2. AIStor Native Iceberg Tables
3. Benchmark Overview
4. Test Environment
5. Test Configuration
6. Methodology
7. Results by Scale Factor
   - 7.1 sf1000 (~1 TB)
   - 7.2 sf3000 (~3 TB)
   - 7.3 sf5000 (~5 TB)
   - 7.4 sf10000 (~10 TB)
8. Combined Query Results
9. Cross-Scale Summary

---

## 1. Executive Summary

All 99 TPC-DS queries were executed across four scale factors against Apache Iceberg tables stored in MinIO AIStor. Two configurations were benchmarked:

- **Pre-Partitioned:** Non-partitioned tables loaded via CTAS; 3-run warm average
- **Partitioned + Compacted:** Tables partitioned by `ss_sold_date_sk` and compacted with Iceberg `OPTIMIZE`; single cold run

### Performance Overview

| Scale Factor | Data Size | Pre-Partitioned Total | Pre-Partitioned Geo Mean | Partitioned Total | Partitioned Geo Mean | Improvement |
|---|---|---|---|---|---|---|
| sf1000 | ~1 TB | 713.5 s (11.9 min) | 6.0 s | 538.8 s (9.0 min) | 4.0 s | 24% faster |
| sf3000 | ~3 TB | 1,225.9 s (20.4 min) | 8.4 s | 789.1 s (13.2 min) | 4.1 s | 36% faster |
| sf5000 | ~5 TB | 1,468.8 s (24.5 min) | 9.3 s | 833.3 s (13.9 min) | 3.9 s | 43% faster |
| sf10000 | ~10 TB | 2,787.4 s (46.5 min) | 14.3 s | 1,944.6 s (32.4 min) | 7.7 s | 30% faster |

### Scaling Efficiency (Partitioned + Compacted)

| Comparison | Data Growth | Time Growth |
|---|---|---|
| sf1000 → sf3000 | 3× | 1.46× |
| sf1000 → sf5000 | 5× | 1.55× |
| sf1000 → sf10000 | 10× | 3.61× |
| sf3000 → sf5000 | 1.67× | 1.06× |
| sf5000 → sf10000 | 2× | 2.33× |

Even with 10× more data, query time increased by only 3.6× — demonstrating that MinIO AIStor and Starburst maintain strong performance at scale. Iceberg partition pruning allows the query engine to skip irrelevant date partitions entirely, so larger datasets do not proportionally increase scan time for date-filtered workloads.

---

## 2. AIStor Native Iceberg Tables

A core architectural differentiator in this benchmark is that the TPC-DS dataset lives in **Apache Iceberg tables managed entirely within MinIO AIStor** — not in an external catalog service.

### What This Means

Traditional lakehouse deployments require a separate catalog stack alongside object storage: a metadata service (Hive Metastore or Nessie), a backing relational database (PostgreSQL), and a REST catalog API layer — typically three additional services that must be deployed, operated, and kept in sync with the storage layer. MinIO AIStor eliminates this entirely by embedding a native Apache Iceberg REST catalog directly into the object store.

In this benchmark, Starburst connects to AIStor's built-in Iceberg REST endpoint and reads all table metadata — schemas, snapshots, manifests, partition statistics — directly from the same system that holds the data. There is no external catalog, no separate metastore, and no additional service in the query path.

### Why It Matters

**Architectural simplification.** A conventional lakehouse stack requires four layers (query engine → REST catalog service → PostgreSQL metastore → object storage) with three distinct APIs and multiple failure points. AIStor Tables collapses this to two layers: query engine → AIStor. The catalog and the data are the same system.

**Engine interoperability.** Because AIStor exposes a standard Iceberg REST catalog API, any engine that speaks Iceberg REST — Starburst, Trino, Apache Spark, Dremio — can read and write the same tables without coordination. There is no proprietary connector, no vendor lock-in at the catalog layer, and no format translation overhead.

**On-premises with full Iceberg semantics.** AIStor is the only on-premises object storage platform with a native, built-in Iceberg REST catalog. Cloud providers (AWS S3 Tables, Google, Azure) offer similar native catalog capabilities but only within their own environments. AIStor brings the same capability to on-premises and hybrid deployments, on commodity x86 hardware, at exabyte scale.

**Operational integrity.** ACID transactions, schema evolution, time travel, and snapshot isolation are guaranteed at the storage layer without requiring a separate transactional database. Catalog metadata updates are atomic and strictly consistent with the underlying data.

The performance results in this report — all 99 TPC-DS queries executed across four scale factors against Iceberg tables stored in AIStor — demonstrate that native Iceberg on high-performance object storage is not only architecturally simpler but also competitive with the fastest published benchmarks on equivalent workloads.

---

## 3. Benchmark Overview

### What is TPC-DS?

TPC-DS (Transaction Processing Performance Council — Decision Support) is an industry-standard benchmark designed to model a realistic decision support workload. It consists of 99 complex SQL queries that exercise a broad range of SQL features including multi-table joins, aggregations, window functions, subqueries, and UNION operations.

### Why TPC-DS?

- Widely used for evaluating data warehouse and analytics query engines
- Covers realistic workloads: reporting, ad hoc queries, iterative OLAP, data mining
- Multiple scale factors allow performance scaling analysis
- Results are directly comparable across different systems and configurations

### Datasets

| Scale Factor | Approx. Size | Largest Table (store_sales) |
|---|---|---|
| sf1000 | ~1 TB | ~2.88 billion rows |
| sf3000 | ~3 TB | ~8.64 billion rows |
| sf5000 | ~5 TB | ~14.4 billion rows |
| sf10000 | ~10 TB | ~28.8 billion rows |

All datasets are stored as Apache Iceberg tables (Parquet format) in the MinIO AIStor `tpcds-iceberg` warehouse.

---

## 4. Test Environment

### 4.1 MinIO AIStor — Storage Layer

| Property | Value |
|---|---|
| Nodes | 8 |
| Software | MinIO AIStor |
| Erasure Coding | EC:4 (12 data + 4 parity per set) |
| Total Usable Capacity | 838 TiB |
| Drives Online | 160 / 160 |
| Iceberg REST Endpoint | Native AIStor Iceberg REST catalog |

**Per-Node (all 8 nodes identical)**

| Component | Specification |
|---|---|
| Server Model | Supermicro SYS-212H-TN |
| CPU | Intel Xeon 6761P — 64 cores / 128 threads @ 2.5 GHz |
| Memory | 256 GB DDR5 @ 6400 MT/s |
| Data Drives | 20 × 7.68 TB Solidigm NVMe SSD |
| Network | 2 × Mellanox ConnectX-7 400 GbE |

### 4.2 Starburst Enterprise — Query Engine

| Property | Value |
|---|---|
| Nodes | 9 total (1 coordinator + 8 workers) |
| Software | Starburst Enterprise |
| Version | 479-e.4 |

**Per-Node (all 9 nodes identical)**

| Component | Specification |
|---|---|
| Server Model | Supermicro X14SBT-GAP |
| CPU | Intel Xeon 6960P — 72 cores / 144 threads @ 2.7 GHz |
| Memory | 512 GB |
| Data Drive | 1 × 7 TB Solidigm NVMe (spill / temp) |
| Network | 2 × Mellanox ConnectX-7 400 GbE |

### 4.3 Network Topology

All Starburst ↔ MinIO data traffic flows over a dedicated 400 GbE fabric. Management access uses separate 10 GbE interfaces.

```
8 × Starburst Workers      ──┐
1 × Starburst Coordinator  ──┤  400 GbE
8 × MinIO AIStor Nodes     ──┘
```

---

## 5. Test Configuration

### 5.1 Starburst Memory Settings

| Parameter | Value |
|---|---|
| JVM Heap (-Xmx) | 400 GB per node |
| query.max-memory-per-node | 280 GB |
| query.max-memory | 2,000 GB (cluster-wide) |

### 5.2 Iceberg Catalog (aistor)

| Parameter | Value |
|---|---|
| Connector | iceberg |
| Catalog Type | REST |
| REST Endpoint | AIStor native Iceberg REST catalog |
| Warehouse | tpcds-iceberg |
| Auth | SigV4 (signing-name: s3tables) |
| File I/O | Native S3 (fs.native-s3.enabled=true) |
| File Format | Parquet |

---

## 6. Methodology

### 6.1 Pre-Partitioned Configuration

**Data Layout:** All 26 TPC-DS tables loaded via `CREATE TABLE ... AS SELECT` from the built-in `tpcds` connector using default settings — no partitioning, no sort ordering, no compaction. Each table is a single-level flat Parquet layout as written by the CTAS.

**Execution:** Queries submitted sequentially via the **Starburst REST API**. Each scale factor run 3 times — one cold run followed by two warm runs. Results reported as the average across all 3 runs per query.

Cold start procedure before each scale factor:
1. `sync; echo 3 > /proc/sys/vm/drop_caches` on all 17 nodes
2. `systemctl restart starburst` on all 9 nodes
3. Wait for all 8 workers to reach ACTIVE state
4. 30-second stabilization hold before first query

### 6.2 Partitioned + Compacted Configuration

**Data Layout:** All 26 TPC-DS tables loaded via CTAS with fact tables (`store_sales`, `web_sales`, `catalog_sales`, and their returns tables) partitioned by `ss_sold_date_sk`. After loading, all tables compacted with:

```sql
ALTER TABLE <table> EXECUTE optimize(file_size_threshold => '128MB')
```

Compaction merges small Parquet files into larger column-aligned files and rewrites Iceberg partition statistics — enabling aggressive partition pruning and predicate pushdown during query execution.

**Execution:** All 99 queries submitted sequentially via **Apache JMeter** driving the **Starburst REST API**. Each scale factor run as a **single cold pass** — OS page cache flushed and Starburst restarted before the run. Results are the single cold run time per query.

### 6.3 Metrics

- Wall clock time per query (seconds, rounded to 0.1 s)
- Geometric mean across all 99 queries
- Total elapsed time = sum of all 99 query times

---

## 7. Results by Scale Factor

### 7.1 sf1000 (~1 TB)

| Metric | Pre-Partitioned | Partitioned + Compacted |
|---|---|---|
| Queries Completed | 99 / 99 | 99 / 99 |
| Total Time | 713.5 s (11.9 min) | 538.8 s (9.0 min) |
| Geometric Mean | 6.0 s | 4.0 s |
| Fastest Query | Q2 — 4.0 s | Q92 — 1.4 s |
| Slowest Query | Q68 — 46.5 s | Q22 — 33.5 s |

**Time Distribution**

| Bucket | Pre-Partitioned | Partitioned + Compacted |
|---|---|---|
| < 10 s | 86 | 88 |
| 10 – 30 s | 11 | 8 |
| 30 – 60 s | 2 | 3 |
| > 60 s | 0 | 0 |

---

### 7.2 sf3000 (~3 TB)

| Metric | Pre-Partitioned | Partitioned + Compacted |
|---|---|---|
| Queries Completed | 99 / 99 | 99 / 99 |
| Total Time | 1,225.9 s (20.4 min) | 789.1 s (13.2 min) |
| Geometric Mean | 8.4 s | 4.1 s |
| Fastest Query | Q2 — 4.0 s | Q41 — 0.7 s |
| Slowest Query | Q35 — 137.6 s | Q67 — 70.1 s |

**Time Distribution**

| Bucket | Pre-Partitioned | Partitioned + Compacted |
|---|---|---|
| < 10 s | 62 | 78 |
| 10 – 30 s | 31 | 14 |
| 30 – 60 s | 4 | 5 |
| > 60 s | 2 | 2 |

---

### 7.3 sf5000 (~5 TB)

| Metric | Pre-Partitioned | Partitioned + Compacted |
|---|---|---|
| Queries Completed | 99 / 99 | 99 / 99 |
| Total Time | 1,468.8 s (24.5 min) | 833.3 s (13.9 min) |
| Geometric Mean | 9.3 s | 3.9 s |
| Fastest Query | Q2 — 4.0 s | Q41 — 0.2 s |
| Slowest Query | Q35 — 178.8 s | Q04 — 56.0 s |

**Time Distribution**

| Bucket | Pre-Partitioned | Partitioned + Compacted |
|---|---|---|
| < 10 s | 57 | 75 |
| 10 – 30 s | 34 | 18 |
| 30 – 60 s | 5 | 6 |
| > 60 s | 3 | 0 |

---

### 7.4 sf10000 (~10 TB)

| Metric | Pre-Partitioned | Partitioned + Compacted |
|---|---|---|
| Queries Completed | 99 / 99 | 99 / 99 |
| Total Time | 2,787.4 s (46.5 min) | 1,944.6 s (32.4 min) |
| Geometric Mean | 14.3 s | 7.7 s |
| Fastest Query | Q2 — 4.0 s | Q41 — 0.8 s |
| Slowest Query | Q35 — 398.3 s | Q67 — 221.2 s |

**Time Distribution**

| Bucket | Pre-Partitioned | Partitioned + Compacted |
|---|---|---|
| < 10 s | 37 | 66 |
| 10 – 30 s | 37 | 13 |
| 30 – 60 s | 16 | 12 |
| > 60 s | 9 | 8 |

---

## 8. Combined Query Results

All times in seconds. **Pre** = pre-partitioned warm average (3 runs). **Part** = partitioned + compacted cold run.

| Query | SF1000 Pre | SF1000 Part | SF1000 CTE+Cache | SF3000 Pre | SF3000 Part | SF3000 CTE+Cache | SF5000 Pre | SF5000 Part | SF5000 CTE+Cache | SF10000 Pre | SF10000 Part | SF10000 CTE+Cache |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q01 | 6.9 | 3.7 | 9.3 | 9.4 | 3.6 | 3.6 | 10.0 | 3.0 | 4.2 | 13.1 | 4.6 | 7.0 |
| Q02 | 4.0 | 3.6 | 4.6 | 4.0 | 2.1 | 4.5 | 4.0 | 1.9 | 7.5 | 4.0 | 2.9 | 13.2 |
| Q03 | 6.8 | 1.5 | 1.6 | 13.6 | 0.7 | 0.9 | 19.1 | 1.0 | 1.3 | 31.1 | 1.9 | 1.4 |
| Q04 | 4.1 | 18.4 | 24.6 | 4.3 | 42.4 | 42.7 | 5.0 | 56.0 | 58.8 | 8.0 | 127.7 | 131.7 |
| Q05 | 6.4 | 4.0 | 3.8 | 8.1 | 4.1 | 3.9 | 4.0 | 4.3 | 3.7 | 7.4 | 7.7 | 10.6 |
| Q06 | 8.4 | 2.7 | 4.5 | 14.2 | 2.3 | 1.7 | 18.2 | 1.9 | 1.7 | 31.0 | 3.5 | 4.0 |
| Q07 | 4.6 | 3.7 | 2.0 | 5.2 | 2.8 | 2.7 | 6.2 | 2.4 | 2.4 | 10.7 | 7.8 | 5.2 |
| Q08 | 6.1 | 2.6 | 2.3 | 9.1 | 1.7 | 2.4 | 10.4 | 1.8 | 2.1 | 15.5 | 3.3 | 3.5 |
| Q09 | 8.7 | 11.7 | 5.6 | 12.9 | 20.1 | 11.3 | 16.1 | 30.2 | 18.2 | 24.4 | 54.0 | 25.3 |
| Q10 | 16.9 | 2.6 | 3.4 | 37.8 | 1.6 | 1.9 | 59.5 | 2.0 | 1.5 | 116.2 | 2.8 | 2.7 |
| Q11 | 4.0 | 9.9 | 11.5 | 4.7 | 21.8 | 20.7 | 5.6 | 28.1 | 27.3 | 8.0 | 57.6 | 63.8 |
| Q12 | 6.2 | 2.7 | 1.5 | 16.3 | 1.3 | 1.2 | 4.7 | 0.7 | 0.6 | 50.2 | 2.1 | 1.8 |
| Q13 | 4.0 | 3.0 | 2.0 | 4.0 | 3.0 | 3.1 | 4.0 | 4.8 | 4.4 | 4.0 | 7.2 | 6.0 |
| Q14 | 4.0 | 13.4 | 10.3 | 4.0 | 19.4 | 12.1 | 4.0 | 21.6 | 12.4 | 4.0 | 44.2 | 23.2 |
| Q15 | 4.0 | 2.0 | 2.1 | 6.0 | 1.1 | 1.7 | 6.3 | 1.5 | 1.6 | 10.8 | 2.5 | 2.0 |
| Q16 | 4.0 | 4.3 | 4.9 | 4.0 | 8.9 | 9.3 | 4.3 | 18.9 | 14.8 | 4.0 | 33.3 | 34.5 |
| Q17 | 4.1 | 4.1 | 3.6 | 6.3 | 4.7 | 3.8 | 7.1 | 5.8 | 5.1 | 13.4 | 9.0 | 9.6 |
| Q18 | 4.1 | 4.8 | 4.2 | 4.9 | 4.3 | 4.3 | 5.3 | 4.0 | 5.4 | 7.7 | 7.7 | 6.3 |
| Q19 | 5.9 | 2.4 | 1.8 | 9.1 | 1.5 | 1.4 | 9.6 | 2.0 | 1.4 | 18.3 | 1.6 | 2.2 |
| Q20 | 10.0 | 2.6 | 1.2 | 16.2 | 1.7 | 1.4 | 5.4 | 0.9 | 1.0 | 18.4 | 2.4 | 2.1 |
| Q21 | 6.1 | 2.4 | 1.6 | 9.0 | 7.0 | 1.1 | 14.8 | 0.4 | 0.4 | 16.2 | 1.9 | 1.5 |
| Q22 | 4.0 | 33.5 | 35.5 | 4.0 | 34.3 | 34.3 | 4.0 | 2.9 | 3.1 | 4.0 | 34.2 | 33.7 |
| Q23 | 5.7 | 31.2 | 28.9 | 8.7 | 63.2 | 64.8 | 12.9 | 42.6 | 30.5 | 15.6 | 198.6 | 171.3 |
| Q24 | 4.0 | 9.2 | 7.5 | 4.9 | 21.8 | 14.9 | 6.8 | 29.2 | 23.1 | 9.5 | 65.7 | 59.0 |
| Q25 | 5.6 | 3.0 | 2.8 | 9.9 | 2.5 | 2.3 | 13.5 | 3.2 | 3.4 | 20.9 | 8.6 | 7.3 |
| Q26 | 5.6 | 2.7 | 1.7 | 6.5 | 1.9 | 1.8 | 6.6 | 1.7 | 2.1 | 11.5 | 4.5 | 3.4 |
| Q27 | 4.0 | 3.5 | 1.2 | 4.0 | 1.9 | 2.3 | 4.0 | 2.2 | 2.0 | 4.0 | 4.6 | 3.4 |
| Q28 | 4.6 | 9.7 | 6.7 | 6.5 | 19.4 | 15.8 | 7.6 | 27.0 | 18.3 | 11.5 | 46.1 | 26.9 |
| Q29 | 4.0 | 5.1 | 4.0 | 4.0 | 8.0 | 7.8 | 4.0 | 17.0 | 17.3 | 4.0 | 22.6 | 27.6 |
| Q30 | 6.6 | 3.7 | 3.1 | 8.8 | 3.9 | 2.7 | 11.6 | 2.4 | 3.2 | 19.4 | 3.1 | 3.2 |
| Q31 | 4.3 | 3.8 | 3.8 | 7.7 | 4.5 | 4.0 | 11.4 | 3.5 | 4.1 | 18.7 | 6.3 | 6.8 |
| Q32 | 4.0 | 1.4 | 1.0 | 4.0 | 0.8 | 0.8 | 4.0 | 0.7 | 0.6 | 4.0 | 1.2 | 1.3 |
| Q33 | 4.0 | 2.6 | 2.1 | 4.7 | 2.0 | 1.6 | 6.0 | 1.6 | 21.3 | 8.7 | 2.4 | 2.0 |
| Q34 | 7.5 | 2.1 | 1.1 | 13.9 | 1.7 | 1.2 | 22.6 | 1.7 | 1.4 | 30.8 | 1.7 | 1.9 |
| Q35 | 46.0 | 3.5 | 2.4 | 137.6 | 2.6 | 1.9 | 178.8 | 2.9 | 2.1 | 398.3 | 3.9 | 4.1 |
| Q36 | 11.5 | 2.0 | 1.3 | 15.5 | 2.2 | 1.9 | 25.1 | 2.2 | 2.6 | 42.6 | 4.5 | 5.8 |
| Q37 | 4.4 | 3.4 | 2.4 | 5.0 | 5.6 | 3.7 | 5.6 | 7.1 | 6.9 | 6.9 | 16.5 | 16.2 |
| Q38 | 7.8 | 5.4 | 5.2 | 14.8 | 7.2 | 7.0 | 20.0 | 9.0 | 7.7 | 40.1 | 19.2 | 19.7 |
| Q39 | 4.0 | 4.0 | 3.2 | 4.0 | 2.7 | 2.6 | 4.0 | 0.9 | 0.9 | 4.0 | 6.5 | 2.7 |
| Q40 | 4.2 | 3.1 | 2.3 | 4.1 | 2.9 | 1.3 | 5.0 | 2.8 | 1.5 | 6.0 | 6.1 | 2.2 |
| Q41 | 8.7 | 1.4 | 0.7 | 10.4 | 0.7 | 0.8 | 15.8 | 0.2 | 0.2 | 23.9 | 0.8 | 0.8 |
| Q42 | 14.1 | 1.6 | 0.5 | 28.8 | 2.5 | 0.7 | 36.5 | 0.5 | 0.4 | 61.7 | 0.9 | 0.8 |
| Q43 | 6.8 | 1.9 | 1.7 | 12.1 | 2.1 | 1.5 | 17.4 | 1.6 | 3.9 | 31.0 | 3.2 | 2.2 |
| Q44 | 4.0 | 7.6 | 4.0 | 5.3 | 12.6 | 7.3 | 4.9 | 15.4 | 12.2 | 7.2 | 30.5 | 16.2 |
| Q45 | 4.0 | 2.1 | 2.1 | 6.3 | 1.4 | 1.4 | 6.3 | 1.5 | 2.0 | 11.0 | 3.8 | 2.2 |
| Q46 | 7.3 | 2.7 | 2.3 | 13.9 | 2.2 | 2.0 | 16.1 | 2.3 | 2.7 | 32.8 | 5.9 | 4.6 |
| Q47 | 5.4 | 8.2 | 6.2 | 7.8 | 10.9 | 8.1 | 8.0 | 11.6 | 8.4 | 12.0 | 28.7 | 13.2 |
| Q48 | 6.7 | 3.6 | 2.4 | 12.0 | 3.7 | 2.4 | 16.2 | 3.4 | 2.1 | 29.7 | 4.9 | 4.9 |
| Q49 | 9.6 | 4.3 | 3.0 | 17.7 | 5.9 | 3.8 | 26.6 | 4.3 | 5.0 | 50.6 | 7.8 | 4.8 |
| Q50 | 9.0 | 6.0 | 2.7 | 16.4 | 21.3 | 20.1 | 21.0 | 25.6 | 17.7 | 28.1 | 66.9 | 59.8 |
| Q51 | 16.4 | 3.9 | 7.9 | 32.8 | 3.9 | 4.7 | 43.5 | 2.3 | 2.5 | 72.4 | 10.0 | 10.9 |
| Q52 | 4.0 | 1.9 | 0.7 | 4.0 | 1.0 | 0.6 | 4.0 | 0.7 | 0.5 | 4.0 | 1.2 | 0.9 |
| Q53 | 8.7 | 2.5 | 2.0 | 18.8 | 1.3 | 1.4 | 26.8 | 1.3 | 1.5 | 46.3 | 3.1 | 3.2 |
| Q54 | 7.3 | 3.6 | 2.9 | 12.4 | 2.3 | 2.4 | 14.7 | 2.3 | 2.6 | 23.7 | 6.5 | 4.8 |
| Q55 | 15.6 | 2.6 | 0.6 | 18.0 | 0.7 | 0.7 | 4.5 | 0.8 | 0.4 | 20.8 | 1.0 | 0.8 |
| Q56 | 4.0 | 3.4 | 1.9 | 4.0 | 1.5 | 1.6 | 4.0 | 1.0 | 21.5 | 4.0 | 2.9 | 2.2 |
| Q57 | 4.1 | 6.1 | 4.0 | 5.3 | 7.0 | 3.8 | 6.2 | 7.7 | 5.9 | 8.2 | 17.2 | 7.0 |
| Q58 | 5.0 | 3.6 | 3.5 | 7.2 | 3.5 | 3.6 | 8.0 | 3.1 | 2.6 | 14.2 | 5.7 | 3.8 |
| Q59 | 4.1 | 4.3 | 4.5 | 6.5 | 2.9 | 4.7 | 7.0 | 4.0 | 6.1 | 11.5 | 6.2 | 11.7 |
| Q60 | 5.3 | 3.2 | 2.2 | 24.4 | 2.2 | 2.5 | 35.4 | 1.5 | 21.9 | 65.7 | 3.1 | 2.3 |
| Q61 | 4.0 | 2.9 | 2.1 | 5.6 | 2.2 | 2.3 | 6.5 | 2.1 | 2.1 | 11.6 | 3.1 | 4.1 |
| Q62 | 4.0 | 3.0 | 1.9 | 4.0 | 2.9 | 2.5 | 4.0 | 3.5 | 2.5 | 4.0 | 4.8 | 4.9 |
| Q63 | 4.1 | 2.2 | 1.4 | 6.0 | 1.3 | 1.7 | 8.0 | 1.6 | 2.0 | 10.5 | 3.3 | 2.6 |
| Q64 | 4.0 | 15.8 | 6.2 | 4.0 | 16.4 | 8.4 | 4.0 | 18.1 | 11.0 | 4.0 | 32.4 | 19.4 |
| Q65 | 4.0 | 6.7 | 4.7 | 4.0 | 9.8 | 10.2 | 4.0 | 9.0 | 7.6 | 4.0 | 28.6 | 24.9 |
| Q66 | 8.7 | 2.3 | 2.5 | 12.3 | 2.3 | 3.8 | 17.7 | 3.5 | 3.6 | 30.5 | 5.4 | 6.0 |
| Q67 | 6.2 | 33.1 | 32.6 | 12.8 | 70.1 | 86.9 | 18.0 | 55.9 | 66.3 | 30.0 | 221.2 | 231.0 |
| Q68 | 46.5 | 2.9 | 1.2 | 108.6 | 2.1 | 2.6 | 119.1 | 2.4 | 2.1 | 300.2 | 2.9 | 1.6 |
| Q69 | 15.7 | 2.4 | 1.9 | 31.6 | 1.5 | 1.7 | 38.5 | 2.1 | 2.2 | 86.3 | 3.1 | 2.0 |
| Q70 | 13.3 | 3.1 | 3.5 | 28.7 | 5.8 | 7.4 | 23.6 | 10.3 | 14.1 | 62.8 | 14.6 | 13.3 |
| Q71 | 9.1 | 2.3 | 1.6 | 16.2 | 1.1 | 1.1 | 17.4 | 1.1 | 1.3 | 43.4 | 1.7 | 1.4 |
| Q72 | 4.0 | 20.0 | 18.1 | 4.0 | 34.0 | 37.1 | 4.0 | 10.6 | 10.3 | 4.0 | 99.9 | 102.7 |
| Q73 | 4.0 | 1.8 | 0.8 | 4.0 | 0.9 | 1.2 | 4.0 | 1.1 | 1.1 | 4.0 | 2.1 | 1.6 |
| Q74 | 4.0 | 5.2 | 6.4 | 4.0 | 15.8 | 12.9 | 4.0 | 15.2 | 16.1 | 4.0 | 30.7 | 40.2 |
| Q75 | 4.8 | 10.8 | 8.1 | 4.4 | 19.4 | 15.1 | 5.2 | 23.7 | 18.5 | 6.0 | 44.0 | 39.0 |
| Q76 | 10.8 | 5.4 | 3.9 | 21.6 | 9.6 | 7.9 | 26.4 | 9.8 | 5.4 | 51.6 | 20.0 | 9.8 |
| Q77 | 7.3 | 2.6 | 1.3 | 10.9 | 1.4 | 2.1 | 14.8 | 1.8 | 1.5 | 31.0 | 1.9 | 2.0 |
| Q78 | 7.5 | 13.5 | 12.7 | 11.5 | 30.9 | 25.2 | 14.3 | 54.0 | 50.0 | 25.3 | 88.7 | 93.8 |
| Q79 | 4.0 | 2.9 | 2.1 | 4.0 | 2.3 | 1.9 | 4.0 | 3.4 | 1.8 | 4.0 | 4.5 | 3.0 |
| Q80 | 4.0 | 5.3 | 3.7 | 4.5 | 6.5 | 2.6 | 5.6 | 6.7 | 3.7 | 8.6 | 8.0 | 4.9 |
| Q81 | 5.1 | 4.4 | 2.9 | 6.0 | 2.6 | 2.8 | 8.8 | 3.6 | 2.9 | 14.2 | 4.7 | 3.8 |
| Q82 | 4.1 | 4.3 | 3.3 | 5.3 | 10.5 | 7.5 | 7.0 | 13.2 | 13.2 | 14.3 | 24.0 | 28.4 |
| Q83 | 4.0 | 4.1 | 2.7 | 4.0 | 2.7 | 2.6 | 4.0 | 1.9 | 2.0 | 4.0 | 2.6 | 2.9 |
| Q84 | 4.7 | 3.2 | 1.7 | 8.1 | 3.8 | 2.6 | 10.8 | 4.9 | 3.7 | 17.1 | 6.1 | 5.7 |
| Q85 | 4.0 | 2.9 | 2.1 | 4.0 | 3.2 | 2.5 | 4.0 | 4.2 | 3.0 | 4.0 | 5.7 | 3.0 |
| Q86 | 4.0 | 2.0 | 1.3 | 6.0 | 3.2 | 3.2 | 5.7 | 2.9 | 2.6 | 10.0 | 2.8 | 3.4 |
| Q87 | 13.3 | 3.2 | 4.5 | 15.3 | 5.6 | 7.7 | 4.5 | 6.9 | 7.1 | 15.8 | 18.9 | 19.4 |
| Q88 | 4.0 | 8.2 | 9.4 | 4.0 | 15.1 | 6.4 | 4.0 | 19.3 | 8.3 | 4.0 | 33.1 | 10.8 |
| Q89 | 7.0 | 2.7 | 1.5 | 11.7 | 1.6 | 1.5 | 15.8 | 1.8 | 1.3 | 24.9 | 4.3 | 3.2 |
| Q90 | 4.0 | 3.2 | 2.5 | 4.0 | 2.7 | 1.2 | 4.0 | 3.1 | 1.7 | 4.0 | 5.8 | 2.0 |
| Q91 | 4.0 | 1.9 | 1.4 | 4.0 | 1.3 | 1.4 | 4.0 | 1.0 | 0.9 | 4.0 | 1.4 | 1.3 |
| Q92 | 6.8 | 1.4 | 0.8 | 12.0 | 0.7 | 0.9 | 14.8 | 0.5 | 0.6 | 32.4 | 1.1 | 0.8 |
| Q93 | 24.9 | 12.9 | 3.2 | 54.1 | 31.0 | 24.3 | 81.5 | 41.3 | 42.5 | 184.4 | 70.0 | 69.7 |
| Q94 | 4.3 | 3.6 | 2.5 | 6.9 | 5.5 | 3.4 | 8.7 | 8.9 | 6.5 | 11.0 | 12.1 | 13.2 |
| Q95 | 4.8 | 7.5 | 6.6 | 8.0 | 11.7 | 10.4 | 12.3 | 16.0 | 17.6 | 21.8 | 34.4 | 32.1 |
| Q96 | 4.6 | 3.6 | 2.3 | 5.1 | 3.6 | 2.3 | 4.3 | 3.9 | 2.3 | 6.5 | 6.6 | 3.6 |
| Q97 | 4.0 | 4.5 | 5.4 | 4.0 | 7.3 | 7.8 | 4.0 | 13.7 | 13.9 | 4.0 | 28.4 | 29.4 |
| Q98 | 8.0 | 3.2 | 1.2 | 13.1 | 2.0 | 2.2 | 16.2 | 1.2 | 1.1 | 27.7 | 2.5 | 2.9 |
| Q99 | 6.7 | 4.6 | 1.8 | 11.8 | 3.9 | 3.8 | 11.3 | 4.1 | 3.3 | 21.8 | 7.0 | 5.0 |
| **Total** | **713.5** | **538.8** | **452.2** | **1,225.9** | **789.1** | **711.6** | **1,468.8** | **833.3** | **805.5** | **2,787.4** | **1,944.6** | **1,761.8** |
| **Geo Mean** | **6.0** | **4.0** | **3.0** | **8.4** | **4.1** | **3.6** | **9.3** | **3.9** | **3.9** | **14.3** | **7.7** | **6.5** |

---

## 9. Cross-Scale Summary

### 9.1 Scaling Efficiency (Partitioned + Compacted)

| Comparison | Data Growth | Time Growth |
|---|---|---|
| sf1000 → sf3000 | 3× | 1.46× |
| sf1000 → sf5000 | 5× | 1.55× |
| sf1000 → sf10000 | 10× | 3.61× |
| sf3000 → sf5000 | 1.67× | 1.06× |
| sf5000 → sf10000 | 2× | 2.33× |

### 9.2 Queries with Highest Execution Time

The following queries require the most compute time at scale. Unlike most TPC-DS queries, these involve broad multi-date-range aggregations, correlated subqueries across large fact tables, or self-joins — patterns that limit how much Iceberg partition pruning can reduce I/O. Their execution time grows more steeply with data volume because the query logic requires scanning a larger fraction of the dataset regardless of partitioning.

| Query | SF1000 (s) | SF3000 (s) | SF5000 (s) | SF10000 (s) | Characteristic |
|---|---|---|---|---|---|
| Q67 | 33.1 | 70.1 | 55.9 | 221.2 | Multi-year rollup across all store sales | | 66.3 | 231.0 |
| Q23 | 31.2 | 63.2 | 42.6 | 198.6 | Two-stage correlated subquery with cross-channel join | | 30.5 | 171.3 |
| Q04 | 18.4 | 42.4 | 56.0 | 127.7 | Year-over-year customer spending comparison | | 58.8 | 131.7 |
| Q72 | 20.0 | 34.0 | 10.6 | 99.9 | Inventory and warehouse join with date range scan | | 10.3 | 102.7 |
| Q78 | 13.5 | 30.9 | 54.0 | 88.7 | Cross-channel sales comparison requiring three full scans | | 50.0 | 93.8 |
| Q93 | 12.9 | 31.0 | 41.3 | 70.0 | Store returns with correlated customer filter | | 42.5 | 69.7 |
| Q50 | 6.0 | 21.3 | 25.6 | 66.9 | Store returns grouped across multiple date windows | | 17.7 | 59.8 |
| Q24 | 9.2 | 21.8 | 29.2 | 65.7 | Item profit calculation requiring two full store sales passes | | 23.1 | 59.0 |
| Q11 | 9.9 | 21.8 | 28.1 | 57.6 | Web and store customer year-over-year aggregation | | 27.3 | 63.8 |
| Q09 | 11.7 | 20.1 | 30.2 | 54.0 | Nine-way conditional aggregation across store sales | | 18.2 | 25.3 |

### 9.3 Biggest Winners from Partitioning

Queries where partitioning + compaction delivered the largest speedup (shown at sf10000):

| Query | Pre-Partitioned (s) | Partitioned (s) | Speedup |
|---|---|---|---|
| Q68 | 300.2 | 2.9 | 103× | | 2.1 | 1.6 |
| Q10 | 116.2 | 2.8 | 42× | | 1.5 | 2.7 |
| Q69 | 86.3 | 3.1 | 28× | | 2.2 | 2.0 |
| Q35 | 398.3 | 3.9 | 102× | | 2.1 | 4.1 |
| Q51 | 72.4 | 10.0 | 7× | | 2.5 | 10.9 |
| Q42 | 61.7 | 0.9 | 69× | | 0.4 | 0.8 |
| Q53 | 46.3 | 3.1 | 15× | | 1.5 | 3.2 |
| Q41 | 23.9 | 0.8 | 30× | | 0.2 | 0.8 |
| Q60 | 65.7 | 3.1 | 21× | | 21.9 | 2.3 |
| Q92 | 32.4 | 1.1 | 29× | | 0.6 | 0.8 |
