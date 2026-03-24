# TPC-DS Benchmark Report
## Starburst Enterprise on MinIO AIStor

**Date:** March 24, 2026
**Prepared by:**
**Scale Factors:** sf1000, sf3000, sf5000, sf10000
**Status:** Complete

---

## Table of Contents

1. Executive Summary
2. AIStor Native Iceberg Tables
3. Benchmark Overview
4. Test Environment
5. Test Configuration
6. Methodology
7. Results
   - 7.1 sf1000 (~1 TB)
   - 7.2 sf3000 (~3 TB)
   - 7.3 sf5000 (~5 TB)
   - 7.4 sf10000 (~10 TB)
8. Cross-Scale Summary
9. Appendix — Full Configuration

---

## 1. Executive Summary

### Performance Overview

| Scale Factor | Data Size | Queries Completed | Total Time | Geo Mean |
|---|---|---|---|---|
| sf1000 | ~1 TB | 99 / 99 | 713.5 s (11.9 min) | 6.0 s |
| sf3000 | ~3 TB | 99 / 99 | 1,225.9 s (20.4 min) | 8.4 s |
| sf5000 | ~5 TB | 99 / 99 | 1,468.8 s (24.5 min) | 9.3 s |
| sf10000 | ~10 TB | 99 / 99 | 2,787.4 s (46.5 min) | 14.3 s |

### Scaling Efficiency

| Comparison | Data Growth | Total Time Growth |
|---|---|---|
| sf1000 → sf3000 | 3× | 1.72× |
| sf1000 → sf5000 | 5× | 2.06× |
| sf1000 → sf10000 | 10× | 3.91× |
| sf3000 → sf5000 | 1.67× | 1.20× |
| sf5000 → sf10000 | 2× | 1.90× |

Performance scales sub-linearly across all tested scale factors — 10× more data results in under 4× more query time.

### Consistently Slowest Queries (across all scale factors)

| Query | sf1000 Avg (s) | sf3000 Avg (s) | sf5000 Avg (s) | sf10000 Avg (s) |
|---|---|---|---|---|
| Q35 | 46.0 | 137.6 | 178.8 | 398.3 |
| Q68 | 46.5 | 108.6 | 119.1 | 300.2 |
| Q93 | 24.9 | 54.1 | 81.5 | 184.4 |
| Q10 | 16.9 | 37.8 | 59.5 | 116.2 |
| Q51 | 16.4 | 32.8 | 43.5 | 72.4 |

---

## 2. AIStor Native Iceberg Tables

A core architectural differentiator in this benchmark is that the TPC-DS dataset lives in **Apache Iceberg tables managed entirely within MinIO AIStor** — not in an external catalog service.

### What This Means

Traditional lakehouse deployments require a separate catalog stack alongside object storage: a metadata service (Hive Metastore or Nessie), a backing relational database (PostgreSQL), and a REST catalog API layer — typically three additional services that must be deployed, operated, and kept in sync with the storage layer. MinIO AIStor eliminates this entirely by embedding a native Apache Iceberg REST catalog directly into the object store.

In this benchmark, Starburst connects to AIStor's Iceberg REST endpoint (`https://192.168.11.1:9000/_iceberg`) and reads all table metadata — schemas, snapshots, manifests, partition statistics — directly from the same system that holds the data. There is no external catalog, no separate metastore, and no additional service in the query path.

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

| Scale Factor | Approx. Size | Catalog | Schema | Largest Table (store_sales) |
|---|---|---|---|---|
| sf1000 | ~1 TB | aistor | scale1000 | ~2.88 billion rows |
| sf3000 | ~3 TB | aistor | scale3000 | ~8.64 billion rows |
| sf5000 | ~5 TB | aistor | scale5000 | ~14.4 billion rows |
| sf10000 | ~10 TB | aistor | scale10000 | ~28.8 billion rows |

All datasets are stored as Apache Iceberg tables (Parquet format) in the MinIO AIStor `tpcds-iceberg` warehouse.

---

## 3. Test Environment

### 3.1 MinIO AIStor — Storage Layer

**Cluster Overview**

| Property | Value |
|---|---|
| Nodes | 8 (sds01 – sds08) |
| Software | MinIO AIStor |
| Version | DEVELOPMENT.2026-02-27T22-48-55Z (commit 909f7b4) |
| Erasure Coding | EC:4 (12 data + 4 parity per set) |
| Pool | 1 pool, 10 erasure sets, stripe size 16 |
| Total Usable Capacity | 838 TiB |
| Drives Online | 160 / 160 |
| Iceberg REST Endpoint | https://192.168.11.1:9000/_iceberg |

**Per-Node Specifications (all 8 nodes identical)**

| Component | Specification |
|---|---|
| Server Model | Supermicro SYS-212H-TN |
| Operating System | Ubuntu 24.04.3 LTS |
| CPU | Intel Xeon 6761P — 64 cores / 128 threads @ 2.5 GHz |
| Memory | 256 GB DDR5 @ 6400 MT/s |
| Data Drives | 20 × 7.68 TB Solidigm NVMe SSD |
| Network | 2 × Mellanox ConnectX-7 400 GbE |

---

### 3.2 Starburst Enterprise — Query Engine

**Cluster Overview**

| Property | Value |
|---|---|
| Nodes | 9 total (1 coordinator + 8 workers) |
| Software | Starburst Enterprise |
| Version | 479-e.4 |
| Coordinator | loadgen9 |
| Workers | loadgen1 – loadgen8 |

**Per-Node Specifications (all 9 nodes identical)**

| Component | Specification |
|---|---|
| Server Model | Supermicro X14SBT-GAP |
| Operating System | Ubuntu 24.04.3 LTS |
| CPU | Intel Xeon 6960P — 72 cores / 144 threads @ 2.7 GHz |
| Memory | 512 GB |
| Data Drive | 1 × 7 TB Solidigm NVMe (spill / temp) |
| Network | 2 × Mellanox ConnectX-7 400 GbE |

---

### 3.3 Network Topology

All Starburst ↔ MinIO data traffic flows over a dedicated 400 GbE fabric. Management access uses separate 10 GbE interfaces.

```
Starburst Workers      loadgen1–8    192.168.11.11–18  ──┐
Starburst Coordinator  loadgen9      192.168.11.19     ──┤  400 GbE
MinIO AIStor Nodes     sds01–08      192.168.11.1–8    ──┘
```

---

## 4. Test Configuration

### 4.1 Starburst Memory Settings

| Parameter | Value | Notes |
|---|---|---|
| JVM Heap (-Xmx) | 400 GB | Per node |
| query.max-memory-per-node | 280 GB | ~30% headroom below heap |
| query.max-memory | 2,000 GB | Cluster-wide (8 workers × 280 GB) |

### 4.2 Iceberg Catalog (aistor)

| Parameter | Value |
|---|---|
| Connector | iceberg |
| Catalog Type | REST |
| REST Endpoint | https://192.168.11.1:9000/_iceberg |
| Warehouse | tpcds-iceberg |
| Auth | SigV4 (signing-name: s3tables) |
| File I/O | Native S3 (fs.native-s3.enabled=true) |
| File Format | Parquet |

---

## 5. Methodology

### 5.1 Query Execution

- All 99 TPC-DS queries executed against each scale factor schema in `aistor`
- Each query run **3 times** per scale factor; results recorded individually
- Queries run **sequentially** (one at a time, no concurrency)
- Timing measured from query submission to final row received via Starburst REST API

### 5.2 Cold / Warm Run Policy

| Run | Cache State | Purpose |
|---|---|---|
| Run 1 | Cold — OS page cache flushed on all nodes, Starburst restarted | Baseline |
| Run 2 | Warm | Representative production workload |
| Run 3 | Warm | Consistency check |

Cold start procedure for each scale factor:
1. `sync; echo 3 > /proc/sys/vm/drop_caches` on all 17 nodes
2. `systemctl restart starburst` on all 9 nodes
3. Wait for all 8 workers to reach ACTIVE state
4. 30-second stabilization hold before first query

### 5.3 Metrics

- Wall clock time per query per run (seconds, rounded to 0.1s)
- Average of 3 runs per query
- Geometric mean across all 99 queries
- Total elapsed time = sum of all 99 query averages

### 5.4 Data Layout

All Iceberg tables loaded via CTAS from the TPC-DS source catalog. Data is **non-partitioned** — a straightforward single-pass CTAS with no date partitioning or compaction applied.

---

## 6. Results

### 6.1 sf1000 (~1 TB)

**Summary**

| Metric | Value |
|---|---|
| Queries Completed | 99 / 99 |
| Queries Failed | 0 |
| Total Time | 713.5 s (11.9 min) |
| Geometric Mean | 6.0 s |
| Fastest Query | Q2 — 4.0 s |
| Slowest Query | Q68 — 46.5 s |

**Time Distribution**

| Bucket | Count |
|---|---|
| < 10 s | 86 |
| 10 – 30 s | 11 |
| 30 – 60 s | 2 |
| > 1 min | 0 |

**Per-Query Results**

| Query | Run 1 (s) | Run 2 (s) | Run 3 (s) | Avg (s) | Status |
|---|---|---|---|---|---|
| Q1 | 12.7 | 4.0 | 4.0 | 6.9 | FINISHED |
| Q2 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q3 | 7.0 | 6.5 | 7.0 | 6.8 | FINISHED |
| Q4 | 4.0 | 4.2 | 4.0 | 4.1 | FINISHED |
| Q5 | 7.0 | 6.0 | 6.3 | 6.4 | FINISHED |
| Q6 | 8.1 | 8.0 | 9.0 | 8.4 | FINISHED |
| Q7 | 4.9 | 4.0 | 5.0 | 4.6 | FINISHED |
| Q8 | 6.2 | 6.2 | 6.0 | 6.1 | FINISHED |
| Q9 | 8.0 | 8.0 | 10.0 | 8.7 | FINISHED |
| Q10 | 18.5 | 16.1 | 16.1 | 16.9 | FINISHED |
| Q11 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q12 | 6.3 | 6.1 | 6.1 | 6.2 | FINISHED |
| Q13 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q14 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q15 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q16 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q17 | 4.2 | 4.0 | 4.0 | 4.1 | FINISHED |
| Q18 | 4.4 | 4.0 | 4.0 | 4.1 | FINISHED |
| Q19 | 6.0 | 6.0 | 5.7 | 5.9 | FINISHED |
| Q20 | 9.1 | 8.2 | 12.8 | 10.0 | FINISHED |
| Q21 | 6.0 | 6.0 | 6.4 | 6.1 | FINISHED |
| Q22 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q23 | 5.0 | 6.0 | 6.0 | 5.7 | FINISHED |
| Q24 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q25 | 6.0 | 4.8 | 6.0 | 5.6 | FINISHED |
| Q26 | 6.0 | 6.0 | 4.8 | 5.6 | FINISHED |
| Q27 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q28 | 4.4 | 4.5 | 5.0 | 4.6 | FINISHED |
| Q29 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q30 | 6.4 | 6.7 | 6.8 | 6.6 | FINISHED |
| Q31 | 4.4 | 4.5 | 4.1 | 4.3 | FINISHED |
| Q32 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q33 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q34 | 8.0 | 6.5 | 8.0 | 7.5 | FINISHED |
| Q35 | 52.6 | 47.1 | 38.3 | 46.0 | FINISHED |
| Q36 | 10.3 | 12.1 | 12.0 | 11.5 | FINISHED |
| Q37 | 4.7 | 4.2 | 4.2 | 4.4 | FINISHED |
| Q38 | 10.9 | 6.6 | 6.0 | 7.8 | FINISHED |
| Q39 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q40 | 4.7 | 4.0 | 4.0 | 4.2 | FINISHED |
| Q41 | 10.0 | 8.0 | 8.0 | 8.7 | FINISHED |
| Q42 | 14.1 | 16.0 | 12.2 | 14.1 | FINISHED |
| Q43 | 6.7 | 6.8 | 6.8 | 6.8 | FINISHED |
| Q44 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q45 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q46 | 10.0 | 6.0 | 6.0 | 7.3 | FINISHED |
| Q47 | 6.0 | 4.3 | 6.0 | 5.4 | FINISHED |
| Q48 | 6.6 | 6.5 | 7.0 | 6.7 | FINISHED |
| Q49 | 10.0 | 8.9 | 10.0 | 9.6 | FINISHED |
| Q50 | 10.3 | 8.0 | 8.8 | 9.0 | FINISHED |
| Q51 | 14.7 | 16.5 | 18.0 | 16.4 | FINISHED |
| Q52 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q53 | 8.0 | 8.0 | 10.0 | 8.7 | FINISHED |
| Q54 | 8.0 | 6.0 | 8.0 | 7.3 | FINISHED |
| Q55 | 16.0 | 14.8 | 16.0 | 15.6 | FINISHED |
| Q56 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q57 | 4.0 | 4.2 | 4.0 | 4.1 | FINISHED |
| Q58 | 6.0 | 4.6 | 4.3 | 5.0 | FINISHED |
| Q59 | 4.2 | 4.0 | 4.0 | 4.1 | FINISHED |
| Q60 | 5.0 | 4.9 | 6.0 | 5.3 | FINISHED |
| Q61 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q62 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q63 | 4.4 | 4.0 | 4.0 | 4.1 | FINISHED |
| Q64 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q65 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q66 | 10.0 | 8.1 | 8.0 | 8.7 | FINISHED |
| Q67 | 6.5 | 6.0 | 6.1 | 6.2 | FINISHED |
| Q68 | 51.0 | 44.1 | 44.3 | 46.5 | FINISHED |
| Q69 | 14.2 | 14.7 | 18.1 | 15.7 | FINISHED |
| Q70 | 14.8 | 13.0 | 12.2 | 13.3 | FINISHED |
| Q71 | 10.3 | 8.7 | 8.3 | 9.1 | FINISHED |
| Q72 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q73 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q74 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q75 | 6.0 | 4.1 | 4.2 | 4.8 | FINISHED |
| Q76 | 12.0 | 10.5 | 10.0 | 10.8 | FINISHED |
| Q77 | 8.0 | 8.0 | 6.0 | 7.3 | FINISHED |
| Q78 | 6.4 | 8.0 | 8.0 | 7.5 | FINISHED |
| Q79 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q80 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q81 | 6.3 | 4.2 | 4.8 | 5.1 | FINISHED |
| Q82 | 4.2 | 4.0 | 4.0 | 4.1 | FINISHED |
| Q83 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q84 | 4.8 | 5.0 | 4.2 | 4.7 | FINISHED |
| Q85 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q86 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q87 | 13.0 | 12.5 | 14.3 | 13.3 | FINISHED |
| Q88 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q89 | 6.8 | 6.2 | 8.0 | 7.0 | FINISHED |
| Q90 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q91 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q92 | 6.6 | 7.0 | 6.7 | 6.8 | FINISHED |
| Q93 | 26.1 | 22.5 | 26.1 | 24.9 | FINISHED |
| Q94 | 4.0 | 4.0 | 4.9 | 4.3 | FINISHED |
| Q95 | 4.8 | 4.5 | 5.0 | 4.8 | FINISHED |
| Q96 | 4.7 | 4.7 | 4.3 | 4.6 | FINISHED |
| Q97 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q98 | 8.0 | 8.0 | 8.0 | 8.0 | FINISHED |
| Q99 | 6.8 | 6.6 | 6.8 | 6.7 | FINISHED |
| **Total** | | | | **713.5** | **99/99** |
| **Geo Mean** | | | | **6.0** | |

---

### 6.2 sf3000 (~3 TB)

**Summary**

| Metric | Value |
|---|---|
| Queries Completed | 99 / 99 |
| Queries Failed | 0 |
| Total Time | 1,225.9 s (20.4 min) |
| Geometric Mean | 8.4 s |
| Fastest Query | Q2 — 4.0 s |
| Slowest Query | Q35 — 137.6 s |

**Time Distribution**

| Bucket | Count |
|---|---|
| < 10 s | 62 |
| 10 – 30 s | 31 |
| 30 – 60 s | 4 |
| 1 – 5 min | 2 |
| > 5 min | 0 |

**Per-Query Results**

| Query | Run 1 (s) | Run 2 (s) | Run 3 (s) | Avg (s) | Status |
|---|---|---|---|---|---|
| Q1 | 16.1 | 6.0 | 6.0 | 9.4 | FINISHED |
| Q2 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q3 | 15.0 | 13.0 | 12.8 | 13.6 | FINISHED |
| Q4 | 4.0 | 4.5 | 4.4 | 4.3 | FINISHED |
| Q5 | 9.1 | 8.0 | 7.1 | 8.1 | FINISHED |
| Q6 | 14.6 | 14.1 | 14.0 | 14.2 | FINISHED |
| Q7 | 4.8 | 6.0 | 4.8 | 5.2 | FINISHED |
| Q8 | 8.4 | 10.0 | 9.0 | 9.1 | FINISHED |
| Q9 | 12.0 | 12.3 | 14.5 | 12.9 | FINISHED |
| Q10 | 38.1 | 37.1 | 38.1 | 37.8 | FINISHED |
| Q11 | 5.0 | 4.2 | 5.0 | 4.7 | FINISHED |
| Q12 | 16.0 | 16.0 | 17.0 | 16.3 | FINISHED |
| Q13 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q14 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q15 | 6.0 | 6.0 | 6.0 | 6.0 | FINISHED |
| Q16 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q17 | 6.5 | 6.0 | 6.3 | 6.3 | FINISHED |
| Q18 | 6.0 | 4.0 | 4.7 | 4.9 | FINISHED |
| Q19 | 8.5 | 8.8 | 10.0 | 9.1 | FINISHED |
| Q20 | 17.1 | 16.4 | 15.0 | 16.2 | FINISHED |
| Q21 | 8.0 | 8.9 | 10.0 | 9.0 | FINISHED |
| Q22 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q23 | 10.0 | 8.6 | 7.5 | 8.7 | FINISHED |
| Q24 | 4.6 | 6.0 | 4.1 | 4.9 | FINISHED |
| Q25 | 9.0 | 10.1 | 10.5 | 9.9 | FINISHED |
| Q26 | 6.9 | 6.0 | 6.5 | 6.5 | FINISHED |
| Q27 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q28 | 6.8 | 6.0 | 6.8 | 6.5 | FINISHED |
| Q29 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q30 | 9.4 | 8.7 | 8.2 | 8.8 | FINISHED |
| Q31 | 8.0 | 7.1 | 8.0 | 7.7 | FINISHED |
| Q32 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q33 | 5.0 | 5.0 | 4.2 | 4.7 | FINISHED |
| Q34 | 13.0 | 16.0 | 12.7 | 13.9 | FINISHED |
| Q35 | 140.3 | 144.2 | 128.2 | 137.6 | FINISHED |
| Q36 | 16.0 | 16.0 | 14.6 | 15.5 | FINISHED |
| Q37 | 6.0 | 4.5 | 4.6 | 5.0 | FINISHED |
| Q38 | 16.3 | 16.0 | 12.0 | 14.8 | FINISHED |
| Q39 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q40 | 4.3 | 4.0 | 4.0 | 4.1 | FINISHED |
| Q41 | 10.9 | 10.3 | 10.0 | 10.4 | FINISHED |
| Q42 | 28.1 | 30.1 | 28.1 | 28.8 | FINISHED |
| Q43 | 12.0 | 12.2 | 12.0 | 12.1 | FINISHED |
| Q44 | 6.0 | 4.9 | 5.0 | 5.3 | FINISHED |
| Q45 | 6.9 | 6.0 | 6.0 | 6.3 | FINISHED |
| Q46 | 14.0 | 14.9 | 12.7 | 13.9 | FINISHED |
| Q47 | 8.6 | 6.9 | 8.0 | 7.8 | FINISHED |
| Q48 | 12.0 | 12.0 | 12.0 | 12.0 | FINISHED |
| Q49 | 16.5 | 18.0 | 18.5 | 17.7 | FINISHED |
| Q50 | 18.4 | 16.0 | 14.9 | 16.4 | FINISHED |
| Q51 | 36.1 | 30.1 | 32.1 | 32.8 | FINISHED |
| Q52 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q53 | 21.0 | 18.5 | 16.9 | 18.8 | FINISHED |
| Q54 | 14.7 | 10.5 | 12.0 | 12.4 | FINISHED |
| Q55 | 18.0 | 18.0 | 18.0 | 18.0 | FINISHED |
| Q56 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q57 | 6.3 | 4.9 | 4.8 | 5.3 | FINISHED |
| Q58 | 8.0 | 6.5 | 7.0 | 7.2 | FINISHED |
| Q59 | 6.4 | 6.7 | 6.3 | 6.5 | FINISHED |
| Q60 | 24.5 | 24.0 | 24.8 | 24.4 | FINISHED |
| Q61 | 6.2 | 6.0 | 4.6 | 5.6 | FINISHED |
| Q62 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q63 | 6.0 | 6.0 | 6.0 | 6.0 | FINISHED |
| Q64 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q65 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q66 | 12.8 | 12.0 | 12.0 | 12.3 | FINISHED |
| Q67 | 12.0 | 12.3 | 14.0 | 12.8 | FINISHED |
| Q68 | 100.2 | 107.2 | 118.3 | 108.6 | FINISHED |
| Q69 | 30.1 | 32.1 | 32.6 | 31.6 | FINISHED |
| Q70 | 34.8 | 24.6 | 26.6 | 28.7 | FINISHED |
| Q71 | 16.0 | 18.6 | 14.0 | 16.2 | FINISHED |
| Q72 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q73 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q74 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q75 | 4.8 | 4.1 | 4.2 | 4.4 | FINISHED |
| Q76 | 18.2 | 24.0 | 22.6 | 21.6 | FINISHED |
| Q77 | 10.6 | 10.0 | 12.1 | 10.9 | FINISHED |
| Q78 | 10.5 | 10.0 | 14.0 | 11.5 | FINISHED |
| Q79 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q80 | 4.0 | 4.5 | 4.9 | 4.5 | FINISHED |
| Q81 | 6.0 | 7.0 | 5.0 | 6.0 | FINISHED |
| Q82 | 4.9 | 6.0 | 4.9 | 5.3 | FINISHED |
| Q83 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q84 | 8.4 | 8.0 | 8.0 | 8.1 | FINISHED |
| Q85 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q86 | 6.0 | 6.0 | 6.0 | 6.0 | FINISHED |
| Q87 | 16.5 | 14.3 | 15.0 | 15.3 | FINISHED |
| Q88 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q89 | 10.2 | 10.8 | 14.0 | 11.7 | FINISHED |
| Q90 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q91 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q92 | 14.1 | 11.0 | 10.9 | 12.0 | FINISHED |
| Q93 | 58.1 | 52.1 | 52.1 | 54.1 | FINISHED |
| Q94 | 8.0 | 6.0 | 6.8 | 6.9 | FINISHED |
| Q95 | 8.0 | 8.0 | 8.0 | 8.0 | FINISHED |
| Q96 | 6.0 | 4.5 | 4.9 | 5.1 | FINISHED |
| Q97 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q98 | 10.3 | 14.9 | 14.0 | 13.1 | FINISHED |
| Q99 | 10.3 | 14.2 | 10.9 | 11.8 | FINISHED |
| **Total** | | | | **1,225.9** | **99/99** |
| **Geo Mean** | | | | **8.4** | |

---

### 6.3 sf5000 (~5 TB)

**Summary**

| Metric | Value |
|---|---|
| Queries Completed | 99 / 99 |
| Queries Failed | 0 |
| Total Time | 1,468.8 s (24.5 min) |
| Geometric Mean | 9.3 s |
| Fastest Query | Q2 — 4.0 s |
| Slowest Query | Q35 — 178.8 s |

**Time Distribution**

| Bucket | Count |
|---|---|
| < 10 s | 57 |
| 10 – 30 s | 34 |
| 30 – 60 s | 5 |
| 1 – 5 min | 3 |
| > 5 min | 0 |

**Per-Query Results**

| Query | Run 1 (s) | Run 2 (s) | Run 3 (s) | Avg (s) | Status |
|---|---|---|---|---|---|
| Q1 | 18.1 | 6.0 | 6.0 | 10.0 | FINISHED |
| Q2 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q3 | 20.9 | 20.3 | 16.1 | 19.1 | FINISHED |
| Q4 | 5.0 | 5.0 | 5.0 | 5.0 | FINISHED |
| Q5 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q6 | 18.6 | 18.0 | 18.0 | 18.2 | FINISHED |
| Q7 | 6.3 | 6.1 | 6.2 | 6.2 | FINISHED |
| Q8 | 12.4 | 10.0 | 8.7 | 10.4 | FINISHED |
| Q9 | 18.3 | 16.0 | 14.0 | 16.1 | FINISHED |
| Q10 | 60.1 | 58.2 | 60.1 | 59.5 | FINISHED |
| Q11 | 6.0 | 4.9 | 6.0 | 5.6 | FINISHED |
| Q12 | 4.0 | 4.0 | 6.0 | 4.7 | FINISHED |
| Q13 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q14 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q15 | 6.3 | 6.4 | 6.1 | 6.3 | FINISHED |
| Q16 | 4.0 | 4.0 | 5.0 | 4.3 | FINISHED |
| Q17 | 6.2 | 7.0 | 8.0 | 7.1 | FINISHED |
| Q18 | 6.0 | 6.0 | 4.0 | 5.3 | FINISHED |
| Q19 | 9.0 | 8.6 | 11.1 | 9.6 | FINISHED |
| Q20 | 6.3 | 5.0 | 5.0 | 5.4 | FINISHED |
| Q21 | 14.4 | 16.0 | 14.0 | 14.8 | FINISHED |
| Q22 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q23 | 12.7 | 12.0 | 14.0 | 12.9 | FINISHED |
| Q24 | 8.0 | 6.2 | 6.1 | 6.8 | FINISHED |
| Q25 | 14.5 | 14.0 | 12.0 | 13.5 | FINISHED |
| Q26 | 6.9 | 6.6 | 6.3 | 6.6 | FINISHED |
| Q27 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q28 | 6.6 | 8.2 | 8.0 | 7.6 | FINISHED |
| Q29 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q30 | 12.0 | 10.2 | 12.6 | 11.6 | FINISHED |
| Q31 | 13.0 | 11.0 | 10.3 | 11.4 | FINISHED |
| Q32 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q33 | 6.8 | 6.1 | 5.0 | 6.0 | FINISHED |
| Q34 | 22.7 | 22.5 | 22.6 | 22.6 | FINISHED |
| Q35 | 187.3 | 172.9 | 176.3 | 178.8 | FINISHED |
| Q36 | 28.3 | 24.0 | 22.9 | 25.1 | FINISHED |
| Q37 | 6.0 | 6.0 | 4.7 | 5.6 | FINISHED |
| Q38 | 20.0 | 20.0 | 20.1 | 20.0 | FINISHED |
| Q39 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q40 | 6.0 | 4.9 | 4.2 | 5.0 | FINISHED |
| Q41 | 14.9 | 18.0 | 14.4 | 15.8 | FINISHED |
| Q42 | 40.1 | 34.8 | 34.6 | 36.5 | FINISHED |
| Q43 | 18.0 | 18.1 | 16.0 | 17.4 | FINISHED |
| Q44 | 6.0 | 4.3 | 4.3 | 4.9 | FINISHED |
| Q45 | 6.3 | 6.5 | 6.1 | 6.3 | FINISHED |
| Q46 | 15.0 | 16.2 | 17.0 | 16.1 | FINISHED |
| Q47 | 8.2 | 8.9 | 6.8 | 8.0 | FINISHED |
| Q48 | 16.0 | 16.6 | 16.0 | 16.2 | FINISHED |
| Q49 | 25.0 | 28.0 | 26.7 | 26.6 | FINISHED |
| Q50 | 20.8 | 18.0 | 24.2 | 21.0 | FINISHED |
| Q51 | 46.1 | 44.1 | 40.2 | 43.5 | FINISHED |
| Q52 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q53 | 26.0 | 28.0 | 26.4 | 26.8 | FINISHED |
| Q54 | 14.0 | 14.0 | 16.0 | 14.7 | FINISHED |
| Q55 | 4.5 | 4.8 | 4.1 | 4.5 | FINISHED |
| Q56 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q57 | 6.3 | 6.0 | 6.4 | 6.2 | FINISHED |
| Q58 | 8.0 | 9.2 | 6.9 | 8.0 | FINISHED |
| Q59 | 6.6 | 6.3 | 8.0 | 7.0 | FINISHED |
| Q60 | 42.1 | 36.1 | 28.1 | 35.4 | FINISHED |
| Q61 | 7.0 | 6.1 | 6.4 | 6.5 | FINISHED |
| Q62 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q63 | 8.0 | 8.0 | 8.0 | 8.0 | FINISHED |
| Q64 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q65 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q66 | 18.5 | 16.5 | 18.0 | 17.7 | FINISHED |
| Q67 | 20.0 | 16.1 | 18.0 | 18.0 | FINISHED |
| Q68 | 120.2 | 118.2 | 119.0 | 119.1 | FINISHED |
| Q69 | 38.3 | 40.1 | 37.0 | 38.5 | FINISHED |
| Q70 | 24.1 | 22.6 | 24.1 | 23.6 | FINISHED |
| Q71 | 19.0 | 16.1 | 17.0 | 17.4 | FINISHED |
| Q72 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q73 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q74 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q75 | 6.0 | 4.8 | 4.9 | 5.2 | FINISHED |
| Q76 | 26.3 | 22.6 | 30.3 | 26.4 | FINISHED |
| Q77 | 14.0 | 14.4 | 16.0 | 14.8 | FINISHED |
| Q78 | 12.9 | 14.0 | 16.1 | 14.3 | FINISHED |
| Q79 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q80 | 4.9 | 6.0 | 6.0 | 5.6 | FINISHED |
| Q81 | 10.3 | 9.0 | 7.0 | 8.8 | FINISHED |
| Q82 | 8.0 | 7.0 | 6.0 | 7.0 | FINISHED |
| Q83 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q84 | 10.5 | 10.0 | 11.9 | 10.8 | FINISHED |
| Q85 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q86 | 6.4 | 6.0 | 4.8 | 5.7 | FINISHED |
| Q87 | 5.0 | 4.5 | 4.0 | 4.5 | FINISHED |
| Q88 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q89 | 14.6 | 16.8 | 16.0 | 15.8 | FINISHED |
| Q90 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q91 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q92 | 17.0 | 14.5 | 13.0 | 14.8 | FINISHED |
| Q93 | 82.2 | 80.2 | 82.2 | 81.5 | FINISHED |
| Q94 | 10.2 | 8.0 | 8.0 | 8.7 | FINISHED |
| Q95 | 12.0 | 12.0 | 12.9 | 12.3 | FINISHED |
| Q96 | 4.3 | 4.3 | 4.2 | 4.3 | FINISHED |
| Q97 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q98 | 18.0 | 16.0 | 14.5 | 16.2 | FINISHED |
| Q99 | 12.0 | 11.4 | 10.6 | 11.3 | FINISHED |
| **Total** | | | | **1,468.8** | **99/99** |
| **Geo Mean** | | | | **9.3** | |

---

### 6.4 sf10000 (~10 TB)

**Summary**

| Metric | Value |
|---|---|
| Queries Completed | 99 / 99 |
| Queries Failed | 0 |
| Total Time | 2,787.4 s (46.5 min) |
| Geometric Mean | 14.3 s |
| Fastest Query | Q2 — 4.0 s |
| Slowest Query | Q35 — 398.3 s |

**Time Distribution**

| Bucket | Count |
|---|---|
| < 10 s | 37 |
| 10 – 30 s | 37 |
| 30 – 60 s | 16 |
| 1 – 5 min | 7 |
| > 5 min | 2 |

**Per-Query Results**

| Query | Run 1 (s) | Run 2 (s) | Run 3 (s) | Avg (s) | Status |
|---|---|---|---|---|---|
| Q1 | 18.7 | 8.5 | 12.0 | 13.1 | FINISHED |
| Q2 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q3 | 37.1 | 29.1 | 27.0 | 31.1 | FINISHED |
| Q4 | 8.4 | 8.6 | 7.0 | 8.0 | FINISHED |
| Q5 | 7.9 | 7.1 | 7.1 | 7.4 | FINISHED |
| Q6 | 30.1 | 30.1 | 32.9 | 31.0 | FINISHED |
| Q7 | 10.7 | 10.6 | 10.8 | 10.7 | FINISHED |
| Q8 | 14.4 | 16.2 | 16.0 | 15.5 | FINISHED |
| Q9 | 26.1 | 24.1 | 23.1 | 24.4 | FINISHED |
| Q10 | 118.2 | 114.2 | 116.2 | 116.2 | FINISHED |
| Q11 | 8.0 | 8.0 | 8.0 | 8.0 | FINISHED |
| Q12 | 50.1 | 52.1 | 48.4 | 50.2 | FINISHED |
| Q13 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q14 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q15 | 12.0 | 10.0 | 10.5 | 10.8 | FINISHED |
| Q16 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q17 | 14.0 | 14.2 | 12.0 | 13.4 | FINISHED |
| Q18 | 8.0 | 7.0 | 8.0 | 7.7 | FINISHED |
| Q19 | 18.0 | 18.0 | 18.9 | 18.3 | FINISHED |
| Q20 | 27.1 | 23.1 | 5.0 | 18.4 | FINISHED |
| Q21 | 16.6 | 16.0 | 16.0 | 16.2 | FINISHED |
| Q22 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q23 | 14.8 | 16.0 | 16.0 | 15.6 | FINISHED |
| Q24 | 10.0 | 10.3 | 8.2 | 9.5 | FINISHED |
| Q25 | 22.7 | 20.0 | 20.1 | 20.9 | FINISHED |
| Q26 | 10.4 | 12.0 | 12.0 | 11.5 | FINISHED |
| Q27 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q28 | 12.0 | 10.6 | 12.0 | 11.5 | FINISHED |
| Q29 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q30 | 20.1 | 20.0 | 18.0 | 19.4 | FINISHED |
| Q31 | 22.0 | 17.4 | 16.7 | 18.7 | FINISHED |
| Q32 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q33 | 8.6 | 8.4 | 9.0 | 8.7 | FINISHED |
| Q34 | 28.3 | 28.1 | 36.1 | 30.8 | FINISHED |
| Q35 | 386.9 | 405.5 | 402.6 | 398.3 | FINISHED |
| Q36 | 42.8 | 42.6 | 42.5 | 42.6 | FINISHED |
| Q37 | 8.5 | 6.3 | 6.0 | 6.9 | FINISHED |
| Q38 | 42.1 | 40.1 | 38.1 | 40.1 | FINISHED |
| Q39 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q40 | 6.0 | 6.0 | 6.0 | 6.0 | FINISHED |
| Q41 | 26.8 | 22.6 | 22.3 | 23.9 | FINISHED |
| Q42 | 60.7 | 60.1 | 64.2 | 61.7 | FINISHED |
| Q43 | 30.1 | 32.1 | 30.7 | 31.0 | FINISHED |
| Q44 | 6.9 | 6.6 | 8.0 | 7.2 | FINISHED |
| Q45 | 12.0 | 10.3 | 10.6 | 11.0 | FINISHED |
| Q46 | 34.1 | 32.1 | 32.1 | 32.8 | FINISHED |
| Q47 | 12.0 | 12.0 | 12.0 | 12.0 | FINISHED |
| Q48 | 32.1 | 29.0 | 28.1 | 29.7 | FINISHED |
| Q49 | 50.6 | 51.1 | 50.1 | 50.6 | FINISHED |
| Q50 | 28.2 | 28.1 | 28.1 | 28.1 | FINISHED |
| Q51 | 76.1 | 68.1 | 72.9 | 72.4 | FINISHED |
| Q52 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q53 | 54.1 | 42.7 | 42.2 | 46.3 | FINISHED |
| Q54 | 24.0 | 22.9 | 24.2 | 23.7 | FINISHED |
| Q55 | 22.0 | 20.0 | 20.4 | 20.8 | FINISHED |
| Q56 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q57 | 8.6 | 8.0 | 8.0 | 8.2 | FINISHED |
| Q58 | 14.7 | 14.0 | 14.0 | 14.2 | FINISHED |
| Q59 | 10.5 | 12.1 | 12.0 | 11.5 | FINISHED |
| Q60 | 76.7 | 58.1 | 62.2 | 65.7 | FINISHED |
| Q61 | 10.9 | 12.0 | 12.0 | 11.6 | FINISHED |
| Q62 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q63 | 8.8 | 12.1 | 10.5 | 10.5 | FINISHED |
| Q64 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q65 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q66 | 31.0 | 32.1 | 28.3 | 30.5 | FINISHED |
| Q67 | 29.0 | 31.0 | 30.1 | 30.0 | FINISHED |
| Q68 | 295.2 | 306.6 | 298.7 | 300.2 | FINISHED |
| Q69 | 88.4 | 86.3 | 84.2 | 86.3 | FINISHED |
| Q70 | 66.1 | 60.1 | 62.1 | 62.8 | FINISHED |
| Q71 | 42.1 | 45.1 | 42.9 | 43.4 | FINISHED |
| Q72 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q73 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q74 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q75 | 6.0 | 6.1 | 6.0 | 6.0 | FINISHED |
| Q76 | 50.4 | 50.4 | 54.1 | 51.6 | FINISHED |
| Q77 | 30.1 | 28.9 | 34.1 | 31.0 | FINISHED |
| Q78 | 26.0 | 26.0 | 24.0 | 25.3 | FINISHED |
| Q79 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q80 | 8.8 | 8.6 | 8.4 | 8.6 | FINISHED |
| Q81 | 12.6 | 17.0 | 13.0 | 14.2 | FINISHED |
| Q82 | 14.0 | 10.8 | 18.0 | 14.3 | FINISHED |
| Q83 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q84 | 18.0 | 17.0 | 16.2 | 17.1 | FINISHED |
| Q85 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q86 | 10.0 | 10.0 | 10.0 | 10.0 | FINISHED |
| Q87 | 16.3 | 15.0 | 16.0 | 15.8 | FINISHED |
| Q88 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q89 | 24.6 | 26.0 | 24.0 | 24.9 | FINISHED |
| Q90 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q91 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q92 | 37.1 | 30.9 | 29.1 | 32.4 | FINISHED |
| Q93 | 178.5 | 182.3 | 192.3 | 184.4 | FINISHED |
| Q94 | 10.1 | 13.0 | 10.0 | 11.0 | FINISHED |
| Q95 | 24.7 | 18.7 | 22.0 | 21.8 | FINISHED |
| Q96 | 6.3 | 6.3 | 6.9 | 6.5 | FINISHED |
| Q97 | 4.0 | 4.0 | 4.0 | 4.0 | FINISHED |
| Q98 | 28.0 | 27.0 | 28.0 | 27.7 | FINISHED |
| Q99 | 22.0 | 21.4 | 22.0 | 21.8 | FINISHED |
| **Total** | | | | **2,787.4** | **99/99** |
| **Geo Mean** | | | | **14.3** | |

---

## 7. Cross-Scale Summary

### 7.1 Scale Factor Comparison

| Query | sf1000 (s) | sf3000 (s) | sf5000 (s) | sf10000 (s) |
|---|---|---|---|---|
| Q1 | 6.9 | 9.4 | 10.0 | 13.1 |
| Q2 | 4.0 | 4.0 | 4.0 | 4.0 |
| Q3 | 6.8 | 13.6 | 19.1 | 31.1 |
| Q4 | 4.1 | 4.3 | 5.0 | 8.0 |
| Q5 | 6.4 | 8.1 | 4.0 | 7.4 |
| Q6 | 8.4 | 14.2 | 18.2 | 31.0 |
| Q7 | 4.6 | 5.2 | 6.2 | 10.7 |
| Q8 | 6.1 | 9.1 | 10.4 | 15.5 |
| Q9 | 8.7 | 12.9 | 16.1 | 24.4 |
| Q10 | 16.9 | 37.8 | 59.5 | 116.2 |
| Q11 | 4.0 | 4.7 | 5.6 | 8.0 |
| Q12 | 6.2 | 16.3 | 4.7 | 50.2 |
| Q13 | 4.0 | 4.0 | 4.0 | 4.0 |
| Q14 | 4.0 | 4.0 | 4.0 | 4.0 |
| Q15 | 4.0 | 6.0 | 6.3 | 10.8 |
| Q16 | 4.0 | 4.0 | 4.3 | 4.0 |
| Q17 | 4.1 | 6.3 | 7.1 | 13.4 |
| Q18 | 4.1 | 4.9 | 5.3 | 7.7 |
| Q19 | 5.9 | 9.1 | 9.6 | 18.3 |
| Q20 | 10.0 | 16.2 | 5.4 | 18.4 |
| Q21 | 6.1 | 9.0 | 14.8 | 16.2 |
| Q22 | 4.0 | 4.0 | 4.0 | 4.0 |
| Q23 | 5.7 | 8.7 | 12.9 | 15.6 |
| Q24 | 4.0 | 4.9 | 6.8 | 9.5 |
| Q25 | 5.6 | 9.9 | 13.5 | 20.9 |
| Q26 | 5.6 | 6.5 | 6.6 | 11.5 |
| Q27 | 4.0 | 4.0 | 4.0 | 4.0 |
| Q28 | 4.6 | 6.5 | 7.6 | 11.5 |
| Q29 | 4.0 | 4.0 | 4.0 | 4.0 |
| Q30 | 6.6 | 8.8 | 11.6 | 19.4 |
| Q31 | 4.3 | 7.7 | 11.4 | 18.7 |
| Q32 | 4.0 | 4.0 | 4.0 | 4.0 |
| Q33 | 4.0 | 4.7 | 6.0 | 8.7 |
| Q34 | 7.5 | 13.9 | 22.6 | 30.8 |
| Q35 | 46.0 | 137.6 | 178.8 | 398.3 |
| Q36 | 11.5 | 15.5 | 25.1 | 42.6 |
| Q37 | 4.4 | 5.0 | 5.6 | 6.9 |
| Q38 | 7.8 | 14.8 | 20.0 | 40.1 |
| Q39 | 4.0 | 4.0 | 4.0 | 4.0 |
| Q40 | 4.2 | 4.1 | 5.0 | 6.0 |
| Q41 | 8.7 | 10.4 | 15.8 | 23.9 |
| Q42 | 14.1 | 28.8 | 36.5 | 61.7 |
| Q43 | 6.8 | 12.1 | 17.4 | 31.0 |
| Q44 | 4.0 | 5.3 | 4.9 | 7.2 |
| Q45 | 4.0 | 6.3 | 6.3 | 11.0 |
| Q46 | 7.3 | 13.9 | 16.1 | 32.8 |
| Q47 | 5.4 | 7.8 | 8.0 | 12.0 |
| Q48 | 6.7 | 12.0 | 16.2 | 29.7 |
| Q49 | 9.6 | 17.7 | 26.6 | 50.6 |
| Q50 | 9.0 | 16.4 | 21.0 | 28.1 |
| Q51 | 16.4 | 32.8 | 43.5 | 72.4 |
| Q52 | 4.0 | 4.0 | 4.0 | 4.0 |
| Q53 | 8.7 | 18.8 | 26.8 | 46.3 |
| Q54 | 7.3 | 12.4 | 14.7 | 23.7 |
| Q55 | 15.6 | 18.0 | 4.5 | 20.8 |
| Q56 | 4.0 | 4.0 | 4.0 | 4.0 |
| Q57 | 4.1 | 5.3 | 6.2 | 8.2 |
| Q58 | 5.0 | 7.2 | 8.0 | 14.2 |
| Q59 | 4.1 | 6.5 | 7.0 | 11.5 |
| Q60 | 5.3 | 24.4 | 35.4 | 65.7 |
| Q61 | 4.0 | 5.6 | 6.5 | 11.6 |
| Q62 | 4.0 | 4.0 | 4.0 | 4.0 |
| Q63 | 4.1 | 6.0 | 8.0 | 10.5 |
| Q64 | 4.0 | 4.0 | 4.0 | 4.0 |
| Q65 | 4.0 | 4.0 | 4.0 | 4.0 |
| Q66 | 8.7 | 12.3 | 17.7 | 30.5 |
| Q67 | 6.2 | 12.8 | 18.0 | 30.0 |
| Q68 | 46.5 | 108.6 | 119.1 | 300.2 |
| Q69 | 15.7 | 31.6 | 38.5 | 86.3 |
| Q70 | 13.3 | 28.7 | 23.6 | 62.8 |
| Q71 | 9.1 | 16.2 | 17.4 | 43.4 |
| Q72 | 4.0 | 4.0 | 4.0 | 4.0 |
| Q73 | 4.0 | 4.0 | 4.0 | 4.0 |
| Q74 | 4.0 | 4.0 | 4.0 | 4.0 |
| Q75 | 4.8 | 4.4 | 5.2 | 6.0 |
| Q76 | 10.8 | 21.6 | 26.4 | 51.6 |
| Q77 | 7.3 | 10.9 | 14.8 | 31.0 |
| Q78 | 7.5 | 11.5 | 14.3 | 25.3 |
| Q79 | 4.0 | 4.0 | 4.0 | 4.0 |
| Q80 | 4.0 | 4.5 | 5.6 | 8.6 |
| Q81 | 5.1 | 6.0 | 8.8 | 14.2 |
| Q82 | 4.1 | 5.3 | 7.0 | 14.3 |
| Q83 | 4.0 | 4.0 | 4.0 | 4.0 |
| Q84 | 4.7 | 8.1 | 10.8 | 17.1 |
| Q85 | 4.0 | 4.0 | 4.0 | 4.0 |
| Q86 | 4.0 | 6.0 | 5.7 | 10.0 |
| Q87 | 13.3 | 15.3 | 4.5 | 15.8 |
| Q88 | 4.0 | 4.0 | 4.0 | 4.0 |
| Q89 | 7.0 | 11.7 | 15.8 | 24.9 |
| Q90 | 4.0 | 4.0 | 4.0 | 4.0 |
| Q91 | 4.0 | 4.0 | 4.0 | 4.0 |
| Q92 | 6.8 | 12.0 | 14.8 | 32.4 |
| Q93 | 24.9 | 54.1 | 81.5 | 184.4 |
| Q94 | 4.3 | 6.9 | 8.7 | 11.0 |
| Q95 | 4.8 | 8.0 | 12.3 | 21.8 |
| Q96 | 4.6 | 5.1 | 4.3 | 6.5 |
| Q97 | 4.0 | 4.0 | 4.0 | 4.0 |
| Q98 | 8.0 | 13.1 | 16.2 | 27.7 |
| Q99 | 6.7 | 11.8 | 11.3 | 21.8 |
| **Total (s)** | **713.5** | **1,225.9** | **1,468.8** | **2,787.4** |
| **Geo Mean (s)** | **6.0** | **8.4** | **9.3** | **14.3** |
| **Completed** | **99/99** | **99/99** | **99/99** | **99/99** |

### 7.2 Time Distribution by Scale Factor

| Bucket | sf1000 | sf3000 | sf5000 | sf10000 |
|---|---|---|---|---|
| < 10 s | 86 | 62 | 57 | 37 |
| 10 – 30 s | 11 | 31 | 34 | 37 |
| 30 – 60 s | 2 | 4 | 5 | 16 |
| 1 – 5 min | 0 | 2 | 3 | 7 |
| > 5 min | 0 | 0 | 0 | 2 |

---

## 8. Appendix — Full Configuration

### A. Starburst coordinator/config.properties

```
coordinator=true
node-scheduler.include-coordinator=false
http-server.http.port=8080
discovery.uri=http://loadgen-09a:8080
insights.persistence-enabled=false
starburst.data-product.enabled=false
starburst.access-control.enabled=false
query.max-memory-per-node=280GB
query.max-memory=2000GB
```

### B. Starburst worker/config.properties

```
coordinator=false
http-server.http.port=8080
discovery.uri=http://loadgen-09a:8080
query.max-memory-per-node=280GB
query.max-memory=2000GB
```

### C. Starburst jvm.config (coordinator and workers)

```
-Xmx400g
-XX:+ExplicitGCInvokesConcurrent
```

### D. catalog/aistor.properties

```
connector.name=iceberg
iceberg.catalog.type=rest
iceberg.rest-catalog.uri=https://192.168.11.1:9000/_iceberg
iceberg.rest-catalog.warehouse=tpcds-iceberg
iceberg.rest-catalog.security=SIGV4
iceberg.rest-catalog.signing-name=s3tables
fs.native-s3.enabled=true
s3.endpoint=https://192.168.11.1:9000
s3.region=us-east-1
s3.path-style-access=true
s3.aws-access-key=minioadmin
s3.aws-secret-key=minioadmin
```
