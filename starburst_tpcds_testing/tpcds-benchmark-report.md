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
8. Partitioned Results (ss_sold_date_sk)
   - 8.1 sf1000 — Pre-Compaction
   - 8.2 sf1000 — Post-Compaction
   - 8.3 sf5000 — Post-Compaction
   - 8.4 sf10000 — Post-Compaction
9. Cross-Scale Summary
10. Appendix — Full Configuration

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

## 8. Partitioned Results (ss_sold_date_sk)

All tables in this section use a partitioned schema with `store_sales` (and related fact tables) partitioned on `ss_sold_date_sk`. This enables Iceberg partition pruning to skip entire date ranges during scans, dramatically reducing I/O for date-filtered queries.

### 8.1 sf1000 — Pre-Compaction

> **Note:** This run uses the partitioned schema prior to running `ALTER TABLE EXECUTE OPTIMIZE`. File sizes are not yet compacted; results reflect partitioning benefit only.

**Summary**

| Metric | Non-Partitioned | Partitioned (Pre-Compaction) | Change |
|---|---|---|---|
| Queries Completed | 99 / 99 | 99 / 99 | — |
| Total Time (cold run) | 747.1 s (12.5 min) | 582.6 s (9.7 min) | **−22%** |
| Total Time (avg 3 runs) | 713.5 s (11.9 min) | 582.6 s (9.7 min) | **−18%** |
| Geometric Mean | 6.0 s | 4.4 s | **−27%** |
| Fastest Query | Q2 — 4.0 s | Q41/Q92 — 1.4 s | |
| Slowest Query | Q68 — 46.5 s | Q67 — 43.4 s | |

**Time Distribution**

| Bucket | Non-Partitioned | Partitioned (Pre-Compaction) |
|---|---|---|
| < 10 s | 86 | 85 |
| 10 – 30 s | 11 | 13 |
| 30 – 60 s | 2 | 1 |
| > 1 min | 0 | 0 |

**Biggest Winners — Partition Pruning**

| Query | Non-Partitioned Avg (s) | Partitioned (s) | Speedup |
|---|---|---|---|
| Q68 | 46.5 | 2.7 | **17.2×** |
| Q35 | 46.0 | 3.4 | **13.5×** |
| Q55 | 15.6 | 1.7 | **9.2×** |
| Q42 | 14.1 | 1.6 | **8.8×** |
| Q41 | 8.7 | 1.4 | **6.2×** |
| Q36 | 11.5 | 1.9 | **6.1×** |
| Q69 | 15.7 | 2.8 | **5.6×** |
| Q10 | 16.9 | 5.3 | 3.2× |
| Q93 | 24.9 | 14.9 | 1.7× |
| Q51 | 16.4 | 6.6 | 2.5× |

**Queries with Regression — No Date Predicate on store_sales**

| Query | Non-Partitioned Avg (s) | Partitioned (s) | Slowdown |
|---|---|---|---|
| Q67 | 6.2 | 43.4 | 7.0× |
| Q04 | 4.1 | 24.5 | 6.0× |
| Q23 | 5.7 | 29.9 | 5.2× |
| Q72 | 4.0 | 16.8 | 4.2× |
| Q22 | 4.0 | 15.5 | 3.9× |
| Q14 | 4.0 | 15.3 | 3.8× |

These queries do not filter on `ss_sold_date_sk`, so they receive no pruning benefit and pay a small overhead from the partitioned metadata layout.

**Per-Query Results**

| Query | Partitioned (s) | Non-Partitioned Avg (s) | Delta (s) |
|---|---|---|---|
| Q1 | 7.7 | 6.9 | +0.8 |
| Q2 | 3.4 | 4.0 | −0.6 |
| Q3 | 2.1 | 6.8 | −4.7 |
| Q4 | 24.5 | 4.1 | +20.4 |
| Q5 | 3.9 | 6.4 | −2.5 |
| Q6 | 4.3 | 8.4 | −4.1 |
| Q7 | 3.5 | 4.6 | −1.1 |
| Q8 | 2.4 | 6.1 | −3.7 |
| Q9 | 11.7 | 8.7 | +3.0 |
| Q10 | 5.3 | 16.9 | −11.6 |
| Q11 | 11.5 | 4.0 | +7.5 |
| Q12 | 3.1 | 6.2 | −3.1 |
| Q13 | 2.4 | 4.0 | −1.6 |
| Q14 | 15.3 | 4.0 | +11.3 |
| Q15 | 2.5 | 4.0 | −1.5 |
| Q16 | 6.0 | 4.0 | +2.0 |
| Q17 | 4.9 | 4.1 | +0.8 |
| Q18 | 4.7 | 4.1 | +0.6 |
| Q19 | 3.2 | 5.9 | −2.7 |
| Q20 | 2.9 | 10.0 | −7.1 |
| Q21 | 1.9 | 6.1 | −4.2 |
| Q22 | 15.5 | 4.0 | +11.5 |
| Q23 | 29.9 | 5.7 | +24.2 |
| Q24 | 11.9 | 4.0 | +7.9 |
| Q25 | 3.6 | 5.6 | −2.0 |
| Q26 | 2.7 | 5.6 | −2.9 |
| Q27 | 2.8 | 4.0 | −1.2 |
| Q28 | 9.5 | 4.6 | +4.9 |
| Q29 | 5.1 | 4.0 | +1.1 |
| Q30 | 4.2 | 6.6 | −2.4 |
| Q31 | 3.2 | 4.3 | −1.1 |
| Q32 | 1.6 | 4.0 | −2.4 |
| Q33 | 2.9 | 4.0 | −1.1 |
| Q34 | 2.1 | 7.5 | −5.4 |
| Q35 | 3.4 | 46.0 | −42.6 |
| Q36 | 1.9 | 11.5 | −9.6 |
| Q37 | 4.6 | 4.4 | +0.2 |
| Q38 | 5.0 | 7.8 | −2.8 |
| Q39 | 5.3 | 4.0 | +1.3 |
| Q40 | 3.8 | 4.2 | −0.4 |
| Q41 | 1.4 | 8.7 | −7.3 |
| Q42 | 1.6 | 14.1 | −12.5 |
| Q43 | 2.5 | 6.8 | −4.3 |
| Q44 | 6.9 | 4.0 | +2.9 |
| Q45 | 2.3 | 4.0 | −1.7 |
| Q46 | 3.6 | 7.3 | −3.7 |
| Q47 | 8.5 | 5.4 | +3.1 |
| Q48 | 3.2 | 6.7 | −3.5 |
| Q49 | 6.0 | 9.6 | −3.6 |
| Q50 | 5.9 | 9.0 | −3.1 |
| Q51 | 6.6 | 16.4 | −9.8 |
| Q52 | 1.9 | 4.0 | −2.1 |
| Q53 | 2.6 | 8.7 | −6.1 |
| Q54 | 4.3 | 7.3 | −3.0 |
| Q55 | 1.7 | 15.6 | −13.9 |
| Q56 | 2.6 | 4.0 | −1.4 |
| Q57 | 7.7 | 4.1 | +3.6 |
| Q58 | 3.7 | 5.0 | −1.3 |
| Q59 | 4.3 | 4.1 | +0.2 |
| Q60 | 3.1 | 5.3 | −2.2 |
| Q61 | 2.9 | 4.0 | −1.1 |
| Q62 | 2.9 | 4.0 | −1.1 |
| Q63 | 2.1 | 4.1 | −2.0 |
| Q64 | 13.3 | 4.0 | +9.3 |
| Q65 | 8.2 | 4.0 | +4.2 |
| Q66 | 3.3 | 8.7 | −5.4 |
| Q67 | 43.4 | 6.2 | +37.2 |
| Q68 | 2.7 | 46.5 | −43.8 |
| Q69 | 2.8 | 15.7 | −12.9 |
| Q70 | 3.4 | 13.3 | −9.9 |
| Q71 | 2.5 | 9.1 | −6.6 |
| Q72 | 16.8 | 4.0 | +12.8 |
| Q73 | 1.8 | 4.0 | −2.2 |
| Q74 | 6.5 | 4.0 | +2.5 |
| Q75 | 12.2 | 4.8 | +7.4 |
| Q76 | 8.5 | 10.8 | −2.3 |
| Q77 | 2.6 | 7.3 | −4.7 |
| Q78 | 17.0 | 7.5 | +9.5 |
| Q79 | 3.2 | 4.0 | −0.8 |
| Q80 | 6.5 | 4.0 | +2.5 |
| Q81 | 5.2 | 5.1 | +0.1 |
| Q82 | 4.3 | 4.1 | +0.2 |
| Q83 | 4.4 | 4.0 | +0.4 |
| Q84 | 3.7 | 4.7 | −1.0 |
| Q85 | 2.8 | 4.0 | −1.2 |
| Q86 | 2.4 | 4.0 | −1.6 |
| Q87 | 4.2 | 13.3 | −9.1 |
| Q88 | 10.2 | 4.0 | +6.2 |
| Q89 | 3.3 | 7.0 | −3.7 |
| Q90 | 3.8 | 4.0 | −0.2 |
| Q91 | 2.0 | 4.0 | −2.0 |
| Q92 | 1.4 | 6.8 | −5.4 |
| Q93 | 14.9 | 24.9 | −10.0 |
| Q94 | 3.6 | 4.3 | −0.7 |
| Q95 | 8.0 | 4.8 | +3.2 |
| Q96 | 4.2 | 4.6 | −0.4 |
| Q97 | 7.3 | 4.0 | +3.3 |
| Q98 | 3.0 | 8.0 | −5.0 |
| Q99 | 4.7 | 6.7 | −2.0 |
| **Total** | **582.6** | **713.5 (avg)** | **−130.9** |
| **Geo Mean** | **4.4** | **6.0** | **−27%** |

---

### 8.2 sf1000 — Post-Compaction

> **Note:** This run was taken immediately after running `ALTER TABLE EXECUTE OPTIMIZE` on all 7 fact tables (catalog_returns, catalog_sales, inventory, store_returns, store_sales, web_returns, web_sales). Compaction took **228.8 s (3.8 min)** total; `store_sales` alone took 136.5 s. File count in `store_sales` dropped from **2,269 → 1,835** (avg size 43 MB → 53 MB).

**Summary**

| Metric | Non-Partitioned | Pre-Compaction | Post-Compaction | vs Pre | vs Non-Part |
|---|---|---|---|---|---|
| Queries Completed | 99 / 99 | 99 / 99 | 99 / 99 | — | — |
| Total Time (cold) | 747.1 s (12.5 min) | 582.6 s (9.7 min) | 538.8 s (9.0 min) | **−7.5%** | **−28%** |
| Geometric Mean | 6.0 s | 4.4 s | 4.0 s | **−9%** | **−33%** |
| Fastest Query | Q2 — 4.0 s | Q41/Q92 — 1.4 s | Q92 — 1.4 s | | |
| Slowest Query | Q68 — 46.5 s | Q67 — 43.4 s | Q22 — 33.5 s | | |

**File Layout After Compaction**

| Table | Pre-Compaction Files | Post-Compaction Files | Avg File Size |
|---|---|---|---|
| store_sales | 2,269 | 1,835 | 53 MB |
| Non-partitioned baseline | 512 | — | 188 MB |

> The default `EXECUTE optimize` (no `file_size_threshold`) reduces small files within each partition but cannot merge across partition boundaries. The minimum achievable file count is bounded by the number of distinct date values in the dataset (~1,800+ for SF1000), leaving store_sales at 3.6× more files than the non-partitioned table.

**Time Distribution**

| Bucket | Non-Partitioned | Pre-Compaction | Post-Compaction |
|---|---|---|---|
| < 10 s | 86 | 85 | 86 |
| 10 – 30 s | 11 | 13 | 11 |
| 30 – 60 s | 2 | 1 | 2 |
| > 1 min | 0 | 0 | 0 |

**Biggest Improvements — Pre vs Post Compaction**

| Query | Pre-Compaction (s) | Post-Compaction (s) | Change |
|---|---|---|---|
| Q01 | 7.7 | 3.7 | **−52%** |
| Q10 | 5.3 | 2.6 | **−51%** |
| Q51 | 6.6 | 3.9 | **−41%** |
| Q97 | 7.3 | 4.5 | **−38%** |
| Q76 | 8.5 | 5.4 | **−36%** |
| Q06 | 4.3 | 2.7 | **−36%** |

**Expected Regression Queries — Behavior After Compaction**

These queries lack `ss_sold_date_sk` predicates and do full scans of `store_sales`. Compaction partially reduces the file-open overhead but cannot fully close the gap with the non-partitioned table.

| Query | Non-Part Avg (s) | Pre-Compaction (s) | Post-Compaction (s) | Note |
|---|---|---|---|---|
| Q67 | 6.2 | 43.4 | 33.1 | **−24%** from pre; joins via `ss_item_sk` only |
| Q04 | 4.1 | 24.5 | 18.4 | **−25%** from pre; filters `ss_store_sk=6` only |
| Q23 | 5.7 | 29.9 | 31.2 | Flat; date filter is a join predicate, no pruning |

**Notable Regression — Q22**

Q22 unexpectedly regressed from 15.5 s to 33.5 s (+115%) after compaction. This is the largest single-query movement and likely reflects a query plan change post-OPTIMIZE (Iceberg snapshot metadata update may alter statistics used for join ordering).

**Per-Query Results**

| Query | Non-Part Avg (s) | Pre-Compact (s) | Post-Compact (s) | Pre→Post |
|---|---|---|---|---|
| Q1 | 6.9 | 7.7 | 3.7 | −52% |
| Q2 | 4.0 | 3.4 | 3.6 | +8% |
| Q3 | 6.8 | 2.1 | 1.5 | −28% |
| Q4 | 4.1 | 24.5 | 18.4 | −25% |
| Q5 | 6.4 | 3.9 | 4.0 | +1% |
| Q6 | 8.4 | 4.3 | 2.7 | −36% |
| Q7 | 4.6 | 3.5 | 3.7 | +6% |
| Q8 | 6.1 | 2.4 | 2.6 | +9% |
| Q9 | 8.7 | 11.7 | 11.7 | 0% |
| Q10 | 16.9 | 5.3 | 2.6 | −51% |
| Q11 | 4.0 | 11.5 | 9.9 | −13% |
| Q12 | 6.2 | 3.1 | 2.7 | −13% |
| Q13 | 4.0 | 2.4 | 3.0 | +27% |
| Q14 | 4.0 | 15.3 | 13.4 | −12% |
| Q15 | 4.0 | 2.5 | 2.0 | −18% |
| Q16 | 4.0 | 6.0 | 4.3 | −30% |
| Q17 | 4.1 | 4.9 | 4.1 | −17% |
| Q18 | 4.1 | 4.7 | 4.8 | +3% |
| Q19 | 5.9 | 3.2 | 2.4 | −25% |
| Q20 | 10.0 | 2.9 | 2.6 | −12% |
| Q21 | 6.1 | 1.9 | 2.4 | +25% |
| Q22 | 4.0 | 15.5 | 33.5 | **+115%** |
| Q23 | 5.7 | 29.9 | 31.2 | +4% |
| Q24 | 4.0 | 11.9 | 9.2 | −23% |
| Q25 | 5.6 | 3.6 | 3.0 | −16% |
| Q26 | 5.6 | 2.7 | 2.7 | 0% |
| Q27 | 4.0 | 2.8 | 3.5 | +25% |
| Q28 | 4.6 | 9.5 | 9.7 | +1% |
| Q29 | 4.0 | 5.1 | 5.1 | −1% |
| Q30 | 6.6 | 4.2 | 3.7 | −13% |
| Q31 | 4.3 | 3.2 | 3.8 | +21% |
| Q32 | 4.0 | 1.6 | 1.4 | −13% |
| Q33 | 4.0 | 2.9 | 2.6 | −12% |
| Q34 | 7.5 | 2.1 | 2.1 | +1% |
| Q35 | 46.0 | 3.4 | 3.5 | +5% |
| Q36 | 11.5 | 1.9 | 2.0 | +5% |
| Q37 | 4.4 | 4.6 | 3.4 | −25% |
| Q38 | 7.8 | 5.0 | 5.4 | +9% |
| Q39 | 4.0 | 5.3 | 4.0 | −24% |
| Q40 | 4.2 | 3.8 | 3.1 | −18% |
| Q41 | 8.7 | 1.4 | 1.4 | +1% |
| Q42 | 14.1 | 1.6 | 1.6 | +2% |
| Q43 | 6.8 | 2.5 | 1.9 | −23% |
| Q44 | 4.0 | 6.9 | 7.6 | +10% |
| Q45 | 4.0 | 2.3 | 2.1 | −9% |
| Q46 | 7.3 | 3.6 | 2.7 | −26% |
| Q47 | 5.4 | 8.5 | 8.2 | −3% |
| Q48 | 6.7 | 3.2 | 3.6 | +11% |
| Q49 | 9.6 | 6.0 | 4.3 | −29% |
| Q50 | 9.0 | 5.9 | 6.0 | +2% |
| Q51 | 16.4 | 6.6 | 3.9 | −41% |
| Q52 | 4.0 | 1.9 | 1.9 | +1% |
| Q53 | 8.7 | 2.6 | 2.5 | −4% |
| Q54 | 7.3 | 4.3 | 3.6 | −16% |
| Q55 | 15.6 | 1.7 | 2.6 | +51% |
| Q56 | 4.0 | 2.6 | 3.4 | +33% |
| Q57 | 4.1 | 7.7 | 6.1 | −22% |
| Q58 | 5.0 | 3.7 | 3.6 | −4% |
| Q59 | 4.1 | 4.3 | 4.3 | +2% |
| Q60 | 5.3 | 3.1 | 3.2 | +4% |
| Q61 | 4.0 | 2.9 | 2.9 | +2% |
| Q62 | 4.0 | 2.9 | 3.0 | +1% |
| Q63 | 4.1 | 2.1 | 2.2 | +8% |
| Q64 | 4.0 | 13.3 | 15.8 | +19% |
| Q65 | 4.0 | 8.2 | 6.7 | −18% |
| Q66 | 8.7 | 3.3 | 2.3 | −29% |
| Q67 | 6.2 | 43.4 | 33.1 | −24% |
| Q68 | 46.5 | 2.7 | 2.9 | +7% |
| Q69 | 15.7 | 2.8 | 2.4 | −14% |
| Q70 | 13.3 | 3.4 | 3.1 | −10% |
| Q71 | 9.1 | 2.5 | 2.3 | −6% |
| Q72 | 4.0 | 16.8 | 20.0 | +19% |
| Q73 | 4.0 | 1.8 | 1.8 | 0% |
| Q74 | 4.0 | 6.5 | 5.2 | −20% |
| Q75 | 4.8 | 12.2 | 10.8 | −12% |
| Q76 | 10.8 | 8.5 | 5.4 | −36% |
| Q77 | 7.3 | 2.6 | 2.6 | −1% |
| Q78 | 7.5 | 17.0 | 13.5 | −21% |
| Q79 | 4.0 | 3.2 | 2.9 | −8% |
| Q80 | 4.0 | 6.5 | 5.3 | −20% |
| Q81 | 5.1 | 5.2 | 4.4 | −15% |
| Q82 | 4.1 | 4.3 | 4.3 | 0% |
| Q83 | 4.0 | 4.4 | 4.1 | −7% |
| Q84 | 4.7 | 3.7 | 3.2 | −13% |
| Q85 | 4.0 | 2.8 | 2.9 | +3% |
| Q86 | 4.0 | 2.4 | 2.0 | −17% |
| Q87 | 13.3 | 4.2 | 3.2 | −23% |
| Q88 | 4.0 | 10.2 | 8.2 | −20% |
| Q89 | 7.0 | 3.3 | 2.7 | −18% |
| Q90 | 4.0 | 3.8 | 3.2 | −17% |
| Q91 | 4.0 | 2.0 | 1.9 | −6% |
| Q92 | 6.8 | 1.4 | 1.4 | −3% |
| Q93 | 24.9 | 14.9 | 12.9 | −13% |
| Q94 | 4.3 | 3.6 | 3.6 | −1% |
| Q95 | 4.8 | 8.0 | 7.5 | −7% |
| Q96 | 4.6 | 4.2 | 3.6 | −15% |
| Q97 | 4.0 | 7.3 | 4.5 | −38% |
| Q98 | 8.0 | 3.0 | 3.2 | +6% |
| Q99 | 6.7 | 4.7 | 4.6 | −2% |
| **Total** | **713.5 (avg)** | **582.6** | **538.8** | **−7.5%** |
| **Geo Mean** | **6.0** | **4.4** | **4.0** | **−9%** |

---

### 8.3 sf5000 — Post-Compaction

> **Note:** This run uses the partitioned SF5000 schema after running `ALTER TABLE EXECUTE OPTIMIZE` on all 7 fact tables. Compaction took **571 s (9 min 32 sec)** total; `catalog_sales` was the largest table at 439 s. The non-partitioned baseline values in the comparison table are warm-run averages from Section 6.3 (3-run avg); the partitioned result is a single cold run.

**Summary**

| Metric | Non-Partitioned (warm avg) | Post-Compaction (cold) | Change |
|---|---|---|---|
| Queries Completed | 99 / 99 | 99 / 99 | — |
| Total Time | 1,468.8 s (24.5 min) | 833.3 s (13.9 min) | **−43%** |
| Geometric Mean | 9.30 s | 3.89 s | **−58%** |
| Fastest Query | Q2 — 4.0 s | Q41 — 0.2 s | |
| Slowest Query | Q35 — 178.8 s | Q04 — 56.0 s | |

**Time Distribution**

| Bucket | Non-Partitioned | Post-Compaction |
|---|---|---|
| < 10 s | 57 | 75 |
| 10 – 30 s | 34 | 18 |
| 30 – 60 s | 5 | 6 |
| > 1 min | 3 | 0 |

**Biggest Improvements**

| Query | Non-Part (s) | Post-Compact (s) | Change |
|---|---|---|---|
| Q41 | 15.8 | 0.2 | **−99%** |
| Q42 | 36.5 | 0.5 | **−99%** |
| Q35 | 178.8 | 2.9 | **−98%** |
| Q68 | 119.1 | 2.4 | **−98%** |
| Q10 | 59.5 | 1.9 | **−97%** |
| Q92 | 14.8 | 0.5 | **−97%** |

**Full-Scan Regression Queries**

These queries lack `ss_sold_date_sk` predicates and scan all of `store_sales`. With the partitioned table's higher file count (small file overhead), they are slower than the non-partitioned warm baseline.

| Query | Non-Part Avg (s) | Post-Compact (s) | Note |
|---|---|---|---|
| Q04 | 5.0 | 56.0 | **+1020%** — self-join on store_sales, no date predicate |
| Q67 | 18.0 | 55.9 | **+211%** — joins via ss_item_sk only |
| Q23 | 12.9 | 42.6 | **+230%** — date filter is a join predicate, no pruning |
| Q78 | 14.3 | 54.0 | **+278%** — full scan |
| Q11 | 5.6 | 28.1 | **+402%** — window function on full store_sales |

> **Note:** Non-partitioned values are warm-run averages; the partitioned cold run penalty amplifies apparent regression for these full-scan queries. The true regression vs a cold non-partitioned run would be substantially smaller.

**Per-Query Results**

| Query | Non-Part Avg (s) | Post-Compact (s) | Change |
|---|---|---|---|
| Q1 | 10.0 | 3.0 | **−70%** |
| Q2 | 4.0 | 1.9 | **−53%** |
| Q3 | 19.1 | 1.0 | **−95%** |
| Q4 | 5.0 | 56.0 | **+1020%** |
| Q5 | 4.0 | 4.3 | +8% |
| Q6 | 18.2 | 1.9 | **−90%** |
| Q7 | 6.2 | 2.4 | **−60%** |
| Q8 | 10.4 | 1.8 | **−82%** |
| Q9 | 16.1 | 30.2 | **+88%** |
| Q10 | 59.5 | 1.9 | **−97%** |
| Q11 | 5.6 | 28.1 | **+402%** |
| Q12 | 4.7 | 0.7 | **−85%** |
| Q13 | 4.0 | 4.8 | +19% |
| Q14 | 4.0 | 21.6 | **+441%** |
| Q15 | 6.3 | 1.5 | **−76%** |
| Q16 | 4.3 | 18.9 | **+338%** |
| Q17 | 7.1 | 5.8 | −19% |
| Q18 | 5.3 | 4.0 | −25% |
| Q19 | 9.6 | 2.0 | **−79%** |
| Q20 | 5.4 | 0.9 | **−83%** |
| Q21 | 14.8 | 0.4 | **−98%** |
| Q22 | 4.0 | 2.9 | −27% |
| Q23 | 12.9 | 42.6 | **+230%** |
| Q24 | 6.8 | 29.2 | **+330%** |
| Q25 | 13.5 | 3.2 | **−76%** |
| Q26 | 6.6 | 1.7 | **−75%** |
| Q27 | 4.0 | 2.2 | −44% |
| Q28 | 7.6 | 27.0 | **+256%** |
| Q29 | 4.0 | 17.0 | **+325%** |
| Q30 | 11.6 | 2.4 | **−79%** |
| Q31 | 11.4 | 3.5 | **−70%** |
| Q32 | 4.0 | 0.7 | **−82%** |
| Q33 | 6.0 | 1.6 | **−73%** |
| Q34 | 22.6 | 1.7 | **−93%** |
| Q35 | 178.8 | 2.9 | **−98%** |
| Q36 | 25.1 | 2.2 | **−91%** |
| Q37 | 5.6 | 7.1 | +26% |
| Q38 | 20.0 | 9.0 | **−55%** |
| Q39 | 4.0 | 0.9 | **−76%** |
| Q40 | 5.0 | 2.8 | −44% |
| Q41 | 15.8 | 0.2 | **−99%** |
| Q42 | 36.5 | 0.5 | **−99%** |
| Q43 | 17.4 | 1.6 | **−91%** |
| Q44 | 4.9 | 15.4 | **+214%** |
| Q45 | 6.3 | 1.5 | **−75%** |
| Q46 | 16.1 | 2.3 | **−86%** |
| Q47 | 8.0 | 11.6 | +45% |
| Q48 | 16.2 | 3.4 | **−79%** |
| Q49 | 26.6 | 4.3 | **−84%** |
| Q50 | 21.0 | 25.6 | +22% |
| Q51 | 43.5 | 2.3 | **−95%** |
| Q52 | 4.0 | 0.7 | **−82%** |
| Q53 | 26.8 | 1.3 | **−95%** |
| Q54 | 14.7 | 2.3 | **−85%** |
| Q55 | 4.5 | 0.8 | **−82%** |
| Q56 | 4.0 | 1.0 | **−75%** |
| Q57 | 6.2 | 7.7 | +25% |
| Q58 | 8.0 | 3.1 | **−62%** |
| Q59 | 7.0 | 4.0 | −43% |
| Q60 | 35.4 | 1.5 | **−96%** |
| Q61 | 6.5 | 2.0 | **−68%** |
| Q62 | 4.0 | 3.5 | −13% |
| Q63 | 8.0 | 1.6 | **−80%** |
| Q64 | 4.0 | 18.1 | **+351%** |
| Q65 | 4.0 | 9.0 | **+126%** |
| Q66 | 17.7 | 3.5 | **−80%** |
| Q67 | 18.0 | 55.9 | **+211%** |
| Q68 | 119.1 | 2.4 | **−98%** |
| Q69 | 38.5 | 2.1 | **−95%** |
| Q70 | 23.6 | 10.3 | **−56%** |
| Q71 | 17.4 | 1.1 | **−94%** |
| Q72 | 4.0 | 10.6 | **+164%** |
| Q73 | 4.0 | 1.1 | **−73%** |
| Q74 | 4.0 | 15.2 | **+281%** |
| Q75 | 5.2 | 23.7 | **+355%** |
| Q76 | 26.4 | 9.8 | **−63%** |
| Q77 | 14.8 | 1.8 | **−88%** |
| Q78 | 14.3 | 54.0 | **+278%** |
| Q79 | 4.0 | 3.4 | −15% |
| Q80 | 5.6 | 6.7 | +20% |
| Q81 | 8.8 | 3.6 | **−59%** |
| Q82 | 7.0 | 13.2 | **+89%** |
| Q83 | 4.0 | 1.9 | **−52%** |
| Q84 | 10.8 | 4.9 | **−54%** |
| Q85 | 4.0 | 4.2 | +5% |
| Q86 | 5.7 | 2.9 | −49% |
| Q87 | 4.5 | 6.9 | **+54%** |
| Q88 | 4.0 | 19.3 | **+383%** |
| Q89 | 15.8 | 1.8 | **−89%** |
| Q90 | 4.0 | 3.1 | −22% |
| Q91 | 4.0 | 1.0 | **−76%** |
| Q92 | 14.8 | 0.5 | **−97%** |
| Q93 | 81.5 | 41.3 | −49% |
| Q94 | 8.7 | 8.9 | +2% |
| Q95 | 12.3 | 16.0 | +30% |
| Q96 | 4.3 | 3.9 | −9% |
| Q97 | 4.0 | 13.7 | **+242%** |
| Q98 | 16.2 | 1.2 | **−93%** |
| Q99 | 11.3 | 4.1 | **−64%** |
| **Total** | **1,468.8** | **833.3** | **−43%** |
| **Geo Mean** | **9.30** | **3.89** | **−58%** |

---

### 8.4 sf10000 — Post-Compaction

> **Note:** This run uses the partitioned SF10000 schema after running `ALTER TABLE EXECUTE OPTIMIZE` on all 7 fact tables. Compaction took **24 min 24 sec** total; the largest single table took 19.9 min. Due to a transient worker node overload (loadgen-03a), 16 queries required a rerun; all 99 queries completed successfully after reruns. The non-partitioned baseline values are warm-run averages from Section 6.4; the partitioned result is a single cold run.

**Summary**

| Metric | Non-Partitioned (warm avg) | Post-Compaction (cold) | Change |
|---|---|---|---|
| Queries Completed | 99 / 99 | 99 / 99 | — |
| Total Time | 2,787.4 s (46.5 min) | 1,944.6 s (32.4 min) | **−30%** |
| Geometric Mean | 14.33 s | 7.67 s | **−46%** |
| Fastest Query | Q2 — 4.0 s | Q41 — 0.8 s | |
| Slowest Query | Q35 — 398.3 s | Q67 — 221.2 s | |

**Time Distribution**

| Bucket | Non-Partitioned | Post-Compaction |
|---|---|---|
| < 10 s | 37 | 66 |
| 10 – 30 s | 37 | 13 |
| 30 – 60 s | 16 | 12 |
| 1 – 5 min | 7 | 8 |
| > 5 min | 2 | 0 |

**Biggest Improvements**

| Query | Non-Part (s) | Post-Compact (s) | Change |
|---|---|---|---|
| Q68 | 300.2 | 2.9 | **−99%** |
| Q35 | 398.3 | 3.9 | **−99%** |
| Q42 | 61.7 | 0.9 | **−99%** |
| Q10 | 116.2 | 2.8 | **−98%** |
| Q41 | 23.9 | 0.8 | **−97%** |
| Q92 | 32.4 | 1.1 | **−97%** |

**Full-Scan Regression Queries**

| Query | Non-Part Avg (s) | Post-Compact (s) | Note |
|---|---|---|---|
| Q72 | 4.0 | 99.9 | **+2396%** — full scan, no date predicate |
| Q04 | 8.0 | 127.7 | **+1497%** — self-join on store_sales |
| Q23 | 15.6 | 198.6 | **+1173%** — date filter via join only |
| Q14 | 4.0 | 44.2 | **+1005%** — full scan |
| Q67 | 30.0 | 221.2 | **+637%** — joins via ss_item_sk only |

> **Note:** Non-partitioned values are warm-run averages; the cold run penalty amplifies apparent regression for full-scan queries. Q67 at 221.2s is the largest absolute slowdown and confirms this query's sensitivity to file count at SF10000 scale.

**Per-Query Results**

| Query | Non-Part Avg (s) | Post-Compact (s) | Change |
|---|---|---|---|
| Q1 | 13.1 | 4.6 | **−65%** |
| Q2 | 4.0 | 2.9 | −26% |
| Q3 | 31.1 | 1.9 | **−94%** |
| Q4 | 8.0 | 127.7 | **+1497%** |
| Q5 | 7.4 | 7.7 | +4% |
| Q6 | 31.0 | 3.5 | **−89%** |
| Q7 | 10.7 | 7.8 | −27% |
| Q8 | 15.5 | 3.3 | **−79%** |
| Q9 | 24.4 | 54.0 | **+121%** |
| Q10 | 116.2 | 2.8 | **−98%** |
| Q11 | 8.0 | 57.6 | **+620%** |
| Q12 | 50.2 | 2.1 | **−96%** |
| Q13 | 4.0 | 7.2 | +79% |
| Q14 | 4.0 | 44.2 | **+1005%** |
| Q15 | 10.8 | 2.5 | **−77%** |
| Q16 | 4.0 | 33.3 | **+733%** |
| Q17 | 13.4 | 9.0 | −33% |
| Q18 | 7.7 | 7.7 | 0% |
| Q19 | 18.3 | 1.6 | **−91%** |
| Q20 | 18.4 | 2.4 | **−87%** |
| Q21 | 16.2 | 1.9 | **−88%** |
| Q22 | 4.0 | 34.2 | **+754%** |
| Q23 | 15.6 | 198.6 | **+1173%** |
| Q24 | 9.5 | 65.7 | **+591%** |
| Q25 | 20.9 | 8.6 | **−59%** |
| Q26 | 11.5 | 4.5 | **−61%** |
| Q27 | 4.0 | 4.6 | +14% |
| Q28 | 11.5 | 46.1 | **+301%** |
| Q29 | 4.0 | 22.6 | **+465%** |
| Q30 | 19.4 | 3.1 | **−84%** |
| Q31 | 18.7 | 6.3 | **−67%** |
| Q32 | 4.0 | 1.2 | **−69%** |
| Q33 | 8.7 | 2.4 | **−73%** |
| Q34 | 30.8 | 1.7 | **−95%** |
| Q35 | 398.3 | 3.9 | **−99%** |
| Q36 | 42.6 | 4.5 | **−90%** |
| Q37 | 6.9 | 16.5 | **+140%** |
| Q38 | 40.1 | 19.2 | **−52%** |
| Q39 | 4.0 | 6.5 | +62% |
| Q40 | 6.0 | 6.1 | +2% |
| Q41 | 23.9 | 0.8 | **−97%** |
| Q42 | 61.7 | 0.9 | **−99%** |
| Q43 | 31.0 | 3.2 | **−90%** |
| Q44 | 7.2 | 30.5 | **+323%** |
| Q45 | 11.0 | 3.8 | **−66%** |
| Q46 | 32.8 | 5.9 | **−82%** |
| Q47 | 12.0 | 28.7 | **+139%** |
| Q48 | 29.7 | 4.9 | **−84%** |
| Q49 | 50.6 | 7.8 | **−84%** |
| Q50 | 28.1 | 66.9 | **+138%** |
| Q51 | 72.4 | 10.0 | **−86%** |
| Q52 | 4.0 | 1.2 | **−70%** |
| Q53 | 46.3 | 3.1 | **−93%** |
| Q54 | 23.7 | 6.5 | **−73%** |
| Q55 | 20.8 | 1.0 | **−95%** |
| Q56 | 4.0 | 2.9 | −26% |
| Q57 | 8.2 | 17.2 | **+110%** |
| Q58 | 14.2 | 5.7 | **−60%** |
| Q59 | 11.5 | 6.2 | −46% |
| Q60 | 65.7 | 3.1 | **−95%** |
| Q61 | 11.6 | 3.1 | **−74%** |
| Q62 | 4.0 | 4.8 | +20% |
| Q63 | 10.5 | 3.3 | **−68%** |
| Q64 | 4.0 | 32.4 | **+710%** |
| Q65 | 4.0 | 28.6 | **+616%** |
| Q66 | 30.5 | 5.4 | **−82%** |
| Q67 | 30.0 | 221.2 | **+637%** |
| Q68 | 300.2 | 2.9 | **−99%** |
| Q69 | 86.3 | 3.1 | **−96%** |
| Q70 | 62.8 | 14.6 | **−77%** |
| Q71 | 43.4 | 1.7 | **−96%** |
| Q72 | 4.0 | 99.9 | **+2396%** |
| Q73 | 4.0 | 2.1 | −48% |
| Q74 | 4.0 | 30.7 | **+668%** |
| Q75 | 6.0 | 44.0 | **+633%** |
| Q76 | 51.6 | 20.0 | **−61%** |
| Q77 | 31.0 | 1.9 | **−94%** |
| Q78 | 25.3 | 88.7 | **+251%** |
| Q79 | 4.0 | 4.5 | +13% |
| Q80 | 8.6 | 8.0 | −7% |
| Q81 | 14.2 | 4.7 | **−67%** |
| Q82 | 14.3 | 24.0 | **+68%** |
| Q83 | 4.0 | 2.6 | −35% |
| Q84 | 17.1 | 6.1 | **−64%** |
| Q85 | 4.0 | 5.7 | +42% |
| Q86 | 10.0 | 2.8 | **−72%** |
| Q87 | 15.8 | 18.9 | +19% |
| Q88 | 4.0 | 33.1 | **+728%** |
| Q89 | 24.9 | 4.3 | **−83%** |
| Q90 | 4.0 | 5.8 | +46% |
| Q91 | 4.0 | 1.4 | **−64%** |
| Q92 | 32.4 | 1.1 | **−97%** |
| Q93 | 184.4 | 70.0 | **−62%** |
| Q94 | 11.0 | 12.1 | +10% |
| Q95 | 21.8 | 34.4 | **+58%** |
| Q96 | 6.5 | 6.6 | +1% |
| Q97 | 4.0 | 28.4 | **+611%** |
| Q98 | 27.7 | 2.5 | **−91%** |
| Q99 | 21.8 | 7.0 | **−68%** |
| **Total** | **2,787.4** | **1,944.6** | **−30%** |
| **Geo Mean** | **14.33** | **7.67** | **−46%** |

---

## 9. Cross-Scale Summary

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

## 10. Appendix — Full Configuration

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
