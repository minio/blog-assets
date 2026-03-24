# TPC-DS Scale3000 & Scale5000 Data Generation Runbook
Date: 2026-03-16
Server: sds01

## Storage Layout
- Scale3000 raw:     /mnt/drive2/tpcdsraw/scale3000
- Scale3000 parquet: /mnt/drive2/tpcdsparquet/scale3000
- Scale5000 raw:     /mnt/drive3/tpcdsraw/scale5000
- Scale5000 parquet: /mnt/drive3/tpcdsparquet/scale5000

---

## Step 1 — Create Directories
```bash
mkdir -p /mnt/drive2/tpcdsraw/scale3000
mkdir -p /mnt/drive2/tpcdsparquet/scale3000
mkdir -p /mnt/drive3/tpcdsraw/scale5000
mkdir -p /mnt/drive3/tpcdsparquet/scale5000
```
Status: DONE

---

## Step 2 — Generate Data (both in parallel)
```bash
nohup /opt/tpcds/dsdgen -scale 3000 -dir /mnt/drive2/tpcdsraw/scale3000 -terminate n > /tmp/dsdgen_3tb.log 2>&1 &
nohup /opt/tpcds/dsdgen -scale 5000 -dir /mnt/drive3/tpcdsraw/scale5000 -terminate n > /tmp/dsdgen_5tb.log 2>&1 &
```
Monitor:
```bash
watch -n 30 'du -sh /mnt/drive2/tpcdsraw/scale3000/ /mnt/drive3/tpcdsraw/scale5000/'
tail -f /tmp/dsdgen_3tb.log
tail -f /tmp/dsdgen_5tb.log
```
Status: DONE (scale3000: 2.7TB, scale5000: 4.5TB completed 2026-03-17)

---

## Step 3 — Convert Scale3000 to Parquet (after generation complete)
```bash
ln -sfn /mnt/drive2/tpcdsraw/scale3000 /tpcds/perf_testing/tpcds-kit/test_data/raw_files
ln -sfn /mnt/drive2/tpcdsparquet/scale3000 /tpcds/perf_testing/tpcds-kit/test_data/parquet
nohup python3 /tpcds/perf_testing/tpcds-kit/parquet_transform_spark.py --custom_dir /mnt/drive4 > /tmp/spark_convert_3tb.log 2>&1 &
```
Monitor:
```bash
watch -n 30 'du -sh /mnt/drive2/tpcdsparquet/scale3000/'
tail -f /tmp/spark_convert_3tb.log
```
Status: IN PROGRESS

---

## Step 4 — Upload Scale3000 to MinIO
```bash
nohup mc cp --recursive /mnt/drive2/tpcdsparquet/scale3000/ myminio/tpcds/scale3000/ > /tmp/mc_upload_3tb.log 2>&1 &
```
Monitor:
```bash
tail -f /tmp/mc_upload_3tb.log
```
Status: DONE (1.2TiB in 4.2min at 5GB/s, completed 2026-03-17 16:05)

---

## Step 5 — Convert Scale5000 to Parquet on sds02 (parallel with scale3000)

### Setup sds02 (one-time)
```bash
# On sds01:
ssh root@192.168.11.2 "apt install -y default-jdk && pip3 install pyspark termcolor python-dotenv --break-system-packages"
rsync -a /tpcds/perf_testing/tpcds-kit/ root@192.168.11.2:/tpcds/perf_testing/tpcds-kit/ --exclude test_data
```

### Copy scale5000 raw data to sds02 via 400Gbps interface
```bash
nohup rsync -a --progress /mnt/drive3/tpcdsraw/scale5000/ root@192.168.11.2:/mnt/drive1/tpcdsraw/scale5000/ > /tmp/rsync_5tb.log 2>&1 &
tail -f /tmp/rsync_5tb.log
```

### Run conversion on sds02
```bash
ssh root@192.168.11.2
ln -sfn /mnt/drive1/tpcdsraw/scale5000 /tpcds/perf_testing/tpcds-kit/test_data/raw_files
ln -sfn /mnt/drive1/tpcdsparquet/scale5000 /tpcds/perf_testing/tpcds-kit/test_data/parquet
nohup python3 /tpcds/perf_testing/tpcds-kit/parquet_transform_spark.py --custom_dir /mnt/drive2/tpcds_tmp > /tmp/spark_convert_5tb.log 2>&1 &
```
Monitor:
```bash
ssh root@192.168.11.2 'watch -n 30 "du -sh /mnt/drive1/tpcdsparquet/scale5000/"'
ssh root@192.168.11.2 'tail -f /tmp/spark_convert_5tb.log'
```
Status: CANCELLED - ran conversion on sds01 directly (data already local on drive3)

---

## Step 5b — Convert Scale5000 on sds01 (actual approach)
```bash
ln -sfn /mnt/drive3/tpcdsraw/scale5000 /tpcds/perf_testing/tpcds-kit/test_data/raw_files
ln -sfn /mnt/drive3/tpcdsparquet/scale5000 /tpcds/perf_testing/tpcds-kit/test_data/parquet
nohup python3 /tpcds/perf_testing/tpcds-kit/parquet_transform_spark.py --custom_dir /mnt/drive4 > /tmp/spark_convert_5tb.log 2>&1 &
```
Monitor:
```bash
watch -n 30 'du -sh /mnt/drive3/tpcdsparquet/scale5000/'
tail -f /tmp/spark_convert_5tb.log
```
Status: DONE (2.1TB, 25 tables, completed 2026-03-17 17:23)

---

## Step 6 — Upload Scale5000 to MinIO
```bash
nohup mc cp --recursive /mnt/drive3/tpcdsparquet/scale5000/ myminio/tpcds/scale5000/ > /tmp/mc_upload_5tb.log 2>&1 &
```
Status: DONE (2.0TiB in 7min at 4.9GB/s, completed 2026-03-17 17:34)

---

## Notes
- dsdgen binary: /opt/tpcds/dsdgen
- Spark memory config: driver=180g, executor=180g (in parquet_transform_spark.py)
- MinIO alias: myminio → https://192.168.11.1:9000
- Convert one scale at a time due to memory constraints
- Use /mnt/drive4 as Spark temp dir (--custom_dir)
