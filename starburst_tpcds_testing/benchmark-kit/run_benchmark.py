#!/usr/bin/env python3
"""
run_benchmark.py
================
TPC-DS Benchmark Runner — Starburst on MinIO AIStor

Runs all 99 TPC-DS queries 3 times against aistor.<schema>.

  Run 1 : Cold  — OS page cache flushed + Starburst restarted
  Run 2 : Warm
  Run 3 : Warm

Results are written to RESULTS_DIR/raw.csv after each query completes.
The script is crash-safe: on restart it skips queries already in raw.csv.

Usage:
    python3 run_benchmark.py --scale sf10000
    python3 run_benchmark.py --scale sf10000 --skip-cold-prep   # resume after crash
"""

import os
import sys
import csv
import time
import json
import math
import re
import subprocess
import argparse
from datetime import datetime

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COORDINATOR   = "http://loadgen-09a:8080"
CATALOG       = "aistor"

QUERIES_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queries")
RESULTS_BASE  = "/root/benchmark_results"

RUNS          = 3
QUERY_TIMEOUT = 3600    # 60 min per query per run — queries exceeding this are cancelled
POLL_INTERVAL = 1       # seconds between REST API polls
STABILIZE_WAIT = 30     # seconds to wait after all workers are active

LOADGEN_NODES = [f"loadgen-0{i}a" for i in range(1, 10)]   # loadgen-01a..loadgen-09a
SDS_NODES     = [f"sds-0{i}a"     for i in range(1, 9)]    # sds-01a..sds-08a
EXPECTED_WORKERS = 9

CSV_HEADERS = ["query", "run1_s", "run2_s", "run3_s", "avg_s", "status", "notes"]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log_file = None

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        with open(_log_file, "a") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Cluster management
# ---------------------------------------------------------------------------

def ssh(node, cmd, timeout=30):
    """Run a command on a remote node. Returns (rc, stdout, stderr)."""
    result = subprocess.run(
        ["ssh",
         "-o", "ConnectTimeout=10",
         "-o", "StrictHostKeyChecking=no",
         "-o", "BatchMode=yes",
         f"root@{node}", cmd],
        capture_output=True, text=True, timeout=timeout
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def drop_page_caches():
    log("=" * 60)
    log("Step 1/3 — Dropping OS page cache on all nodes")
    log("=" * 60)
    all_nodes = SDS_NODES + LOADGEN_NODES
    for node in all_nodes:
        try:
            rc, _, err = ssh(node, "sync; echo 3 > /proc/sys/vm/drop_caches")
            log(f"  {node}: {'OK' if rc == 0 else 'FAILED — ' + err}")
        except Exception as e:
            log(f"  {node}: ERROR — {e}")


def restart_starburst():
    log("=" * 60)
    log("Step 2/3 — Restarting Starburst on all 9 nodes")
    log("=" * 60)
    result = subprocess.run(
        ["bash", "-c",
         "source /root/starburst/starburst_hosts && "
         "clush -w \"${ALL_NODES}\" sudo systemctl restart starburst"],
        capture_output=True, text=True, timeout=120
    )
    log(f"  Exit code: {result.returncode}")
    if result.stdout:
        log(f"  stdout: {result.stdout[:300]}")
    if result.stderr:
        log(f"  stderr: {result.stderr[:300]}")


def wait_for_cluster():
    log("=" * 60)
    log(f"Step 3/3 — Waiting for {EXPECTED_WORKERS} workers to be ACTIVE")
    log("=" * 60)
    log("  Sleeping 30s for Starburst to start accepting connections...")
    time.sleep(30)

    deadline = time.time() + 300    # wait up to 5 min
    while time.time() < deadline:
        try:
            count = get_active_node_count()
            log(f"  Active nodes: {count}/{EXPECTED_WORKERS}")
            if count >= EXPECTED_WORKERS:
                log(f"  All workers active. Stabilizing for {STABILIZE_WAIT}s...")
                time.sleep(STABILIZE_WAIT)
                log("  Cluster ready.")
                return
        except Exception as e:
            log(f"  Not ready yet: {e}")
        time.sleep(10)

    log("  WARNING: Timed out waiting for all workers. Proceeding anyway.")


def get_active_node_count():
    """Query system.runtime.nodes to count active workers."""
    sql = "SELECT count(*) FROM system.runtime.nodes WHERE state = 'active'"
    resp = requests.post(
        f"{COORDINATOR}/v1/statement",
        headers={"X-Trino-User": "admin", "Content-Type": "text/plain"},
        data=sql,
        timeout=15
    )
    resp.raise_for_status()
    result = _poll(resp.json(), timeout=60)
    rows = result.get("rows", [])
    if rows:
        return int(rows[0][0])
    return 0


# ---------------------------------------------------------------------------
# REST API — query execution
# ---------------------------------------------------------------------------

def _poll(initial, timeout=QUERY_TIMEOUT):
    """
    Poll nextUri until the query finishes (or times out / errors).
    Accumulates all result rows across pages.
    Returns dict: {state, query_id, rows, rows_processed, error}
    """
    response    = initial
    next_uri    = response.get("nextUri", "")
    query_id    = response.get("id", "unknown")
    all_rows    = list(response.get("data", []))
    state       = response.get("stats", {}).get("state", "UNKNOWN")
    rows_proc   = 0
    deadline    = time.time() + timeout

    while next_uri:
        if time.time() > deadline:
            try:
                requests.delete(next_uri,
                                headers={"X-Trino-User": "admin"}, timeout=10)
            except Exception:
                pass
            return {"state": "TIMEOUT", "query_id": query_id,
                    "rows": all_rows, "rows_processed": rows_proc}

        time.sleep(POLL_INTERVAL)

        try:
            r = requests.get(next_uri,
                             headers={"X-Trino-User": "admin"}, timeout=30)
            r.raise_for_status()
            response = r.json()
        except Exception as e:
            return {"state": "ERROR", "error": str(e), "query_id": query_id,
                    "rows": all_rows, "rows_processed": rows_proc}

        if "error" in response:
            return {"state": "FAILED",
                    "error": response["error"].get("message", "unknown"),
                    "query_id": query_id,
                    "rows": all_rows, "rows_processed": rows_proc}

        next_uri   = response.get("nextUri", "")
        state      = response.get("stats", {}).get("state", "UNKNOWN")
        rows_proc  = response.get("stats", {}).get("processedRows", rows_proc)
        all_rows.extend(response.get("data", []))

    return {"state": state, "query_id": query_id,
            "rows": all_rows, "rows_processed": rows_proc}


def execute_query(query_label, sql, run_num, schema):
    """Submit one query and time it. Returns (elapsed_s, state, notes)."""
    log(f"    Run {run_num}: submitting...")
    t_start = time.time()

    try:
        resp = requests.post(
            f"{COORDINATOR}/v1/statement",
            headers={
                "X-Trino-User":    "admin",
                "X-Trino-Catalog": CATALOG,
                "X-Trino-Schema":  schema,
                "Content-Type":    "text/plain",
            },
            data=sql,
            timeout=30
        )
        resp.raise_for_status()
        initial = resp.json()

        if "error" in initial:
            elapsed = round(time.time() - t_start, 1)
            msg = initial["error"].get("message", "Submit error")[:120]
            log(f"    Run {run_num}: FAILED immediately — {msg}")
            return None, "FAILED", msg

        result  = _poll(initial)
        elapsed = round(time.time() - t_start, 1)
        state   = result["state"]

        if state == "FINISHED":
            log(f"    Run {run_num}: FINISHED in {elapsed}s "
                f"(rows_processed={result.get('rows_processed','?')})")
            return elapsed, "FINISHED", ""
        elif state == "TIMEOUT":
            log(f"    Run {run_num}: TIMEOUT after {elapsed}s")
            return None, "TIMEOUT", f"Exceeded {QUERY_TIMEOUT}s"
        else:
            err = result.get("error", "")[:120]
            log(f"    Run {run_num}: {state} — {err}")
            return None, state, err

    except Exception as e:
        elapsed = round(time.time() - t_start, 1)
        log(f"    Run {run_num}: EXCEPTION after {elapsed}s — {e}")
        return None, "ERROR", str(e)[:120]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_completed(results_csv):
    """Return set of query labels that FINISHED successfully in raw.csv.
    FAILED/TIMEOUT/ERROR queries are NOT skipped so they get retried."""
    done = set()
    if not os.path.exists(results_csv):
        return done
    with open(results_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "FINISHED":
                done.add(row["query"])
    return done


def write_result(results_csv, query, r1, r2, r3, status, notes):
    """Append one result row. Creates file with header if needed."""
    write_header = not os.path.exists(results_csv)
    with open(results_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(CSV_HEADERS)
        finished_runs = [r for r in [r1, r2, r3] if r is not None]
        avg = round(sum(finished_runs) / len(finished_runs), 1) if finished_runs else ""
        w.writerow([query,
                    r1 if r1 is not None else "",
                    r2 if r2 is not None else "",
                    r3 if r3 is not None else "",
                    avg, status, notes])


# ---------------------------------------------------------------------------
# Query file helpers
# ---------------------------------------------------------------------------

def load_queries():
    """Return list of (label, sql) sorted numerically by query number."""
    def num(name):
        m = re.search(r"(\d+)", name)
        return int(m.group(1)) if m else 0

    files = sorted(
        [f for f in os.listdir(QUERIES_DIR) if f.endswith(".sql")],
        key=num
    )
    result = []
    for fname in files:
        label = f"Q{num(fname)}"
        with open(os.path.join(QUERIES_DIR, fname), "r") as f:
            sql = f.read().strip().rstrip(";")
        result.append((label, sql, fname))
    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results_csv):
    log("\n" + "=" * 60)
    log("BENCHMARK SUMMARY")
    log("=" * 60)

    if not os.path.exists(results_csv):
        log("No results file found.")
        return

    rows = []
    with open(results_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    finished = [r for r in rows if r["status"] == "FINISHED" and r["avg_s"]]
    failed   = [r["query"] for r in rows if r["status"] != "FINISHED"]
    avgs     = [float(r["avg_s"]) for r in finished]

    total    = round(sum(avgs), 1)
    geo_mean = round(math.exp(sum(math.log(a) for a in avgs) / len(avgs)), 1) if avgs else 0

    log(f"  Queries completed : {len(finished)} / {len(rows)}")
    log(f"  Queries failed    : {len(failed)}  {failed}")
    log(f"  Total time        : {total}s  ({round(total/3600, 2)}h)")
    log(f"  Geometric mean    : {geo_mean}s")

    if finished:
        slowest = max(finished, key=lambda r: float(r["avg_s"]))
        fastest = min(finished, key=lambda r: float(r["avg_s"]))
        log(f"  Slowest query     : {slowest['query']} ({slowest['avg_s']}s)")
        log(f"  Fastest query     : {fastest['query']} ({fastest['avg_s']}s)")

    log(f"\n  Results saved to  : {results_csv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TPC-DS benchmark runner for Starburst")
    parser.add_argument("--scale", required=True,
                        choices=["sf100", "sf1000", "sf3000", "sf5000", "sf10000"],
                        help="Scale factor to benchmark")
    parser.add_argument("--skip-cold-prep", action="store_true",
                        help="Skip restart + cache flush (use when resuming after crash)")
    args = parser.parse_args()

    schema      = args.scale.replace("sf", "scale")   # sf10000 -> scale10000
    results_dir = os.path.join(RESULTS_BASE, args.scale)
    results_csv = os.path.join(results_dir, "raw.csv")
    log_path    = os.path.join(results_dir, "benchmark.log")

    os.makedirs(results_dir, exist_ok=True)

    global _log_file
    _log_file = log_path

    log("=" * 60)
    log(f"TPC-DS Benchmark — {args.scale}")
    log(f"Catalog  : {CATALOG}.{schema}")
    log(f"Runs     : {RUNS}  (Run 1 cold, Run 2-3 warm)")
    log(f"Timeout  : {QUERY_TIMEOUT}s per query per run")
    log(f"Results  : {results_csv}")
    log(f"Log      : {log_path}")
    log("=" * 60)

    # --- Cold start prep ---
    if not args.skip_cold_prep:
        drop_page_caches()
        restart_starburst()
        wait_for_cluster()
    else:
        log("Skipping cold prep (--skip-cold-prep)")

    # --- Load queries ---
    queries = load_queries()
    log(f"\nLoaded {len(queries)} query files from {QUERIES_DIR}")

    # --- Skip already-done queries (crash recovery) ---
    completed = load_completed(results_csv)
    if completed:
        log(f"Resuming: {len(completed)} queries already completed, skipping")

    total = len(queries)
    done  = len(completed)

    # --- Benchmark loop ---
    for idx, (label, sql, fname) in enumerate(queries, 1):
        if label in completed:
            continue

        log(f"\n{'─' * 60}")
        log(f"[{idx}/{total}] {label}  ({fname})")
        log(f"{'─' * 60}")

        r1 = r2 = r3 = None
        final_status = "FINISHED"
        final_notes  = ""

        for run_num in range(1, RUNS + 1):
            elapsed, state, notes = execute_query(label, sql, run_num, schema)

            if state == "FINISHED":
                if run_num == 1: r1 = elapsed
                if run_num == 2: r2 = elapsed
                if run_num == 3: r3 = elapsed
            else:
                final_status = state
                final_notes  = notes
                break   # skip remaining runs for this query

        write_result(results_csv, label, r1, r2, r3, final_status, final_notes)
        done += 1
        log(f"  Saved to CSV. Progress: {done}/{total}")

    print_summary(results_csv)


if __name__ == "__main__":
    main()
