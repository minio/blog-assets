#!/usr/bin/env python3
"""
TPC-DS JMeter Results Analyzer
Analyzes results from ~/results/ (or path passed as argument)
"""

import os
import sys
import csv
import glob
from datetime import datetime, timezone
from collections import defaultdict

RESULTS_DIR = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/results")

TEST_FILES = {
    "01-CREATE-TPCDS":   "01-create-tpcds-tree.csv",
    "02-INSERT-TPCDS":   "02-insert-tpcds-tree.csv",
    "03-OPTIMIZE-TPCDS": "03-optimize-tpcds-tree.csv",
    "04-TPCDS-Iceberg":  "04-tpcds-iceberg-tree.csv",
    "05-DROP-TPCDS":     "05-drop-tpcds-tree.csv",
}

SEP  = "=" * 70
SEP2 = "-" * 70


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["elapsed"] = int(row["elapsed"])
            row["timeStamp"] = int(row["timeStamp"])
            row["success"] = row["success"].lower() == "true"
            rows.append(row)
    return rows


def fmt_ms(ms):
    if ms >= 60_000:
        return f"{ms/60000:.1f}m"
    if ms >= 1_000:
        return f"{ms/1000:.2f}s"
    return f"{ms}ms"


def percentile(values, p):
    if not values:
        return 0
    s = sorted(values)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


def analyze_test(name, rows):
    total      = len(rows)
    errors     = [r for r in rows if not r["success"]]
    successes  = [r for r in rows if r["success"]]
    elapsed    = [r["elapsed"] for r in rows]
    start_ts   = min(r["timeStamp"] for r in rows)
    end_ts     = max(r["timeStamp"] + r["elapsed"] for r in rows)
    duration   = end_ts - start_ts

    print(f"\n{SEP}")
    print(f"  {name}")
    print(SEP)
    print(f"  Samples   : {total}  |  Errors: {len(errors)}  ({100*len(errors)/total:.1f}%)")
    print(f"  Duration  : {fmt_ms(duration)}")
    print(f"  Avg       : {fmt_ms(int(sum(elapsed)/len(elapsed)))}  |  "
          f"Min: {fmt_ms(min(elapsed))}  |  Max: {fmt_ms(max(elapsed))}")
    print(f"  p50       : {fmt_ms(percentile(elapsed,50))}  |  "
          f"p90: {fmt_ms(percentile(elapsed,90))}  |  "
          f"p99: {fmt_ms(percentile(elapsed,99))}")

    if errors:
        print(f"\n  ERRORS:")
        for r in errors:
            print(f"    [{r['label']}]  {r['responseMessage'][:80]}")

    return {
        "name": name, "total": total, "errors": len(errors),
        "duration": duration, "avg": sum(elapsed)/len(elapsed),
        "min": min(elapsed), "max": max(elapsed),
        "p90": percentile(elapsed, 90), "p99": percentile(elapsed, 99),
        "rows": rows,
    }


def analyze_queries(rows):
    """Detailed per-query breakdown for the TPC-DS benchmark test."""
    queries = [r for r in rows if r["label"].startswith("Q")]
    if not queries:
        return

    def qnum(r):
        return int("".join(filter(str.isdigit, r["label"])))

    queries_sorted = sorted(queries, key=lambda r: r["elapsed"], reverse=True)

    print(f"\n  {'Query':<10} {'Elapsed':>10}  {'Status'}")
    print(f"  {'-'*10} {'-'*10}  {'-'*6}")
    for r in sorted(queries, key=qnum):
        status = "OK" if r["success"] else "FAIL"
        print(f"  {r['label']:<10} {fmt_ms(r['elapsed']):>10}  {status}")

    print(f"\n  Top 10 slowest queries:")
    for r in queries_sorted[:10]:
        print(f"    {r['label']:<8} {fmt_ms(r['elapsed'])}")

    elapsed = [r["elapsed"] for r in queries]
    print(f"\n  Query stats  avg={fmt_ms(int(sum(elapsed)/len(elapsed)))}  "
          f"p50={fmt_ms(percentile(elapsed,50))}  "
          f"p90={fmt_ms(percentile(elapsed,90))}  "
          f"p99={fmt_ms(percentile(elapsed,99))}")


def analyze_inserts(rows):
    """Highlight slow inserts (partitioned fact tables)."""
    inserts = [r for r in rows if r["label"].startswith("INSERT INTO")]
    if not inserts:
        return
    slow = sorted(inserts, key=lambda r: r["elapsed"], reverse=True)[:8]
    print(f"\n  Top 8 slowest inserts:")
    for r in slow:
        table = r["label"].replace("INSERT INTO ", "")
        print(f"    {table:<35} {fmt_ms(r['elapsed'])}")


def suite_summary(results):
    print(f"\n{SEP}")
    print(f"  SUITE SUMMARY")
    print(SEP)
    total_dur = sum(r["duration"] for r in results)
    total_err = sum(r["errors"] for r in results)
    total_samp = sum(r["total"] for r in results)
    print(f"  {'Test':<22} {'Samples':>8} {'Errors':>7} {'Duration':>10} {'Avg':>10} {'p90':>10}")
    print(f"  {'-'*22} {'-'*8} {'-'*7} {'-'*10} {'-'*10} {'-'*10}")
    for r in results:
        err_str = f"{r['errors']} ({100*r['errors']/r['total']:.0f}%)" if r["errors"] else "0"
        print(f"  {r['name']:<22} {r['total']:>8} {err_str:>7} "
              f"{fmt_ms(r['duration']):>10} {fmt_ms(int(r['avg'])):>10} {fmt_ms(r['p90']):>10}")
    print(f"  {'-'*22} {'-'*8} {'-'*7} {'-'*10}")
    print(f"  {'TOTAL':<22} {total_samp:>8} {total_err:>7} {fmt_ms(total_dur):>10}")
    status = "PASS" if total_err == 0 else "FAIL"
    print(f"\n  Overall result: {status}")


def main():
    if not os.path.isdir(RESULTS_DIR):
        print(f"Results directory not found: {RESULTS_DIR}")
        sys.exit(1)

    print(f"\n{SEP}")
    print(f"  TPC-DS JMeter Results Analysis")
    print(f"  Directory : {RESULTS_DIR}")
    print(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEP)

    results = []
    for name, filename in TEST_FILES.items():
        path = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(path):
            print(f"\n  [SKIP] {name} — {filename} not found")
            continue
        rows = load_csv(path)
        stats = analyze_test(name, rows)
        results.append(stats)

        if name == "04-TPCDS-Iceberg":
            analyze_queries(rows)
        elif name == "02-INSERT-TPCDS":
            analyze_inserts(rows)

    if results:
        suite_summary(results)

    print()


if __name__ == "__main__":
    main()
