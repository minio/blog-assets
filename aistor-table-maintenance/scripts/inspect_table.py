#!/usr/bin/env python3
"""Print the health of iot.sensor_readings, or watch AIStor clean it up.

  inspect_table.py                          one report
  inspect_table.py --save out/before.json   one report, saved for comparison
  inspect_table.py --compare out/before.json  report next to a saved one
  inspect_table.py --watch out/before.json  poll until all three jobs have run
"""
import argparse
import json
import sys
import time

import lakehouse as lh

ROWS = [
    ("rows", "rows"),
    ("snapshots", "snapshots"),
    ("data_files", "data files (current)"),
    ("avg_file_kb", "avg data file KB"),
    ("objects_on_storage", "objects on storage"),
    ("storage_mb", "storage MB"),
    ("query_ms", "DuckDB query ms"),
]


def compare(before: dict, after: lh.Health) -> None:
    a = after.as_dict()
    print(f"\n  {'':24}{'before':>12}{'after':>12}")
    for key, label in ROWS:
        print(f"  {label:24}{before[key]:>12,}{a[key]:>12,}")


def watch(before: dict, timeout: int, min_snapshots: int) -> int:
    """Wait until compaction, snapshot expiration and unreferenced file
    removal have each visibly changed the table."""
    print(f"Watching {lh.IDENTIFIER} (every 15s, up to {timeout}s). Starting point: "
          f"{before['data_files']} data files, {before['snapshots']} snapshots, "
          f"{before['objects_on_storage']} objects on storage.\n")
    cat = lh.catalog()
    done = {}
    objects_at_compaction = None
    deadline = time.time() + timeout
    while time.time() < deadline:
        h = lh.health(cat, query_runs=0)
        compacted = h.data_files < before["data_files"]
        if compacted and objects_at_compaction is None:
            objects_at_compaction = h.objects_on_storage
        checks = {
            "compaction": compacted,
            "snapshot expiration": h.snapshots <= min_snapshots + 1 < before["snapshots"],
            # Wait for the sweep after compaction, which removes the files that
            # compaction and snapshot expiration left unreferenced.
            "unreferenced file removal": (objects_at_compaction is not None
                                          and h.objects_on_storage < objects_at_compaction),
        }
        for job, ok in checks.items():
            if ok and job not in done:
                done[job] = time.strftime("%H:%M:%S")
        flags = "  ".join(f"[{'x' if j in done else ' '}] {j}" for j in checks)
        print(f"  {time.strftime('%H:%M:%S')}  files {h.data_files:>4}  snapshots {h.snapshots:>4}  "
              f"objects {h.objects_on_storage:>5}   {flags}")
        if len(done) == len(checks):
            print("\nAll three maintenance jobs have run.")
            return 0
        time.sleep(15)
    print("\nTimed out. Check `./scripts/run.sh status` for job status.", file=sys.stderr)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    ap.add_argument("--compare")
    ap.add_argument("--watch", metavar="BEFORE_JSON")
    ap.add_argument("--min-snapshots", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=600, help="seconds, with --watch")
    args = ap.parse_args()

    if args.watch:
        with open(args.watch) as f:
            return watch(json.load(f), args.timeout, args.min_snapshots)

    h = lh.health()
    lh.print_health(h, f"{lh.WAREHOUSE}.{lh.IDENTIFIER}")
    if args.save:
        with open(args.save, "w") as f:
            json.dump(h.as_dict(), f, indent=2)
    if args.compare:
        with open(args.compare) as f:
            compare(json.load(f), h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
