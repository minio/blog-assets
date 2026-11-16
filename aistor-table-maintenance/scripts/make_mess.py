#!/usr/bin/env python3
"""Stream small micro-batches into iot.sensor_readings, the way a streaming
ingest job or CDC connector does.

Every append is its own Iceberg commit, so each batch leaves one small Parquet
file, one manifest, one manifest list and one snapshot behind.

  make_mess.py [--batches 200] [--rows 500]
"""
import argparse
import random
import time
from datetime import datetime, timedelta, timezone

import pyarrow as pa

import lakehouse as lh

SITES = ["amsterdam", "austin", "bangalore", "frankfurt", "sao-paulo", "tokyo"]
DEVICES = [f"dev-{i:03d}" for i in range(60)]


def batch(start_id: int, rows: int, t0: datetime) -> pa.Table:
    rnd = random.Random(start_id)
    devices = [rnd.choice(DEVICES) for _ in range(rows)]
    return pa.Table.from_pydict(
        {
            "reading_id": list(range(start_id, start_id + rows)),
            "device_id": devices,
            "site": [SITES[int(d[-3:]) % len(SITES)] for d in devices],
            "ts": [t0 + timedelta(milliseconds=10 * i) for i in range(rows)],
            "temperature_c": [round(rnd.gauss(22, 4), 2) for _ in range(rows)],
            "humidity_pct": [round(rnd.uniform(20, 80), 1) for _ in range(rows)],
        },
        schema=lh.arrow_schema(),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--rows", type=int, default=500, help="rows per batch")
    args = ap.parse_args()

    cat = lh.catalog()
    table = lh.ensure_table(cat)
    start_id = (table.current_snapshot() and lh.health(cat, query_runs=0).rows) or 0
    print(f"Streaming {args.batches} micro-batches of {args.rows} rows into "
          f"{lh.WAREHOUSE}.{lh.IDENTIFIER}")

    started = time.perf_counter()
    t0 = datetime.now(timezone.utc)
    for i in range(args.batches):
        table.append(batch(start_id + i * args.rows, args.rows, t0 + timedelta(seconds=5 * i)))
        if (i + 1) % 25 == 0 or i + 1 == args.batches:
            print(f"  committed {i + 1:>4}/{args.batches} batches")
    print(f"Done in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
