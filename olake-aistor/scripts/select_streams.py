#!/usr/bin/env python3
"""Set every stream that `olake discover` selected to CDC with positional deletes.

discover writes selected_streams.json with all streams selected. This keeps
that selection and pins the settings the demo relies on, so the test's
expected row counts hold:
  sync_mode      cdc   full backfill on the first sync, then the WAL
  normalization  true  one Iceberg column per source column
  update_type    pos   positional deletes (Iceberg v2)
  append_mode    false upserts, so an UPDATE replaces the old row
"""
import json
import sys

path = sys.argv[1]
with open(path) as f:
    doc = json.load(f)

streams = doc.get("selected_streams", {})
names = []
for namespace, entries in streams.items():
    for s in entries:
        s["sync_mode"] = "cdc"
        s["normalization"] = True
        s["update_type"] = "pos"
        s["append_mode"] = False
        names.append(f"{namespace}.{s['stream_name']}")

if not names:
    sys.exit(f"No streams found in {path}. Did discover run?")

with open(path, "w") as f:
    json.dump(doc, f, indent=2)
print("Selected (cdc, pos deletes): " + ", ".join(sorted(names)))
