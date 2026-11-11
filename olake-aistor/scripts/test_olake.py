#!/usr/bin/env python3
"""Verify what OLake wrote to AIStor Tables.

  test_olake.py initial   after the first sync (full backfill)
  test_olake.py changes   after postgres/changes.sql and the second (CDC) sync

Two independent views of the same tables:
1. The AIStor Iceberg REST catalog API, with an OAuth client_credentials token
   from AIStor's own token endpoint: namespace, tables and snapshots exist.
2. DuckDB attached to the same catalog with SigV4: the rows themselves.
"""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

import duckdb

AISTOR_URL = os.environ.get("AISTOR_URL", "http://minio:9000")
AISTOR_HOSTPORT = os.environ.get("AISTOR_HOSTPORT", "minio:9000")
WAREHOUSE = os.environ.get("WAREHOUSE", "analytics")
ACCESS_KEY = os.environ.get("ACCESS_KEY", "olake-svc")
SECRET_KEY = os.environ.get("SECRET_KEY", "olake-svc-secret")
CATALOG_URL = f"{AISTOR_URL}/_iceberg/v1"
TABLES = ("customers", "orders")

# Expected state of the source after each phase (see postgres/*.sql).
EXPECTED = {
    "initial": {"customers": 8, "orders": 40, "amount": 6345.00},
    "changes": {"customers": 10, "orders": 45, "amount": 8188.39},
}
# Rows retired by the UPDATEs in changes.sql. With update_type "pos", each one
# becomes a positional delete, not an equality delete.
EXPECTED_POS_DELETES = {"customers": 2, "orders": 3}

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f"\n       {detail}" if detail else ""))


def http(method, url, data=None, headers=None):
    body = None if data is None else data.encode()
    req = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def catalog_token():
    status, body = http(
        "POST", f"{CATALOG_URL}/oauth/tokens",
        urllib.parse.urlencode({
            "grant_type": "client_credentials",
            "client_id": ACCESS_KEY,
            "client_secret": SECRET_KEY,
            "scope": "catalog",
        }),
        {"Content-Type": "application/x-www-form-urlencoded"},
    )
    return status, (json.loads(body).get("access_token") if status == 200 else body)


def catalog(path, token):
    status, body = http("GET", f"{CATALOG_URL}/{WAREHOUSE}/{path}",
                        headers={"Authorization": f"Bearer {token}"})
    try:
        return status, json.loads(body)
    except ValueError:
        return status, body


def find_namespace(token):
    """OLake names the namespace <prefix>_<db>_<schema>, e.g. olake_shop_public."""
    status, body = catalog("namespaces", token)
    if status != 200:
        return None, f"list namespaces -> {status}: {body}"
    names = [".".join(ns) for ns in body.get("namespaces", [])]
    hits = [n for n in names if n.endswith("shop_public")]
    return (hits[0] if hits else None), f"namespaces: {names}"


def catalog_checks(phase):
    print("\n== AIStor Iceberg REST catalog (OAuth) ==")
    status, token = catalog_token()
    check("AIStor issues a catalog token for olake-svc", status == 200,
          "" if status == 200 else f"{status}: {token}")
    if status != 200:
        return None

    ns, detail = find_namespace(token)
    check("OLake created a namespace for shop.public", ns is not None,
          f"using '{ns}'" if ns else detail)
    if not ns:
        return None

    status, body = catalog(f"namespaces/{urllib.parse.quote(ns)}/tables", token)
    listed = sorted(t["name"] for t in body.get("identifiers", [])) if status == 200 else []
    check("customers and orders are registered in the catalog",
          all(t in listed for t in TABLES), f"tables: {listed}")

    for t in TABLES:
        status, body = catalog(f"namespaces/{urllib.parse.quote(ns)}/tables/{t}", token)
        meta = body.get("metadata", {}) if status == 200 else {}
        snaps = meta.get("snapshots", [])
        current = meta.get("current-snapshot-id")
        summary = next((s.get("summary", {}) for s in snaps
                        if s.get("snapshot-id") == current), {})
        want = 1 if phase == "initial" else 2
        check(f"{t}: {len(snaps)} snapshot(s), format v{meta.get('format-version')}",
              current is not None and len(snaps) >= want,
              f"current snapshot: {summary.get('operation')}, "
              f"total-records={summary.get('total-records')}, "
              f"total-position-deletes={summary.get('total-position-deletes')}")
        if phase == "changes":
            pos = summary.get("total-position-deletes")
            eq = summary.get("total-equality-deletes")
            want_pos = EXPECTED_POS_DELETES[t]
            check(f"{t}: updates written as {want_pos} positional deletes, no equality deletes",
                  pos == str(want_pos) and eq in ("0", None),
                  f"total-position-deletes={pos}, total-equality-deletes={eq}")
    return ns


def duckdb_checks(phase, ns):
    print("\n== DuckDB reading the tables (SigV4) ==")
    exp = EXPECTED[phase]
    con = duckdb.connect()
    try:
        con.execute("INSTALL iceberg; LOAD iceberg;")
        con.execute(f"""
            CREATE SECRET aistor_s3 (
              TYPE S3, KEY_ID '{ACCESS_KEY}', SECRET '{SECRET_KEY}',
              ENDPOINT '{AISTOR_HOSTPORT}', URL_STYLE 'path', USE_SSL 0,
              REGION 'us-east-1')""")
        con.execute(f"""
            ATTACH '{WAREHOUSE}' AS aistor (
              TYPE ICEBERG, ENDPOINT '{AISTOR_URL}/_iceberg',
              AUTHORIZATION_TYPE 'sigv4', SIGV4_SERVICE 's3tables',
              SIGV4_REGION 'us-east-1', SECRET aistor_s3)""")
        check("DuckDB attaches the AIStor warehouse", True)
    except Exception as e:  # noqa: BLE001
        check("DuckDB attaches the AIStor warehouse", False, str(e))
        return

    def q(sql):
        return con.execute(sql).fetchall()

    c = f'aistor."{ns}".customers'
    o = f'aistor."{ns}".orders'

    try:
        n, distinct = q(f"SELECT count(*), count(DISTINCT id) FROM {c}")[0]
        check(f"customers: {n} rows (expected {exp['customers']}), one per id",
              n == exp["customers"] and distinct == n, f"distinct ids: {distinct}")

        n, distinct, amount = q(f"SELECT count(*), count(DISTINCT id), sum(amount) FROM {o}")[0]
        check(f"orders: {n} rows (expected {exp['orders']}), one per id",
              n == exp["orders"] and distinct == n, f"distinct ids: {distinct}")
        check(f"orders: amount total {amount:.2f} (expected {exp['amount']:.2f})",
              abs(float(amount) - exp["amount"]) < 0.01)

        cols = [r[0] for r in q(f"DESCRIBE {c}")]
        check("OLake metadata columns present (_olake_id, _op_type, _cdc_lsn)",
              all(x in cols for x in ("_olake_id", "_op_type", "_cdc_lsn")),
              f"columns: {cols}")

        if phase == "changes":
            liam = q(f"SELECT tier FROM {c} WHERE name = 'Liam Carter'")
            check("UPDATE replayed: Liam Carter is gold", liam == [("gold",)], f"got {liam}")
            zara = q(f"SELECT city FROM {c} WHERE name = 'Zara Ali'")
            check("UPDATE replayed: Zara Ali moved to Abu Dhabi",
                  zara == [("Abu Dhabi",)], f"got {zara}")
            delivered = q(f"SELECT list(id ORDER BY id) FROM {o} WHERE status = 'delivered'")[0][0]
            check("UPDATE replayed: orders 1, 4, 7 delivered", delivered == [1, 4, 7],
                  f"got {delivered}")
            ops = dict(q(f"SELECT _op_type, count(*) FROM {c} GROUP BY 1"))
            check("customers rows arrived via snapshot (r), insert (c) and update (u)",
                  ops.get("c") == 2 and ops.get("u") == 2, f"_op_type counts: {ops}")
    except Exception as e:  # noqa: BLE001
        check("DuckDB queries", False, str(e))


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "initial"
    if phase not in EXPECTED:
        sys.exit(f"Usage: {sys.argv[0]} initial|changes")
    print(f"Verifying OLake -> AIStor Tables ({phase})")
    ns = catalog_checks(phase)
    if ns:
        duckdb_checks(phase, ns)
    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    sys.exit(0 if results and all(results) else 1)


if __name__ == "__main__":
    main()
