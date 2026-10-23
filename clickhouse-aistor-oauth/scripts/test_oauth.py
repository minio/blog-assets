#!/usr/bin/env python3
"""End-to-end OAuth test: Keycloak -> AIStor STS -> ClickHouse.

1. Get an OAuth access token from Keycloak (client_credentials grant).
2. Exchange it at AIStor STS (AssumeRoleWithWebIdentity) for temporary S3 creds.
3. Use those creds from ClickHouse to write and read Parquet on AIStor.
4. Check the negative cases: tampered token, missing session token, readonly policy.
5. AIStor Tables: ClickHouse creates, writes and reads an Iceberg table through
   the AIStor Iceberg REST catalog, authenticating with OAuth (client_credentials)
   against AIStor's own token endpoint and, separately, against Keycloak.
6. Ingest: copy the Iceberg table into a local MergeTree table for hot data.

Stdlib only, so it runs in a bare python image.
"""
import base64
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET

KEYCLOAK_URL = os.environ.get("KEYCLOAK_URL", "http://keycloak:8080/realms/aistor")
AISTOR_URL = os.environ.get("AISTOR_URL", "http://minio:9000")
CLICKHOUSE_URL = os.environ.get("CLICKHOUSE_URL", "http://clickhouse:8123")
CLICKHOUSE_USER = os.environ.get("CLICKHOUSE_USER", "default")
CLICKHOUSE_PASSWORD = os.environ.get("CLICKHOUSE_PASSWORD", "clickhouse")
BUCKET = os.environ.get("BUCKET", "clickhouse-oauth")

RW_CLIENT = ("clickhouse", "clickhouse-secret")
RO_CLIENT = ("clickhouse-ro", "clickhouse-ro-secret")

# AIStor Tables (Iceberg REST catalog)
CATALOG_URL = f"{AISTOR_URL}/_iceberg/v1"
WAREHOUSE = os.environ.get("WAREHOUSE", "analytics")
NAMESPACE = os.environ.get("NAMESPACE", "demo")
# AIStor user: OAuth client for AIStor's token endpoint, and storage keys for
# the data files (the catalog doesn't vend credentials in this setup).
# Demo values only.
SVC_USER = ("clickhouse-svc", "clickhouse-svc-secret")
ICEBERG_CLIENT = ("clickhouse-iceberg", "clickhouse-iceberg-secret")
ICEBERG_SETTINGS = {"allow_database_iceberg": "1", "allow_insert_into_iceberg": "1"}

STS_NS = {"sts": "https://sts.amazonaws.com/doc/2011-06-15/"}

results = []


def check(name, ok, detail=""):
    results.append(ok)
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f"\n       {detail}" if detail else ""))


def post(url, data, headers=None, method="POST"):
    """Send a request and return (status, body) without raising on HTTP errors."""
    body = None if data is None else data if isinstance(data, bytes) else data.encode()
    req = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def get_token(client_id, client_secret):
    status, body = post(
        f"{KEYCLOAK_URL}/protocol/openid-connect/token",
        urllib.parse.urlencode({
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }),
        {"Content-Type": "application/x-www-form-urlencoded"},
    )
    if status != 200:
        raise RuntimeError(f"Keycloak token request failed ({status}): {body}")
    return json.loads(body)["access_token"]


def jwt_claims(token):
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))


def assume_role(token, duration=3600):
    """Return (status, creds_dict_or_error_text) from AIStor STS."""
    status, body = post(
        AISTOR_URL,
        urllib.parse.urlencode({
            "Action": "AssumeRoleWithWebIdentity",
            "Version": "2011-06-15",
            "DurationSeconds": str(duration),
            "WebIdentityToken": token,
        }),
        {"Content-Type": "application/x-www-form-urlencoded"},
    )
    if status != 200:
        return status, body
    creds = ET.fromstring(body).find(".//sts:Credentials", STS_NS)
    return status, {
        "access_key": creds.findtext("sts:AccessKeyId", namespaces=STS_NS),
        "secret_key": creds.findtext("sts:SecretAccessKey", namespaces=STS_NS),
        "session_token": creds.findtext("sts:SessionToken", namespaces=STS_NS),
        "expiration": creds.findtext("sts:Expiration", namespaces=STS_NS),
    }


def catalog_token(client_id, client_secret):
    """OAuth client_credentials token from AIStor's Iceberg REST token endpoint."""
    status, body = post(
        f"{CATALOG_URL}/oauth/tokens",
        urllib.parse.urlencode({
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "catalog",
        }),
        {"Content-Type": "application/x-www-form-urlencoded"},
    )
    if status != 200:
        raise RuntimeError(f"AIStor catalog token request failed ({status}): {body}")
    return json.loads(body)["access_token"]


def catalog(method, path, token):
    """Call the AIStor Iceberg REST catalog. Returns (status, body)."""
    return post(f"{CATALOG_URL}/{WAREHOUSE}/{path}", None,
                {"Authorization": f"Bearer {token}"}, method=method)


def ch(sql, settings=None):
    """Run a query over the ClickHouse HTTP interface. Returns (ok, text)."""
    url = CLICKHOUSE_URL
    if settings:
        url += "/?" + urllib.parse.urlencode(settings)
    status, body = post(
        url,
        sql,
        {"X-ClickHouse-User": CLICKHOUSE_USER, "X-ClickHouse-Key": CLICKHOUSE_PASSWORD},
    )
    return status == 200, body.strip()


def lit(s):
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def s3_args(path, creds, with_token=True):
    url = f"{AISTOR_URL}/{BUCKET}/{path}"
    args = [lit(url), lit(creds["access_key"]), lit(creds["secret_key"])]
    if with_token:
        args.append(lit(creds["session_token"]))
    args.append("'Parquet'")
    return ", ".join(args)


def short(text, n=200):
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "..."


def main():
    print("== 1. OAuth token from Keycloak ==")
    token = get_token(*RW_CLIENT)
    claims = jwt_claims(token)
    check("Got access token for client 'clickhouse'", bool(token))
    print(f"       iss={claims.get('iss')} aud={claims.get('aud')} "
          f"azp={claims.get('azp')} policy={claims.get('policy')}")

    print("\n== 2. Exchange token at AIStor STS ==")
    status, creds = assume_role(token)
    check("AssumeRoleWithWebIdentity returned temporary credentials", status == 200,
          f"AccessKeyId={creds['access_key']} expires={creds['expiration']}"
          if status == 200 else short(creds))
    if status != 200:
        return summary()

    bad = token[:-4] + ("AAAA" if not token.endswith("AAAA") else "BBBB")
    status, err = assume_role(bad)
    check("Tampered token is rejected by STS", status != 200, short(str(err)))

    print("\n== 3. ClickHouse <-> AIStor with STS credentials ==")
    ok, out = ch(
        f"INSERT INTO FUNCTION s3({s3_args('trips/trips.parquet', creds)}) "
        "SELECT number AS id, toDate('2026-01-01') + (number % 30) AS day, "
        "round(randCanonical() * 100, 2) AS fare FROM numbers(10000) "
        "SETTINGS s3_truncate_on_insert = 1"
    )
    check("ClickHouse wrote Parquet to AIStor", ok, "" if ok else short(out))

    ok, out = ch(
        f"SELECT count(), round(sum(fare), 2), min(day), max(day) "
        f"FROM s3({s3_args('trips/*.parquet', creds)}) FORMAT TSV"
    )
    check("ClickHouse read the data back", ok and out.startswith("10000"), short(out))

    ok, out = ch(f"SELECT count() FROM s3({s3_args('trips/*.parquet', creds, with_token=False)})")
    check("Read without session token is denied", not ok, short(out))

    print("\n== 4. Policy from token claim (readonly client) ==")
    ro_token = get_token(*RO_CLIENT)
    status, ro_creds = assume_role(ro_token)
    check("Readonly client got STS credentials", status == 200,
          f"policy={jwt_claims(ro_token).get('policy')}" if status == 200 else short(ro_creds))
    if status == 200:
        ok, out = ch(f"SELECT count() FROM s3({s3_args('trips/*.parquet', ro_creds)})")
        check("Readonly creds can read", ok and out == "10000", short(out))
        ok, out = ch(
            f"INSERT INTO FUNCTION s3({s3_args('trips/denied.parquet', ro_creds)}) "
            "SELECT 1 AS id SETTINGS s3_truncate_on_insert = 1"
        )
        check("Readonly creds cannot write", not ok, short(out))

    print("\n== 5. AIStor Tables via Iceberg REST catalog (AIStor OAuth endpoint) ==")
    iceberg_roundtrip(
        "aistor_tables", "trips",
        oauth_server_uri=f"{CATALOG_URL}/oauth/tokens",
        credential=SVC_USER, scope="catalog",
    )

    ch("DROP DATABASE IF EXISTS aistor_bad")
    ok, out = ch(
        catalog_database_sql("aistor_bad", f"{CATALOG_URL}/oauth/tokens",
                             (SVC_USER[0], "wrong-secret"), "catalog"),
        ICEBERG_SETTINGS,
    )
    if ok:
        # The database may attach lazily; the first catalog call must fail.
        ok, out = ch("SHOW TABLES FROM aistor_bad", ICEBERG_SETTINGS)
    check("Wrong OAuth client secret is rejected", not ok, short(out))
    ch("DROP DATABASE IF EXISTS aistor_bad")

    print("\n== 6. AIStor Tables via Iceberg REST catalog (Keycloak OAuth) ==")
    iceberg_roundtrip(
        "aistor_tables_kc", "trips_kc",
        oauth_server_uri=f"{KEYCLOAK_URL}/protocol/openid-connect/token",
        credential=ICEBERG_CLIENT, scope="openid",
    )

    print("\n== 7. Ingest Iceberg data into a local MergeTree table ==")
    ingest_to_mergetree("aistor_tables", "trips")

    return summary()


def ingest_to_mergetree(db, table):
    """Copy an AIStor Tables Iceberg table into a ClickHouse MergeTree table,
    the usual pattern for keeping hot data local to ClickHouse."""
    source = f"{db}.`{NAMESPACE}.{table}`"
    target = f"default.{table}_hot"
    totals = "SELECT count(), round(sum(fare), 2), min(day), max(day) FROM {} FORMAT TSV"

    ch(f"DROP TABLE IF EXISTS {target}")
    ok, out = ch(
        f"CREATE TABLE {target} (id Int64, day Date, fare Float64) "
        "ENGINE = MergeTree ORDER BY (day, id)"
    )
    check(f"Created MergeTree table {target}", ok, short(out))
    if not ok:
        return

    ok, out = ch(f"INSERT INTO {target} SELECT id, day, fare FROM {source}", ICEBERG_SETTINGS)
    check(f"Ingested {source} into {target}", ok, short(out))
    if not ok:
        return

    ok_src, src = ch(totals.format(source), ICEBERG_SETTINGS)
    ok_dst, dst = ch(totals.format(target))
    check("MergeTree totals match the Iceberg source", ok_src and ok_dst and src == dst,
          f"iceberg:   {src}\n       mergetree: {dst}")


def catalog_database_sql(db, oauth_server_uri, credential, scope):
    """DataLakeCatalog database on the AIStor Tables catalog. The engine
    arguments are storage keys for the data files; catalog_credential is the
    OAuth client used to get a bearer token for catalog calls."""
    return (
        f"CREATE DATABASE {db} ENGINE = DataLakeCatalog({lit(CATALOG_URL)}, "
        f"{lit(SVC_USER[0])}, {lit(SVC_USER[1])}) "
        f"SETTINGS catalog_type = 'rest', warehouse = {lit(WAREHOUSE)}, "
        f"catalog_credential = {lit(credential[0] + ':' + credential[1])}, "
        f"oauth_server_uri = {lit(oauth_server_uri)}, auth_scope = {lit(scope)}, "
        f"storage_endpoint = {lit(AISTOR_URL)}"
    )


def iceberg_roundtrip(db, table, oauth_server_uri, credential, scope):
    admin = catalog_token(*SVC_USER)
    # Clean up earlier runs through the catalog API: ClickHouse DROP TABLE on
    # prefixed REST catalogs is tracked in ClickHouse/ClickHouse#112234.
    catalog("DELETE", f"namespaces/{NAMESPACE}/tables/{table}?purgeRequested=true", admin)

    fq = f"{db}.`{NAMESPACE}.{table}`"
    ok, out = ch(f"DROP DATABASE IF EXISTS {db}")
    ok, out = ch(catalog_database_sql(db, oauth_server_uri, credential, scope), ICEBERG_SETTINGS)
    check(f"ClickHouse attached the catalog as database '{db}'", ok, short(out))
    if not ok:
        return

    ok, out = ch(f"SHOW TABLES FROM {db}", ICEBERG_SETTINGS)
    check("ClickHouse listed tables through the catalog", ok, short(out) or "(none yet)")
    if not ok:
        return

    # The catalog assigns the actual table location. ClickHouse writes an
    # initial v1.metadata.json under this path and treats it as an existing
    # table on later runs, so make it unique per run.
    location = f"{AISTOR_URL}/{WAREHOUSE}/{NAMESPACE}/{table}-{uuid.uuid4().hex[:8]}/"
    ok, out = ch(
        f"CREATE TABLE {fq} (id Int64, day Date, fare Float64) "
        f"ENGINE = IcebergS3({lit(location)}, {lit(SVC_USER[0])}, {lit(SVC_USER[1])})",
        ICEBERG_SETTINGS,
    )
    check(f"ClickHouse created Iceberg table {NAMESPACE}.{table}", ok, short(out))
    if not ok:
        return

    ok, out = ch(
        f"INSERT INTO {fq} SELECT number AS id, toDate('2026-01-01') + (number % 30) AS day, "
        "round(randCanonical() * 100, 2) AS fare FROM numbers(10000)",
        ICEBERG_SETTINGS,
    )
    check("ClickHouse inserted 10000 rows", ok, short(out))

    ok, out = ch(f"SELECT count(), round(sum(fare), 2), min(day), max(day) FROM {fq} FORMAT TSV",
                 ICEBERG_SETTINGS)
    check("ClickHouse read the Iceberg table back", ok and out.startswith("10000"), short(out))

    # Confirm the commit landed in the AIStor catalog, not just in ClickHouse.
    status, body = catalog("GET", f"namespaces/{NAMESPACE}/tables/{table}", admin)
    snapshots = json.loads(body)["metadata"].get("snapshots", []) if status == 200 else []
    check("AIStor catalog has the committed snapshot", len(snapshots) == 1,
          f"metadata-location={json.loads(body).get('metadata-location')}"
          if status == 200 else short(body))


def summary():
    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if results and all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
