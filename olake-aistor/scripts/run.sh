#!/bin/bash
#
# OLake + AIStor Tables demo
# Postgres CDC -> OLake -> Iceberg tables in AIStor Tables.
#
#   ./scripts/run.sh          start the stack, run both syncs and verify each
#   ./scripts/run.sh sync     run one more OLake sync (continues from state.json)
#   ./scripts/run.sh test     re-verify the tables (after the change sync)
#   ./scripts/run.sh down     stop everything and remove volumes and OLake output
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if [ ! -f .env ]; then
	echo "Missing .env - copy .env.example to .env and set MINIO_LICENSE." >&2
	exit 1
fi

CFG=/mnt/config
# Fixed prefix so the Iceberg namespace is predictable: olake_shop_public
PREFIX=(--destination-database-prefix olake)

olake() { docker compose run --rm olake "$@"; }

step() { printf '\n==> %s\n' "$*"; }

sync() {
	local state=()
	[ -f olake/state.json ] && state=(--state "$CFG/state.json")
	olake sync \
		--config "$CFG/source.json" \
		--destination "$CFG/destination.json" \
		--available-streams "$CFG/available_streams.json" \
		--selected-streams "$CFG/selected_streams.json" \
		"${PREFIX[@]}" "${state[@]}"
}

case "${1:-up}" in
up)
	# The demo expects a fresh Postgres and warehouse. A rerun on top of an
	# earlier one would fail the "initial" checks and re-apply changes.sql.
	if [ -f olake/state.json ]; then
		echo "A previous demo run exists (olake/state.json)." >&2
		echo "  Re-verify it:   $0 test" >&2
		echo "  Start fresh:    $0 down && $0" >&2
		exit 1
	fi

	step "Starting Postgres and AIStor"
	docker compose up -d --wait postgres minio
	docker compose up minio-init

	step "Checking the Postgres source (connection + CDC prerequisites)"
	olake check --config "$CFG/source.json"
	step "Checking the AIStor destination (test table in the catalog)"
	olake check --destination "$CFG/destination.json"

	step "Discovering streams"
	olake discover --config "$CFG/source.json" "${PREFIX[@]}"
	docker compose run --rm tester select

	step "First sync: full backfill of customers and orders"
	sync
	docker compose run --rm tester initial

	step "Changing rows in Postgres (7 inserts, 5 updates)"
	docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U postgres -d shop -f /changes.sql

	step "Second sync: CDC from the WAL"
	sync
	docker compose run --rm tester changes

	echo ""
	echo "AIStor console: http://localhost:9001  (minioadmin / minioadmin)"
	echo "Postgres:       localhost:5432          (postgres / postgres, db shop)"
	;;
sync)
	sync
	;;
test)
	docker compose run --rm tester "${2:-changes}"
	;;
down)
	docker compose --profile olake --profile test down -v
	rm -rf olake/available_streams.json olake/selected_streams.json olake/streams.json \
		olake/state.json olake/stats.json olake/sync_* olake/logs olake/user_id.txt olake/*.log
	;;
*)
	echo "Usage: $0 [up|sync|test|down]" >&2
	exit 1
	;;
esac
