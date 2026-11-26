#!/bin/bash
#
# AIStor Tables: automated table maintenance demo
#
#   ./scripts/run.sh            start AIStor, make a messy table, turn on
#                               maintenance and watch AIStor clean it up
#   ./scripts/run.sh start      start AIStor and Jupyter only, to present from
#                               the notebook (http://localhost:8888)
#   ./scripts/run.sh mess [N]   stream N more micro-batches (default 200)
#   ./scripts/run.sh policy     (re)apply the maintenance policy
#   ./scripts/run.sh status     maintenance config, job status, metrics, table health
#   ./scripts/run.sh lab        start Jupyter at http://localhost:8888
#   ./scripts/run.sh down       stop everything and remove volumes
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if [ ! -f .env ]; then
	echo "Missing .env - copy .env.example to .env and set MINIO_LICENSE." >&2
	exit 1
fi

WAREHOUSE=aistor/lakehouse
TABLE=$WAREHOUSE/iot/sensor_readings
BATCHES=200
ROWS=500
MIN_SNAPSHOTS=5

step() { printf '\n==> %s\n' "$*"; }
mc() { docker compose run --rm mc "$@"; }
py() { docker compose run --rm lab python "$@"; }

mess() {
	py scripts/make_mess.py --batches "${1:-$BATCHES}" --rows "$ROWS"
}

# The warehouse policy applies to every table in the warehouse. Values are set
# for a live demo; see "Moving to production" in the README.
policy() {
	mc table warehouse maintenance set "$WAREHOUSE" \
		--type icebergCompaction --target-file-size-mb 128 --interval 1
	mc table warehouse maintenance set "$WAREHOUSE" \
		--type icebergSnapshotManagement --max-age-hours 0 --min-snapshots "$MIN_SNAPSHOTS" --interval 1
	mc table warehouse maintenance set "$WAREHOUSE" \
		--type icebergUnreferencedFileRemoval --unreferenced-days 0 --noncurrent-days 1 --interval 1
	mc table warehouse maintenance info "$WAREHOUSE"
}

metrics() {
	curl -s http://localhost:9000/minio/metrics/v3/tables |
		grep -E '^minio_tables_(compaction_files_(rewritten|added)_total|maintenance_(snapshots_expired_total|orphan_files_marked_total|cycle_total))' ||
		echo "(no metrics yet)"
}

case "${1:-up}" in
up)
	if docker compose ps -q minio 2>/dev/null | grep -q .; then
		echo "The demo is already running." >&2
		echo "  Check it:       $0 status" >&2
		echo "  Start fresh:    $0 down && $0" >&2
		exit 1
	fi

	step "Starting AIStor and creating the 'lakehouse' warehouse"
	docker compose build -q lab
	docker compose up -d --wait minio
	docker compose run --rm minio-init

	step "Streaming $BATCHES micro-batches into iot.sensor_readings (no maintenance yet)"
	mess

	step "Table health before maintenance"
	py scripts/inspect_table.py --save out/before.json

	step "Turning on automated maintenance for the warehouse"
	policy

	step "Watching AIStor clean up the table"
	py scripts/inspect_table.py --watch out/before.json --min-snapshots "$MIN_SNAPSHOTS"

	step "Before and after"
	py scripts/inspect_table.py --compare out/before.json

	step "Maintenance job status"
	mc table maintenance status "$TABLE"

	echo ""
	echo "AIStor console: http://localhost:9001  (minioadmin / minioadmin)"
	echo "Notebook:       $0 lab, then http://localhost:8888"
	;;
start)
	step "Starting AIStor, the 'lakehouse' warehouse and Jupyter"
	docker compose build -q lab
	docker compose up -d --wait minio
	docker compose run --rm minio-init
	docker compose up -d lab
	echo ""
	echo "Notebook:       http://localhost:8888/lab/tree/table-maintenance.ipynb"
	echo "AIStor console: http://localhost:9001  (minioadmin / minioadmin)"
	echo "When the notebook says so, run: $0 policy"
	;;
mess)
	mess "${2:-$BATCHES}"
	;;
policy)
	policy
	;;
status)
	step "Warehouse maintenance policy"
	mc table warehouse maintenance info "$WAREHOUSE"
	step "Job status for iot.sensor_readings"
	mc table maintenance status "$TABLE" || true
	step "Maintenance metrics"
	metrics
	step "Table health"
	if [ -f out/before.json ]; then
		py scripts/inspect_table.py --compare out/before.json
	else
		py scripts/inspect_table.py
	fi
	;;
lab)
	docker compose up -d lab
	echo "Jupyter: http://localhost:8888/lab/tree/table-maintenance.ipynb"
	;;
down)
	docker compose --profile tools --profile lab down -v --remove-orphans
	rm -f out/*.json
	;;
*)
	echo "Usage: $0 [up|start|mess [N]|policy|status|lab|down]" >&2
	exit 1
	;;
esac
