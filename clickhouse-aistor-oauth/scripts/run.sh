#!/bin/bash
#
# ClickHouse + AIStor OAuth test
# Starts Keycloak, AIStor and ClickHouse, then runs the end-to-end test.
#
#   ./scripts/run.sh          start the stack and run the test
#   ./scripts/run.sh test     re-run the test against a running stack
#   ./scripts/run.sh down     stop everything and remove volumes
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if [ ! -f .env ]; then
	echo "Missing .env - copy .env.example to .env and set MINIO_LICENSE." >&2
	exit 1
fi

case "${1:-up}" in
up)
	docker compose up -d --wait keycloak minio clickhouse
	docker compose up minio-init
	docker compose run --rm tester
	echo ""
	echo "Keycloak admin:  http://localhost:8080  (admin / admin)"
	echo "AIStor console:  http://localhost:9001  (minioadmin / minioadmin)"
	echo "ClickHouse HTTP: http://localhost:8123  (default / clickhouse)"
	;;
test)
	docker compose run --rm tester
	;;
down)
	docker compose --profile test down -v
	;;
*)
	echo "Usage: $0 [up|test|down]" >&2
	exit 1
	;;
esac
