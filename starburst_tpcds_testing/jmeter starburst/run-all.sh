#!/bin/bash

JMETER="${JMETER_HOME:-jmeter}"
DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${HOME}/results"

mkdir -p "$RESULTS_DIR"

EXTRA_ARGS=("$@")

run() {
  echo ">>> Running: $1"
  "$JMETER" -n -t "$DIR/$1" "${EXTRA_ARGS[@]}" && echo "<<< Completed: $1" || { echo "!!! Failed: $1"; exit 1; }
}

run "01-CREATE-TPCDS.jmx"
run "02-INSERT-INTO-TPCDS.jmx"
run "03-OPTIMIZE-TPCDS.jmx"
run "04-TPCDS-Iceberg.jmx"
run "05-DROP-TPCDS.jmx"

echo "All tests completed."
