#!/bin/bash
#
# run-benchmark.sh — Run 04-TPCDS-Iceberg with varying concurrency and loops.
#
# Usage:
#   ./run-benchmark.sh                        # run full MATRIX below
#   ./run-benchmark.sh -t 4 -l 3             # single run: 4 threads, 3 loops
#   ./run-benchmark.sh -t 4 -l 3 -Jfoo=bar   # single run + extra JMeter args
#
# Each run writes results to: ~/results/<tag>/
# where tag = t<threads>-l<loops>-<timestamp>  e.g. t4-l3-20260318-143022
#
# Edit MATRIX to define your test combinations: "threads loops"
#
MATRIX=(
  "1 1"
  "2 1"
  "4 1"
  "4 3"
)

JMETER="${JMETER_HOME:-jmeter}"
DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_BASE="${HOME}/results"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

# Parse -t / -l flags; remaining args passed through to JMeter
OPT_T=""
OPT_L=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -t) OPT_T="$2"; shift 2 ;;
    -l) OPT_L="$2"; shift 2 ;;
    *)  EXTRA_ARGS+=("$1"); shift ;;
  esac
done

run_config() {
  local T=$1
  local L=$2
  local TAG="t${T}-l${L}-${TIMESTAMP}"
  local OUT="${RESULTS_BASE}/${TAG}"

  mkdir -p "$OUT"
  echo ""
  echo ">>> threads=${T}  loops=${L}  output=${OUT}"

  "$JMETER" -n \
    -t "$DIR/04-TPCDS-Iceberg.jmx" \
    -Jthreads="${T}" \
    -Jloops="${L}" \
    -Jsummary_report_path="${OUT}/04-tpcds-iceberg-summary.csv" \
    -Jresults_table_path="${OUT}/04-tpcds-iceberg-table.csv" \
    -Jresults_tree_path="${OUT}/04-tpcds-iceberg-tree.csv" \
    "${EXTRA_ARGS[@]}" \
    && echo "<<< Completed: ${TAG}" \
    || { echo "!!! Failed: ${TAG}"; exit 1; }
}

echo "=== TPC-DS Iceberg Benchmark ==="
echo "=== $(date) ==="

if [[ -n "$OPT_T" && -n "$OPT_L" ]]; then
  run_config "$OPT_T" "$OPT_L"
elif [[ -n "$OPT_T" || -n "$OPT_L" ]]; then
  echo "Error: -t and -l must be used together" >&2
  exit 1
else
  for config in "${MATRIX[@]}"; do
    T=$(echo "$config" | awk '{print $1}')
    L=$(echo "$config" | awk '{print $2}')
    run_config "$T" "$L"
  done
fi

echo ""
echo "=== All benchmark runs completed ==="
echo "Results in: ${RESULTS_BASE}/"
ls -ld "${RESULTS_BASE}"/t*-l* 2>/dev/null
