#!/bin/sh
# Entry point for the tester container.
#   select            set every discovered stream to CDC with positional deletes
#   initial|changes   verify the synced tables after the first or second sync
set -e

case "$1" in
select)
	exec python -u /scripts/select_streams.py /mnt/config/selected_streams.json
	;;
*)
	pip install -q duckdb==1.5.5
	exec python -u /scripts/test_olake.py "$@"
	;;
esac
