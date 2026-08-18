#!/bin/bash
ps -ef | grep dbg_lif | grep -v grep
echo "--- logs:"
for f in /tmp/dbg7_*.log; do
  echo "=== $f ($(wc -c < $f) bytes)"
  cat "$f"
done