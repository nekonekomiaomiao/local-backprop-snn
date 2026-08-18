#!/bin/bash
sleep 90
for f in /tmp/dbg_*.log; do
  echo "=== $f"
  cat "$f"
done
echo "--- running count: $(ps -ef | grep dbg_lif | grep -v grep | wc -l)"