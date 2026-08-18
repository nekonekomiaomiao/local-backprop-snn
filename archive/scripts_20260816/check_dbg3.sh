#!/bin/bash
sleep 300
for f in /tmp/dbg3_*.log; do
  echo "=== $f"
  cat "$f"
done
echo "--- running: $(ps -ef | grep dbg_lif | grep -v grep | wc -l)"