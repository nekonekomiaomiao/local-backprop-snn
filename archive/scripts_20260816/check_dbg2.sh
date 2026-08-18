#!/bin/bash
for f in /tmp/dbg*.log; do
  echo "=== $f"
  tail -25 "$f"
done
echo "--- running: $(ps -ef | grep dbg_lif | grep -v grep | wc -l)"