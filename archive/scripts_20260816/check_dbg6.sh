#!/bin/bash
sleep 60
for f in /tmp/dbg5_lif_isiLearn.log /tmp/dbg5_lifR_isiLearn.log; do
  echo "=== $f"
  cat "$f"
done
echo "--- running: $(ps -ef | grep dbg_lif | grep -v grep | wc -l)"