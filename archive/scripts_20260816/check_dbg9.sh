#!/bin/bash
sleep 240
for f in /tmp/dbg9_*.log; do
  echo "=== $f"
  cat "$f"
done
echo "--- training tail:"
tail -2 "/root/Default Project/exp4/lif2_tm0p5_isi50_a15_20k/run_log.txt"