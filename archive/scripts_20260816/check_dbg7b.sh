#!/bin/bash
for f in /tmp/dbg7_*.log; do
  echo "=== $f ($(wc -c < $f) bytes)"
  cat "$f"
done
echo "--- training:"
tail -3 "/root/Default Project/exp4/lif2_tm0p5_isi50_a15_20k/run_log.txt"
echo "--- running: $(ps -ef | grep -cE 'dbg_lif|mnist_shared')"