#!/bin/bash
sleep 200
for f in /tmp/dbg8_*.log; do
  echo "=== $f"
  cat "$f"
done
echo "--- training tail:"
tail -2 "/root/Default Project/exp4/lif2_tm0p5_isi50_a15_20k/run_log.txt"
echo "--- running: $(ps -ef | grep -cE 'dbg_lif|mnist_shared')"