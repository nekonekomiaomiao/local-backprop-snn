#!/bin/bash
ps -ef | grep -E 'dbg_lif|mnist_shared' | grep -v grep
echo "--- logs:"
for f in /tmp/dbg6_*.log; do
  echo "=== $f ($(wc -c < $f) bytes)"
  cat "$f"
done
echo "--- training log:"
head -5 "/root/Default Project/exp4/lif2_tm0p5_isi50_a15_20k/run_log.txt"