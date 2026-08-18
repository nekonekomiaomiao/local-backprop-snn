#!/bin/bash
sleep 1200
cd "/root/Default Project"
for D in lif5_s1_t5000_k1_isi50_a15_14k lif5_s2_t5000_k1_isi50_a15_14k; do
  echo "=== $D:"
  tail -3 "exp4/$D/run_log.txt" 2>/dev/null
  echo "--- eval.log:"
  cat "exp4/${D}_ckpts/eval.log" 2>/dev/null | tail -5
done