#!/bin/bash
sleep 1800
cd "/root/Default Project"
for D in lif5_s1_t5000_k1_isi50_a15_14k lif5_s2_t5000_k1_isi50_a15_14k; do
  echo "=== $D:"
  tail -2 "exp4/$D/run_log.txt" 2>/dev/null
  cat "exp4/${D}_ckpts/eval.log" 2>/dev/null | tail -4
done
echo "=== memory:"
free -g | head -2