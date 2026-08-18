#!/bin/bash
sleep 1700
for d in probe_lif_p4 probe_lif_p5; do
  echo "=== exp4/$d"
  tail -8 "/root/Default Project/exp4/$d/run_log.txt"
done
echo "=== exp4/lif3_t2000_k1_isi50_a15_20k"
tail -4 "/root/Default Project/exp4/lif3_t2000_k1_isi50_a15_20k/run_log.txt"
echo "=== exp4/lif2_tm0p5_isi50_a15_20k (control)"
tail -2 "/root/Default Project/exp4/lif2_tm0p5_isi50_a15_20k/run_log.txt"