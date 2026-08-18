#!/bin/bash
sleep 900
for d in probe_lif_p1 probe_lif_p2 probe_lif_p3; do
  echo "=== exp4/$d"
  tail -6 "/root/Default Project/exp4/$d/run_log.txt"
done
echo "--- lif2 control:"
tail -2 "/root/Default Project/exp4/lif2_tm0p5_isi50_a15_20k/run_log.txt"