#!/bin/bash
sleep 420
for d in probe_lif_p1 probe_lif_p2 probe_lif_p3; do
  echo "=== exp4/$d"
  tail -8 "/root/Default Project/exp4/$d/run_log.txt"
done