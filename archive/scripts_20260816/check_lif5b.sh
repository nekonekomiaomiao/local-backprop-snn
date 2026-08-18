#!/bin/bash
sleep 3300
echo "=== lif5 tail:"
tail -4 "/root/Default Project/exp4/lif5_t5000_k1_isi50_a15_20k/run_log.txt"
echo "=== lif5 eval log:"
cat "/root/Default Project/exp4/lif5_ckpts/lif5_eval.log" 2>/dev/null
echo "=== lif2 control tail:"
tail -2 "/root/Default Project/exp4/lif2_tm0p5_isi50_a15_20k/run_log.txt"