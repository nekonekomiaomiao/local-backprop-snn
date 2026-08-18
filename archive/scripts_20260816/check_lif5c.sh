#!/bin/bash
sleep 3400
echo "=== lif5 tail:"
tail -4 "/root/Default Project/exp4/lif5_t5000_k1_isi50_a15_20k/run_log.txt"
echo "=== lif5 eval log:"
cat "/root/Default Project/exp4/lif5_ckpts/lif5_eval.log" 2>/dev/null
echo "=== running:"
ps -ef | grep -E 'lif5|watch_lif5' | grep -v grep | wc -l