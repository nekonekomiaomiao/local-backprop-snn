#!/bin/bash
sleep 3300
echo "=== lif5 tail:"
tail -4 "/root/Default Project/exp4/lif5_t5000_k1_isi50_a15_20k/run_log.txt"
echo "=== watcher:"
cat /tmp/watch_lif5.log
echo "=== eval results so far:"
ls "/root/Default Project/exp4/lif5_ckpts/" 2>/dev/null
cat "/root/Default Project/exp4/lif5_ckpts/lif5_eval.log" 2>/dev/null