#!/bin/bash
sleep 420
echo "=== s1 eval log:"
cat "/root/Default Project/exp4/lif5_s1_t5000_k1_isi50_a15_14k/ms_eval.log" 2>&1
echo "=== procs:"
ps -ef | grep -E "eval_multiseed" | grep -v grep