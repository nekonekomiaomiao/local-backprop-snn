#!/bin/bash
ps -ef | grep -E "eval_multiseed|mnist_shared" | grep -v grep
echo "=== s2 tail:"
tail -3 "/root/Default Project/exp4/lif5_s2_t5000_k1_isi50_a15_14k/run_log.txt"
echo "=== watch_s2:"
cat /tmp/watch_s2.log