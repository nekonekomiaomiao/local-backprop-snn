#!/bin/bash
cd "/root/Default Project"
pkill -f "lif4_t3000_k1_isi50_a15_20k" || true
sleep 2
mkdir -p exp4/lif5_t5000_k1_isi50_a15_20k
setsid nohup bash -c 'cd "/root/Default Project/exp4/lif5_t5000_k1_isi50_a15_20k" && python3 ../../mnist_shared.py 0 20000 1.0 200 1.5e-8 3000 30 5000 1000 0.02 1.0 0 0 50 0.5 > run_log.txt 2>&1' < /dev/null &
sleep 3
ps -ef | grep lif5 | grep -v grep | head -2
mkdir -p exp4/probe_lif_p7
setsid nohup bash -c 'cd "/root/Default Project/exp4/probe_lif_p7" && python3 ../../mnist_shared.py 0 2000 1.0 200 1.5e-8 3000 30 10000 1000 0.02 1.0 0 0 50 0.5 > run_log.txt 2>&1' < /dev/null &
sleep 3
ps -ef | grep probe_lif_p7 | grep -v grep | head -2