#!/bin/bash
DIR="/root/Default Project/exp4/lif5_s2_t5000_k1_isi50_a15_14k"
cd "$DIR"
setsid nohup python3 ../../mnist_shared.py 2 14000 1.0 200 1.5e-8 3000 30 5000 1000 0.02 1.0 0 0 50 0.5 \
  > run_log.txt 2>&1 < /dev/null &
PID=$!
sleep 5
if kill -0 $PID 2>/dev/null; then
  echo "s2 rerun launched pid=$PID in $DIR"
  ls run_log.txt
else
  echo "FAILED to launch s2"
  tail -5 run_log.txt
fi