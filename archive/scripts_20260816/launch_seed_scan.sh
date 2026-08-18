#!/bin/bash
cd "/root/Default Project"
RUN() {
  local SEED=$1 DIR=$2
  mkdir -p "exp4/$DIR"
  setsid nohup python3 mnist_shared.py $SEED 14000 1.0 200 1.5e-8 3000 30 5000 1000 0.02 1.0 0 0 50 0.5 \
    > "exp4/$DIR/run_log.txt" 2>&1 < /dev/null &
  echo "launched seed=$SEED -> exp4/$DIR  pid=$!"
}
RUN 1 lif5_s1_t5000_k1_isi50_a15_14k
RUN 2 lif5_s2_t5000_k1_isi50_a15_14k
sleep 5
ps -ef | grep mnist_shared | grep -v grep