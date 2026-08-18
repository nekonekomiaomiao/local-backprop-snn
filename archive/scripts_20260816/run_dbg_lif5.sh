#!/bin/bash
cd "/root/Default Project"
CKPT=exp4/reset_a15_cont/mnist_checkpoint.npz
launch() {
  local name=$1; shift
  setsid nohup python3 dbg_lif.py $CKPT "$@" > /tmp/$name.log 2>&1 < /dev/null &
  local pid=$!
  sleep 3
  if kill -0 $pid 2>/dev/null; then echo "$name: alive (pid $pid)"; else echo "$name: DEAD"; fi
}
launch dbg5_if_cont 0.0 0 200 1.0
launch dbg5_lif_isiLearn 0.5 50 200 1.0 0 "" 1
launch dbg5_lifR_isiLearn 0.5 50 200 1.0 0 rand 1