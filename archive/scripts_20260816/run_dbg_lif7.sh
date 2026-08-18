#!/bin/bash
cd "/root/Default Project"
launch() {
  local name=$1; shift
  setsid nohup python3 dbg_lif.py exp4/reset_a15_cont/mnist_checkpoint.npz "$@" > /tmp/$name.log 2>&1 < /dev/null &
  local pid=$!
  sleep 3
  if kill -0 $pid 2>/dev/null; then echo "$name: alive (pid $pid)"; else echo "$name: DEAD"; fi
}
launch dbg7_if_cont 0.0 0 200 1.0
launch dbg7_lif_fixed 0.5 50 200 1.0
launch dbg7_lif_rand 0.5 50 200 1.0 0 rand