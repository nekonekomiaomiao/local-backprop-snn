#!/bin/bash
cd "/root/Default Project"
launch() {
  local name=$1; shift
  setsid nohup python3 dbg_lif.py exp4/reset_a15_cont/mnist_checkpoint.npz "$@" > /tmp/$name.log 2>&1 < /dev/null &
  local pid=$!
  sleep 3
  if kill -0 $pid 2>/dev/null; then echo "$name: alive"; else echo "$name: DEAD"; fi
}
# random-init signal under target/kappa variants
launch dbg9_lifR_t1000_k0p2 0.5 50 200 1.0 0 rand 0 0.2 1000 0.2
launch dbg9_lifR_t2000_k0p2 0.5 50 200 1.0 0 rand 0 0.2 2000 0.2
launch dbg9_lifR_t3000_k0p2 0.5 50 200 1.0 0 rand 0 0.2 3000 0.2
launch dbg9_lifR_t1000_k1 0.5 50 200 1.0 0 rand 0 0.2 1000 1.0
launch dbg9_ifR_t1000_k0p2 0.0 0 200 1.0 0 rand 0 0.2 1000 0.2
launch dbg9_ifR_t1000_k0 0.0 0 200 1.0 0 rand 0 0.2 1000 0.0