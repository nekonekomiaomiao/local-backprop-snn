#!/bin/bash
cd "/root/Default Project"
# 1) decisive 20k LIF training (fixed protocol: learn=False ISI, tau_m=0.5, ISI=50, sample 1s)
DIR=exp4/lif2_tm0p5_isi50_a15_20k
mkdir -p $DIR
setsid nohup bash -c 'cd "/root/Default Project/'$DIR'" && python3 ../../mnist_shared.py 0 20000 1.0 200 1.5e-8 3000 30 1000 1000 0.02 0.2 0 0 50 0.5 > run_log.txt 2>&1' < /dev/null &
echo "training launched: $DIR"
# 2) signal-level diagnostics (rand bug fixed)
launch() {
  local name=$1; shift
  setsid nohup python3 dbg_lif.py exp4/reset_a15_cont/mnist_checkpoint.npz "$@" > /tmp/$name.log 2>&1 < /dev/null &
  local pid=$!
  sleep 3
  if kill -0 $pid 2>/dev/null; then echo "$name: alive (pid $pid)"; else echo "$name: DEAD"; fi
}
launch dbg6_if_cont 0.0 0 200 1.0
launch dbg6_lif_fixed 0.5 50 200 1.0
launch dbg6_lif_rand 0.5 50 200 1.0 0 rand