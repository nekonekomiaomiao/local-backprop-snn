#!/bin/bash
# launch dbg_lif random-init runs (diagnostic batch 2, 2026-08-16)
cd "/root/Default Project"
CKPT=exp4/reset_a15_cont/mnist_checkpoint.npz
for cfg in "0.0 0" "0.5 50" "0.2 50"; do
  set -- $cfg
  TM=$1; ISI=$2
  name="dbgr_tm${TM}_isi${ISI}"
  setsid nohup python3 dbg_lif.py $CKPT $TM $ISI 200 1.0 0 rand > /tmp/${name}.log 2>&1 < /dev/null &
  echo "launched $name"
done
echo all-launched