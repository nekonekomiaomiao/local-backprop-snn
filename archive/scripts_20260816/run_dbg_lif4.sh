#!/bin/bash
cd "/root/Default Project"
CKPT=exp4/reset_a15_cont/mnist_checkpoint.npz
setsid nohup python3 dbg_lif.py $CKPT 0.5 50 200 1.0 > /tmp/dbg4_lif_fixed.log 2>&1 < /dev/null &
PID=$!
echo "pid=$PID"
sleep 5
if kill -0 $PID 2>/dev/null; then echo "alive at 5s"; else echo "DEAD at 5s"; fi
ps -ef | grep dbg_lif | grep -v grep
wc -c /tmp/dbg4_lif_fixed.log