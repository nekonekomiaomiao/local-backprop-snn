#!/bin/bash
# dbg batch 3: IF control + isi_learn bug reproduction
cd "/root/Default Project"
CKPT=exp4/reset_a15_cont/mnist_checkpoint.npz
setsid nohup python3 dbg_lif.py $CKPT 0.0 0 200 1.0 > /tmp/dbg3_if_cont.log 2>&1 < /dev/null &
echo "launched dbg3_if_cont"
setsid nohup python3 dbg_lif.py $CKPT 0.5 50 200 1.0 0 "" 1 > /tmp/dbg3_lif_isiLearn.log 2>&1 < /dev/null &
echo "launched dbg3_lif_isiLearn"
setsid nohup python3 dbg_lif.py $CKPT 0.5 50 200 1.0 > /tmp/dbg3_lif_fixed.log 2>&1 < /dev/null &
echo "launched dbg3_lif_fixed"
setsid nohup python3 dbg_lif.py $CKPT 0.5 50 200 1.0 0 rand 1 > /tmp/dbg3_lifR_isiLearn.log 2>&1 < /dev/null &
echo "launched dbg3_lifR_isiLearn"