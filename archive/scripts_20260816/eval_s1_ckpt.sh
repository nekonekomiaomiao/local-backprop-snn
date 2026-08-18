#!/bin/bash
cd "/root/Default Project"
setsid nohup python3 eval_multiseed.py exp4/lif5_s1_t5000_k1_isi50_a15_14k/ckpt_14000.npz \
  > exp4/lif5_s1_t5000_k1_isi50_a15_14k/ms_eval.log 2>&1 < /dev/null &
echo "launched pid=$!"
sleep 600
echo "--- after 10 min:"
cat exp4/lif5_s1_t5000_k1_isi50_a15_14k/ms_eval.log