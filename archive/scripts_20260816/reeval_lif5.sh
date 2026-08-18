#!/bin/bash
cd "/root/Default Project"
for c in 12000 16000 20000; do
  python3 eval_mf_batch.py --tau_m=0.5 --isi_steps=100 --kappa=1.0 --n=1000 "exp4/lif5_ckpts/ckpt_${c}.npz" 2>&1 | tail -1
done