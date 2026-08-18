#!/bin/bash
DIR="/root/Default Project/exp4/lif5_t5000_k1_isi50_a15_20k"
SNAP="/root/Default Project/exp4/lif5_ckpts"
mkdir -p "$SNAP" || exit 1
cd "$SNAP" || exit 1
LAST=""
while true; do
  if [ -f "$DIR/mnist_checkpoint.npz" ]; then
    H=$(md5sum "$DIR/mnist_checkpoint.npz" | cut -d' ' -f1)
    if [ "$H" != "$LAST" ]; then
      STEP=$(grep -o 'checkpoint saved @ [0-9]*' "$DIR/run_log.txt" | tail -1 | grep -o '[0-9]*')
      [ -z "$STEP" ] && STEP=$(stat -c %Y "$DIR/mnist_checkpoint.npz")
      cp "$DIR/mnist_checkpoint.npz" "ckpt_${STEP}.npz"
      echo "[$(date +%H:%M)] snapshot @ ${STEP}"
      if [ -n "$LAST" ]; then
        python3 ../../eval_mf_batch.py --tau_m=0.5 --isi_steps=100 --kappa=1.0 --n=1000 "ckpt_${STEP}.npz" >> lif5_eval.log 2>&1 &
      fi
      LAST="$H"
    fi
  fi
  if ! pgrep -f "lif5_t5000_k1" > /dev/null; then
    echo "training finished; last snapshot evaluated"
    break
  fi
  sleep 300
done