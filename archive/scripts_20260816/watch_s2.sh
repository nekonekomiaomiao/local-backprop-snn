#!/bin/bash
DIR="/root/Default Project/exp4/lif5_s2_t5000_k1_isi50_a15_14k"
SNAP="/root/Default Project/exp4/lif5_s2_t5000_k1_isi50_a15_14k_ckpts"
mkdir -p "$SNAP"
LAST=""
while true; do
  if [ -f "$DIR/mnist_checkpoint.npz" ]; then
    H=$(md5sum "$DIR/mnist_checkpoint.npz" | cut -d' ' -f1)
    if [ "$H" != "$LAST" ]; then
      STEP=$(grep -o 'checkpoint saved @ [0-9]*' "$DIR/run_log.txt" | tail -1 | grep -o '[0-9]*')
      [ -z "$STEP" ] && STEP=$(stat -c %Y "$DIR/mnist_checkpoint.npz")
      cp "$DIR/mnist_checkpoint.npz" "$SNAP/ckpt_${STEP}.npz"
      echo "[$(date +%H:%M)] s2 snapshot @ ${STEP}"
      LAST="$H"
    fi
  fi
  if ! pgrep -f "mnist_shared.py 2 14000" > /dev/null; then
    echo "s2 rerun finished"
    break
  fi
  sleep 200
done