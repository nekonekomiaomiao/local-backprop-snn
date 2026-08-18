#!/bin/bash
cd "/root/Default Project"
declare -A LAST
for D in lif5_s1_t5000_k1_isi50_a15_14k lif5_s2_t5000_k1_isi50_a15_14k; do
  LAST[$D]=""
done
while true; do
  ALIVE=0
  for D in lif5_s1_t5000_k1_isi50_a15_14k lif5_s2_t5000_k1_isi50_a15_14k; do
    CK="exp4/$D/mnist_checkpoint.npz"
    if [ -f "$CK" ]; then
      H=$(md5sum "$CK" | cut -d' ' -f1)
      if [ "${LAST[$D]}" != "$H" ]; then
        STEP=$(grep -o 'checkpoint saved @ [0-9]*' "exp4/$D/run_log.txt" | tail -1 | grep -o '[0-9]*')
        mkdir -p "exp4/${D}_ckpts"
        cp "$CK" "exp4/${D}_ckpts/ckpt_${STEP}.npz"
        echo "[$(date +%H:%M)] $D snapshot @ ${STEP}"
        if [ -n "${LAST[$D]}" ]; then
          python3 eval_multiseed.py "exp4/${D}_ckpts/ckpt_${STEP}.npz" >> "exp4/${D}_ckpts/eval.log" 2>&1 &
        fi
        LAST[$D]="$H"
      fi
    fi
    pgrep -f "mnist_shared.py [12] 14000" > /dev/null && ALIVE=$((ALIVE + 1))
  done
  if [ "$ALIVE" -eq 0 ]; then echo "both runs finished"; break; fi
  sleep 240
done