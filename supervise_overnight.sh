#!/bin/bash
# Overnight supervisor: keep s1 multi-seed eval and s2 training alive until completion.
# Usage: setsid nohup bash supervise_overnight.sh > /tmp/overnight_supervisor.log 2>&1 &
ROOT="/root/Default Project"
S1_CKPT="$ROOT/exp4/lif5_s1_t5000_k1_isi50_a15_14k/ckpt_14000.npz"
S1_LOG="$ROOT/exp4/lif5_s1_t5000_k1_isi50_a15_14k/ms_eval.log"
S2_DIR="$ROOT/exp4/lif5_s2_t5000_k1_isi50_a15_14k"
S2_LOG="$S2_DIR/run_log.txt"

# Lower OOM priority for newly spawned tasks.
protect() {
  for p in "$@"; do
    [ -e "/proc/$p" ] && echo -800 > "/proc/$p/oom_score_adj" 2>/dev/null || true
  done
}

echo "[supervisor] start $(date '+%F %T')"

while true; do
  # ---- s1 eval ----
  if [ -f "$S1_CKPT" ] && ! grep -q "mean=" "$S1_LOG" 2>/dev/null; then
    if ! pgrep -f "eval_multiseed.py exp4/lif5_s1_t5000_k1_isi50_a15_14k/ckpt_14000.npz" > /dev/null; then
      echo "[supervisor] restart s1 eval $(date '+%F %T')"
      cd "$ROOT"
      setsid nohup python3 eval_multiseed.py "$S1_CKPT" > "$S1_LOG" 2>&1 < /dev/null &
      sleep 2
      protect $(pgrep -f "eval_multiseed.py exp4/lif5_s1_t5000_k1_isi50_a15_14k/ckpt_14000.npz")
    fi
  fi

  # ---- s2 training ----
  if [ -f "$S2_DIR/mnist_checkpoint.npz" ] && grep -q "training done" "$S2_LOG" 2>/dev/null; then
    : # already done
  elif ! pgrep -f "mnist_shared.py 2 14000 1.0 200 1.5e-8 3000 30 5000 1000 0.02 1.0 0 0 50 0.5" > /dev/null; then
    if [ -f "$S2_LOG" ] && grep -q "training done" "$S2_LOG" 2>/dev/null; then
      : # already done
    else
      echo "[supervisor] restart s2 training $(date '+%F %T')"
      cd "$S2_DIR"
      setsid nohup python3 ../../mnist_shared.py 2 14000 1.0 200 1.5e-8 3000 30 5000 1000 0.02 1.0 0 0 50 0.5 \
        > "$S2_LOG" 2>&1 < /dev/null &
      sleep 2
      protect $(pgrep -f "mnist_shared.py 2 14000 1.0 200 1.5e-8 3000 30 5000 1000 0.02 1.0 0 0 50 0.5")
    fi
  fi

  # ---- s2 multi-seed eval (fire when training done) ----
  S2_EVAL_LOG="$S2_DIR/ms_eval.log"
  if [ -f "$S2_LOG" ] && grep -q "training done" "$S2_LOG" 2>/dev/null; then
    if ! grep -q "mean=" "$S2_EVAL_LOG" 2>/dev/null; then
      if ! pgrep -f "eval_multiseed.py $S2_DIR/mnist_checkpoint.npz" > /dev/null; then
        echo "[supervisor] launch s2 eval $(date '+%F %T')"
        cd "$ROOT"
        setsid nohup python3 eval_multiseed.py "$S2_DIR/mnist_checkpoint.npz" > "$S2_EVAL_LOG" 2>&1 < /dev/null &
        sleep 2
        protect $(pgrep -f "eval_multiseed.py $S2_DIR/mnist_checkpoint.npz")
      fi
    fi
  fi

  # ---- memory ----
  AVAIL_KB=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  if [ "${AVAIL_KB:-0}" -lt 1048576 ]; then
    echo "[supervisor] WARNING low memory: ${AVAIL_KB}kB $(date '+%F %T')"
  fi

  sleep 60
done
