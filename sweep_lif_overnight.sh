#!/bin/bash
# Overnight LIF parameter sweep runner.
# Queue of runs: <tag> <TARGET> <KAPPA> <TAU_M> <ISI> <TAU_E> <alpha>
# Runs 2 at a time (concurrency 2), oom-protected, logs per run, summary at end.
ROOT="/root/Default Project"
LOGD="$ROOT/logs"
mkdir -p "$LOGD"

# tag TARGET KAPPA TAU_M ISI TAU_E alpha
QUEUE=(
  "lif_base  5000 1.0 0.5 50 0.2 1.5e-8"
  "lif_t3000 3000 1.0 0.5 50 0.2 1.5e-8"
  "lif_t7000 7000 1.0 0.5 50 0.2 1.5e-8"
  "lif_k0p2  5000 0.2 0.5 50 0.2 1.5e-8"
  "lif_k0p5  5000 0.5 0.5 50 0.2 1.5e-8"
  "lif_tm0p2 5000 1.0 0.2 50 0.2 1.5e-8"
  "lif_tm1p0 5000 1.0 1.0 50 0.2 1.5e-8"
  "lif_isi0  5000 1.0 0.5 0  0.2 1.5e-8"
  "lif_isi100 5000 1.0 0.5 100 0.2 1.5e-8"
)

N_TRAIN=1000
CONC=2

run_one() {
  local tag="$1" tg="$2" kp="$3" tm="$4" isi="$5" te="$6" al="$7"
  local logf="$LOGD/table_lif_${tag}.log"
  if grep -q "TSVROW|" "$logf" 2>/dev/null; then
    echo "[sweep] $tag already done, skip"
    return 0
  fi
  echo "[sweep] launch $tag (T=$tg K=$kp TM=$tm ISI=$isi TE=$te alpha=$al) $(date '+%F %T')"
  cd "$ROOT"
  setsid nohup python3 mnist_lif_table.py shared $N_TRAIN 1.0 200 "$al" 3000 30 "$tg" 0 "$kp" 0 0.02 "$isi" "$tm" "$te" \
    > "$logf" 2>&1 < /dev/null &
  local pid=$!
  sleep 2
  [ -e "/proc/$pid" ] && echo -800 > "/proc/$pid/oom_score_adj" 2>/dev/null
  echo $pid
}

wait_slots() {
  # wait until fewer than CONC sweep processes running
  while true; do
    local n=$(pgrep -f "mnist_lif_table.py shared" | wc -l)
    [ "$n" -lt "$CONC" ] && break
    sleep 10
  done
}

echo "[sweep] start $(date '+%F %T')  queue=${#QUEUE[@]} runs  N=$N_TRAIN  conc=$CONC"

declare -a PIDS
for spec in "${QUEUE[@]}"; do
  # fill up to CONC slots
  while [ "$(pgrep -f 'mnist_lif_table.py shared' | wc -l)" -ge "$CONC" ]; do
    sleep 20
  done
  read -r tag tg kp tm isi te al <<< "$spec"
  run_one "$tag" "$tg" "$kp" "$tm" "$isi" "$te" "$al"
done

# wait for all to finish
while [ "$(pgrep -f 'mnist_lif_table.py shared' | wc -l)" -gt 0 ]; do
  sleep 30
done

echo "[sweep] all done $(date '+%F %T')"
echo "=== per-run summary (TSVROW) ==="
for spec in "${QUEUE[@]}"; do
  read -r tag tg kp tm isi te al <<< "$spec"
  logf="$LOGD/table_lif_${tag}.log"
  if grep -q "TSVROW|" "$logf" 2>/dev/null; then
    grep "TSVROW|" "$logf" | sed "s/^/$tag /"
  else
    echo "$tag FAILED (no TSVROW)"
  fi
done
echo "[sweep] done marker"