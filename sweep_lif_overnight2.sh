#!/bin/bash
# Overnight LIF parameter sweep - batch 2 (waits for batch 1, then runs).
# Runs: protocol references (IF/reset), TAU_E scaling probes, alpha probes.
ROOT="/root/Default Project"
LOGD="$ROOT/logs"

# tag TARGET KAPPA TAU_M ISI TAU_E alpha RESET
QUEUE=(
  "lif_if_ref   1000 0.2 0.0  0  0.2 1.5e-8 0"
  "lif_reset_ref 1000 0.2 0.0 0  0.2 1.5e-8 1"
  "lif_te0p5    5000 1.0 1.0  50 0.5 1.5e-8 0"
  "lif_te0p1    5000 1.0 0.5  50 0.1 1.5e-8 0"
  "lif_a3e8     5000 1.0 0.5  50 0.2 3e-8   0"
  "lif_a7p5e9   5000 1.0 0.5  50 0.2 7.5e-9 0"
)

N_TRAIN=1000
CONC=2
BATCH1_MARKER="/tmp/sweep_lif_overnight.log"

echo "[sweep2] start $(date '+%F %T')"

# ---- wait for batch 1 ----
for i in $(seq 1 600); do
  if grep -q "\[sweep\] all done" "$BATCH1_MARKER" 2>/dev/null; then
    echo "[sweep2] batch1 done, proceed"
    break
  fi
  sleep 20
done
if ! grep -q "\[sweep\] all done" "$BATCH1_MARKER" 2>/dev/null; then
  echo "[sweep2] WARNING: batch1 not done after 3h20m; proceeding anyway"
fi

# ---- fix up CSV: batch-1 rows lack RESET col (41 fields); insert RESET=0 ----
cd "$ROOT"
python3 - <<'PY'
import csv
p = "mnist_lif_table_results.csv"
with open(p, newline="") as fh:
    text = fh.read().splitlines()
out = [text[0]]
changed = 0
for line in text[1:]:
    if not line.strip():
        continue
    if line.count(",") == 40:  # 41 fields, old format
        fields = line.split(",")
        newline = ",".join(fields[:12] + ["0"] + fields[12:])
        out.append(newline)
        changed += 1
    else:
        out.append(line)
with open(p, "w", newline="") as fh:
    fh.write("\n".join(out) + "\n")
print(f"[sweep2] CSV fixup: {changed} old-format rows updated with RESET=0")
PY

run_one() {
  local tag="$1" tg="$2" kp="$3" tm="$4" isi="$5" te="$6" al="$7" rs="$8"
  local logf="$LOGD/table_lif_${tag}.log"
  if grep -q "TSVROW|" "$logf" 2>/dev/null; then
    echo "[sweep2] $tag already done, skip"
    return 0
  fi
  echo "[sweep2] launch $tag (T=$tg K=$kp TM=$tm ISI=$isi TE=$te alpha=$al RESET=$rs) $(date '+%F %T')"
  setsid nohup python3 mnist_lif_table.py shared $N_TRAIN 1.0 200 "$al" 3000 30 "$tg" 0 "$kp" 0 0.02 "$isi" "$tm" "$te" "$rs" \
    > "$logf" 2>&1 < /dev/null &
  local pid=$!
  sleep 2
  [ -e "/proc/$pid" ] && echo -800 > "/proc/$pid/oom_score_adj" 2>/dev/null
  echo $pid
}

for spec in "${QUEUE[@]}"; do
  while [ "$(pgrep -f 'mnist_lif_table.py shared' | wc -l)" -ge "$CONC" ]; do
    sleep 20
  done
  read -r tag tg kp tm isi te al rs <<< "$spec"
  run_one "$tag" "$tg" "$kp" "$tm" "$isi" "$te" "$al" "$rs"
done

while [ "$(pgrep -f 'mnist_lif_table.py shared' | wc -l)" -gt 0 ]; do
  sleep 30
done

echo "[sweep2] all done $(date '+%F %T')"
echo "=== per-run summary (TSVROW) ==="
for spec in "${QUEUE[@]}"; do
  read -r tag tg kp tm isi te al rs <<< "$spec"
  logf="$LOGD/table_lif_${tag}.log"
  if grep -q "TSVROW|" "$logf" 2>/dev/null; then
    grep "TSVROW|" "$logf" | sed "s/^/$tag /"
  else
    echo "$tag FAILED (no TSVROW)"
  fi
done
echo "[sweep2] done marker"