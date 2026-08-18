#!/bin/bash
# Launch a spiking fine-tune run from a mean-field checkpoint, detached.
# Usage: run_finetune.sh <init_ckpt> <ALPHA> <N> [SEED] [outdir]
set -e
ROOT="/root/Default Project"
INIT="$1"; ALPHA="$2"; N="$3"; SEED="${4:-0}"
OUTDIR="${5:-$ROOT/meanfield_ckpts/finetune_a${ALPHA}_n${N}_s${SEED}}"
mkdir -p "$OUTDIR"
cd "$OUTDIR"
setsid nohup python3 "$ROOT/mnist_finetune_mf.py" "$INIT" "$ALPHA" "$N" "$SEED" > run_log.txt 2> run_err.txt < /dev/null &
echo "launched fine-tune pid=$! -> $OUTDIR (alpha=$ALPHA N=$N seed=$SEED init=$INIT)"
