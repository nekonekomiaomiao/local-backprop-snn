#!/bin/bash
cd "/root/Default Project"
launch() {
  local dir=$1; shift
  mkdir -p "exp4/$dir"
  setsid nohup bash -c 'cd "/root/Default Project/exp4/'$dir'" && python3 ../../mnist_shared.py '"$@"' > run_log.txt 2>&1' < /dev/null &
  local pid=$!
  sleep 3
  if kill -0 $pid 2>/dev/null; then echo "$dir: alive"; else echo "$dir: DEAD"; fi
}
# P1: kappa=1.0 (balance +5544)
launch probe_lif_p1 "0 2000 1.0 200 1.5e-8 3000 30 1000 1000 0.02 1.0 0 0 50 0.5"
# P2: TARGET=2000 (balance +8524)
launch probe_lif_p2 "0 2000 1.0 200 1.5e-8 3000 30 2000 1000 0.02 0.2 0 0 50 0.5"
# P3: TARGET=2000 + kappa=1.0
launch probe_lif_p3 "0 2000 1.0 200 1.5e-8 3000 30 2000 1000 0.02 1.0 0 0 50 0.5"