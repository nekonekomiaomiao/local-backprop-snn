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
# main 20k with winning config: TARGET=2000, kappa=1.0
launch lif3_t2000_k1_isi50_a15_20k "0 20000 1.0 200 1.5e-8 3000 30 2000 1000 0.02 1.0 0 0 50 0.5"
# extra probes: TARGET=3000/kappa=1.0 and TARGET=2000/kappa=0.5
launch probe_lif_p4 "0 2000 1.0 200 1.5e-8 3000 30 3000 1000 0.02 1.0 0 0 50 0.5"
launch probe_lif_p5 "0 2000 1.0 200 1.5e-8 3000 30 2000 1000 0.02 0.5 0 0 50 0.5"