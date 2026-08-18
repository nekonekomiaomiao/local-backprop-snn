#!/bin/bash
sleep 1200
echo "=== probe_lif_p4 (final)"
tail -4 "/root/Default Project/exp4/probe_lif_p4/run_log.txt"
echo "=== probe_lif_p6"
tail -6 "/root/Default Project/exp4/probe_lif_p6/run_log.txt"
echo "=== lif4"
tail -3 "/root/Default Project/exp4/lif4_t3000_k1_isi50_a15_20k/run_log.txt"