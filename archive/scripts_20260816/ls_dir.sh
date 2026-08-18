#!/bin/bash
ls "/root/Default Project/exp4/" | head -20
echo "---"
ls "/root/Default Project/exp4/lif5_t5000_k1_isi50_a15_20k/" 2>&1 | head -5
echo "--- tail:"
tail -3 "/root/Default Project/exp4/lif5_t5000_k1_isi50_a15_20k/run_log.txt" 2>&1