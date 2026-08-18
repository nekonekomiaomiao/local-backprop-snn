#!/bin/bash
cd "/root/Default Project"
echo "=== foreground test run ==="
timeout 90 python3 dbg_lif.py exp4/reset_a15_cont/mnist_checkpoint.npz 0.5 50 100 1.0 2>&1 | head -40
echo "EXIT=$?"