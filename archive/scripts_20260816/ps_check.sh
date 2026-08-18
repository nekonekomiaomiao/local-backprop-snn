#!/bin/bash
ps -ef | grep eval_multiseed | grep -v grep
echo "---procs---"
nproc