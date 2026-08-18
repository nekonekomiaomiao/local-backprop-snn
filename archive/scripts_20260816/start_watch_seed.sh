#!/bin/bash
cd "/root/Default Project"
setsid nohup bash watch_seed_scan.sh > /tmp/watch_seed_scan.log 2>&1 < /dev/null &
sleep 3
ps -ef | grep watch_seed_scan | grep -v grep
cat /tmp/watch_seed_scan.log