#!/bin/bash
cd "/root/Default Project"
setsid nohup bash watch_s2.sh > /tmp/watch_s2.log 2>&1 < /dev/null &
sleep 3
ps -ef | grep watch_s2 | grep -v grep
cat /tmp/watch_s2.log