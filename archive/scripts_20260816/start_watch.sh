#!/bin/bash
cd "/root/Default Project"
setsid nohup bash watch_lif5.sh > /tmp/watch_lif5.log 2>&1 < /dev/null &
sleep 3
ps -ef | grep watch_lif5 | grep -v grep
cat /tmp/watch_lif5.log