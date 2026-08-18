#!/usr/bin/env python3
"""Aggregate LIF sweep trajectory files (traj_*.csv) into a compact evolution table.

For each run: loss/acc/align/corr at snapshot 1 (step 200), mid, and last snapshot,
and their deltas — showing how alignment/correlation relate to training evolution
and final frozen acc.
"""
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = sorted(glob.glob(os.path.join(HERE, "traj_T*.csv")))
if not FILES:
    print("no traj files yet")
    sys.exit(0)


def parse_name(fn):
    b = os.path.basename(fn)
    m = re.match(r"traj_T(\d+)_K([\d.]+)_TM([\d.]+)_ISI(\d+)_TE([\d.]+)_RS(\d+)", b)
    if not m:
        return b
    t, k, tm, isi, te, rs = m.groups()
    return f"T{t}/K{k}/TM{tm}/ISI{isi}/TE{te}/RS{rs}"


print("# LIF 扫描轨迹演化汇总（traj_*.csv）\n")
print("| run | steps | loss@last | acc@last | alAll@first→@last (Δ) | alAll_std@last | corrW1/W2/W3@last | biasW1@last |")
print("|---|---|---|---|---|---|---|")
for f in FILES:
    try:
        d = np.genfromtxt(f, delimiter=",", names=True)
    except Exception as e:
        print(f"| {parse_name(f)} | PARSE ERR {e} |")
        continue
    steps = int(d["step"][-1])
    loss_last = d["loss_roll"][-1]
    acc_last = d["acc_roll"][-1]
    al_first = d["alAll"][0]
    al_last = d["alAll"][-1]
    al_std_last = d["alAll_std"][-1]
    c1, c2, c3 = d["cW1"][-1], d["cW2"][-1], d["cW3"][-1]
    b1 = d["bW1"][-1]
    # evolution: slope of alAll between first and last
    mid_i = len(d) // 2
    al_mid = d["alAll"][mid_i]
    print(f"| {parse_name(f)} | {steps} | {loss_last:.4f} | {acc_last:.3f} | "
          f"{al_first:+.3f} → {al_mid:+.3f} → {al_last:+.3f} (Δ{al_last - al_first:+.3f}) | {al_std_last:.3f} | "
          f"{c1:+.3f}/{c2:+.3f}/{c3:+.3f} | {b1:.3f} |")

print("\n说明：traj 每 200 样本一行；alAll=200 窗口平均对齐（cos 与下降方向），corr=窗口内 (E,d) 相关。")
print("对齐随训练下降 → 局部学习早期对齐高、后期噪声主导；corr@last 与 frozen acc 的关系见分析表。")