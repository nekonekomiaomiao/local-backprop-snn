"""Generate the 0.8-campaign comparison table markdown from CSV + run logs.
Usage: python3 make_campaign_table.py [out.md]
Reads:
  - mnist_0p8_campaign.csv  (hand/script maintained: one row per combo)
  - meanfield_ckpts/*/run_log.txt  (mean-field test acc trajectory, parsed for 25k/50k/.../300k)
  - exp*/finetune dirs run_log.txt  (fine-tune final acc)
"""
import csv
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "TABLE_0P8_CAMPAIGN.md")


def mf_trajectory(d):
    """Parse meanfield_ckpts/<d>/run_log.txt -> {step: acc} and final n=1000 acc."""
    d = d.replace("meanfield_ckpts/", "")
    p = os.path.join(ROOT, "meanfield_ckpts", d, "run_log.txt")
    if not os.path.exists(p):
        return {}, None
    pts, final = {}, None
    for line in open(p):
        m1 = re.search(r"(\d+)/\d+\s+test\s+([\d.]+)", line)
        if m1 and int(m1.group(1)) % 25000 == 0:
            pts[int(m1.group(1))] = float(m1.group(2))
        m2 = re.search(r"final test acc \(n=1000\) = ([\d.]+)", line)
        if m2:
            final = float(m2.group(1))
    return pts, final


def finetune_acc(d):
    p = os.path.join(ROOT, d, "run_log.txt")
    if not os.path.exists(p):
        return None
    for line in open(p):
        m = re.search(r"final test acc \(n=(\d+)\) = ([\d.]+)", line)
        if m:
            return float(m.group(2))
    return None


rows = list(csv.DictReader(open(os.path.join(ROOT, "mnist_0p8_campaign.csv"))))

lines = []
lines.append("# 0.8 目标冲刺：参数组合对比表（2026-08-15 晚起）\n")
lines.append("> 数据源：`mnist_0p8_campaign.csv`（生成脚本 `make_campaign_table.py`）；均值场 acc 由 run_log 解析。\n")

lines.append("## 1. 均值场 GD 训练（LR=1e-8, seed=0, N=300k, 每 25k 保存 checkpoint）\n")
lines.append("| κ | 25k | 50k | 75k | 100k | 150k | 200k | 250k | 300k(final n=1000) |")
lines.append("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    if r["kind"] != "meanfield":
        continue
    pts, final = mf_trajectory(r["dir"])
    cells = [r["kappa"]]
    for s in [25000, 50000, 75000, 100000, 150000, 200000, 250000]:
        cells.append(f"{pts.get(s, 0):.3f}" if s in pts else "-")
    cells.append(f"**{final:.3f}**" if final is not None else "-")
    lines.append("| " + " | ".join(cells) + " |")

lines.append("\n## 2. 均值场权重 → 脉冲读出（能力证明：权重质量不是瓶颈）\n")
lines.append("> 均值场 300k 权重（无噪声 acc 0.96）灌进脉冲网络。不 reset（跨样本状态污染）时 acc 仅 0.12-0.20；**每样本 reset 后 0.93-0.95**——证明架构 + 读出协议容量足够，瓶颈在训练协议与状态污染。\n")
lines.append("| κ ckpt | 脉冲单次(不 reset) | 脉冲单次(reset) | 长窗(reset) | 计数(reset) | 均值场对照 |")
lines.append("|---|---|---|---|---|---|")
for r in rows:
    if r["kind"] != "mf_pulse_eval":
        continue
    lines.append(f"| {r['dir']} | {r['note'].split(';')[0] if ';' in r['note'] else '-'} | {r['pulse_final']} | {r['pulse_avg']} | {r['pulse_spike']} | {r['meanfield']} |")

lines.append("\n## 3. reset 协议脉冲训练（正面啃：纯脉冲、随机初始化、在线局部学习）\n")
lines.append("| 目录 | κ | α | N(样本) | 最终 n=1000 | 独立评估 单次/长窗/计数 |")
lines.append("|---|---|---|---|---|---|")
for r in rows:
    if r["kind"] != "pulse_train":
        continue
    acc = finetune_acc(r["dir"])
    ev = f"{r['pulse_final']}/{r['pulse_avg']}/{r['pulse_spike']}" if r['pulse_final'] else "-"
    fin = f"**{acc:.3f}**" if acc is not None else "-"
    lines.append(f"| {r['dir']} | {r['kappa']} | {r['alpha']} | {r['N']} | {fin} | {ev} |")

lines.append("\n## 4. 历史基线（0.8 目标前，不 reset 协议）\n")
lines.append("| 配置 | 冻结 acc |")
lines.append("|---|---|")
for r in rows:
    if r["kind"] != "baseline":
        continue
    lines.append(f"| {r['note']} | {r['pulse_final']} |")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(lines) + "\n")
print(f"wrote {OUT} ({len(lines)} lines)")
