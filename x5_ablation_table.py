#!/usr/bin/env python3
"""X5: 主消融表编排 —— 混合长跑验收(EXPERIMENT_PROGRESS 记录)与 1k 快照(CSV 实测)。
数据均为既有测量，本脚本只做编排，输出 docs/paper/ablation_main.md。"""
import csv
import os

CSV = "mnist_lif_table_results.csv"

def get(T, K, TM, ISI, TE=0.2, a="1.5e-08", RS=0):
    for r in csv.DictReader(open(CSV, newline="")):
        if (float(r["TARGET"]) == T and float(r["KAPPA"]) == K and float(r["TAU_M"]) == TM
                and int(r["ISI"]) == ISI and float(r["TAU_E"]) == TE and r["alpha"] == a and int(r["RESET"]) == RS):
            return r
    return None

out = []
out.append("# 主消融表（Main Ablation）—— 论文编排版\n")
out.append("> 长跑验收 = 完整训练(14k-40k)后的多 seed 冻结评估(5×n=500, τ_m=0.5/ISI=100 读出)，记录于 EXPERIMENT_PROGRESS.md；")
out.append("> 1k 快照 = 同协议 1000 样本训练快照(mnist_lif_table_results.csv)。\n")

out.append("## 表 A · 协议消融\n")
out.append("| 协议 | 配置(TARGET/κ/τ_m/ISI) | 长跑验收 acc | 1k冻结 | 1k对齐 | corrW3 | SNR_W3 | sign_W3 |")
out.append("|---|---|---|---|---|---|---|---|")
A = [
    ("IF 无泄漏（对照）",       1000, 0.2, 0.0, 0,  "0.348*(训练内置)",  "if"),
    ("硬清零 reset（对照）",    1000, 0.2, 0.0, 0,  "0.824 (单/长/计 0.817/0.848/0.850)", "reset"),
    ("LIF+ISI 未标定（对照）",  1000, 0.2, 0.5, 50, "0.501",  "uncal"),
    ("**LIF+ISI 重标定（主结果）**", 5000, 1.0, 0.5, 50, "**0.877±0.007**", "recal"),
]
for name, T, K, TM, ISI, lr, tag in A:
    r = get(T, K, TM, ISI, RS=(1 if tag == "reset" else 0))
    if not r:
        print("缺数据:", name); continue
    out.append(f"| {name} | T{T}/K{K}/τ_m{TM}/ISI{ISI} | {lr} | {float(r['frozen_acc']):.3f} | "
               f"{float(r['align_all']):+.3f} | {float(r['corr_W3']):+.3f} | {float(r['snr_W3']):.3f} | {float(r['sign_W3']):.3f} |")
out.append("\n*IF 无泄漏长跑仅训练内置评估 0.348。\n")

out.append("## 表 B · 参数单变量消融（1k 快照口径，其余参数=基线 T5000/K1.0/τ_m0.5/ISI50）\n")
base = get(5000, 1.0, 0.5, 50)
out.append(f"基线 1k 冻结 acc = **{float(base['frozen_acc']):.3f}**\n")
sweeps = [
    ("TARGET", [("3000", "T"), ("5000", "T"), ("7000", "T")], "TARGET"),
    ("KAPPA",  [("0.2", "K"), ("0.5", "K"), ("1.0", "K")], "KAPPA"),
    ("TAU_M",  [("0.2", "TM"), ("0.5", "TM"), ("1.0", "TM")], "TAU_M"),
    ("ISI",    [("0", "I"), ("50", "I"), ("100", "I")], "ISI"),
]
for pname, opts, label in sweeps:
    out.append(f"### {label}\n")
    out.append("| {0} | 1k 冻结acc | loss平台 | Δacc |".format(label))
    out.append("|---|---|---|---|")
    for vl, dim in opts:
        T, K, TM, ISI = 5000.0, 1.0, 0.5, 50
        if dim == "T": T = float(vl)
        elif dim == "K": K = float(vl)
        elif dim == "TM": TM = float(vl)
        else: ISI = int(vl)
        r = get(T, K, TM, ISI)
        if not r:
            print("缺参数点:", pname, vl); continue
        acc = float(r["frozen_acc"])
        out.append(f"| {vl} | {acc:.3f} | {float(r['loss_plateau']):.4f} | {acc - float(base['frozen_acc']):+.3f} |")
    out.append("")

os.makedirs("docs/paper", exist_ok=True)
path = "docs/paper/ablation_main.md"
open(path, "w", encoding="utf-8").write("\n".join(out))
print("已生成:", path, "|", len(out), "行")
