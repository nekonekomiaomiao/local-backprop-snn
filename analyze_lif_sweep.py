#!/usr/bin/env python3
"""Analyze LIF parameter sweep results from mnist_lif_table_results.csv.

Relationship analysis:
  1. Single-variable response: for each swept param (TARGET/KAPPA/TAU_M/ISI),
     list runs differing ONLY in that param (others == baseline) and show
     loss_plateau / frozen_acc / align_all / corr / snr / eff / sign response.
  2. Cross-metric correlations: align_all vs loss, corr vs loss, sign vs loss,
     updateSNR vs loss, bias vs align, etc.  (Spearman + Pearson)
  3. Emit docs/TABLE_LIF_PARAM_SWEEP.md
"""
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "mnist_lif_table_results.csv")

BASELINE = dict(TARGET=5000.0, KAPPA=1.0, TAU_M=0.5, ISI=50, TAU_E=0.2, alpha=1.5e-8)
PARAMS = ["TARGET", "KAPPA", "TAU_M", "ISI", "TAU_E", "alpha"]
PARAM_RANGES = {
    "TARGET": [3000.0, 5000.0, 7000.0],
    "KAPPA": [0.2, 0.5, 1.0],
    "TAU_M": [0.2, 0.5, 1.0],
    "ISI": [0, 50, 100],
    "TAU_E": [0.1, 0.2, 0.5],
    "alpha": [7.5e-9, 1.5e-8, 3e-8],
}
# floats compare exactly: values written by repr(float) of ints/str -> exact
FLOAT_PARAMS = ["TARGET", "KAPPA", "TAU_M", "TAU_E", "alpha"]
INT_PARAMS = ["ISI"]

METRICS = [
    ("loss_plateau", "loss 平台(末100)"),
    ("loss_std", "loss std"),
    ("frozen_acc", "冻结 test acc"),
    ("align_all", "对齐 ALL"),
    ("align_std", "对齐 std"),
    ("corr_W1", "corr(e,d) W1"),
    ("corr_W2", "corr(e,d) W2"),
    ("corr_W3", "corr(e,d) W3"),
    ("bias_W1", "bias W1"),
    ("bias_W2", "bias W2"),
    ("bias_W3", "bias W3"),
    ("snr_W1", "SNR W1"),
    ("snr_W2", "SNR W2"),
    ("snr_W3", "SNR W3"),
    ("eff_W1", "expEff W1"),
    ("eff_W2", "expEff W2"),
    ("eff_W3", "expEff W3"),
    ("sign_W1", "signCons W1"),
    ("sign_W2", "signCons W2"),
    ("sign_W3", "signCons W3"),
]

# 协议对照（跨协议比较；来自 TASK_LIST 已知对照 + 本轮补充）
PROTOCOLS = []  # filled after rows load


def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return pearson(rx, ry)


def key_eq(row, base):
    for k in PARAMS:
        if abs(row[k] - base[k]) > 1e-9:
            return False
    return True

def main():
    if not os.path.exists(CSV_PATH):
        print("no CSV yet:", CSV_PATH)
        return 1
    with open(CSV_PATH, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("CSV empty")
        return 1
    for r in rows:
        for k in FLOAT_PARAMS:
            r[k] = float(r[k])
        for k in INT_PARAMS:
            r[k] = int(float(r[k]))
        r["RESET"] = int(float(r["RESET"])) if r.get("RESET") else 0
        # convert every metric column to float
        for k in r:
            if k in ("which", "seed"):
                continue
            if r[k] not in (None, ""):
                r[k] = float(r[k])
        r["seed"] = int(float(r["seed"]))
    # drop smoke-test leftovers (tiny N)
    rows = [r for r in rows if int(r["N"]) >= 500]
    n_rows = len(rows)
    print(f"# LIF 参数扫描关系分析  ({n_rows} runs, N>=500)\n")

    base = [r for r in rows if key_eq(r, BASELINE)]
    print(f"基线组合 {BASELINE}（RESET=0）：{len(base)} 个复现")

    # ---- protocol comparison ----
    # matches by (TARGET,KAPPA,TAU_M,ISI,RESET)
    def find(r0):
        for r in rows:
            if all(abs(r[k] - r0[k]) < 1e-9 for k in
                   ("TARGET", "KAPPA", "TAU_M") if k != "TAU_M" or r0[k] is not None) and \
               all(int(r[k]) == int(r0[k]) for k in ("ISI", "RESET")):
                return r
        return None
    specs = [
        ("IF 无泄漏（对照 0.348 模型）", dict(TARGET=1000.0, KAPPA=0.2, TAU_M=0.0, ISI=0, RESET=0)),
        ("硬清零 reset（对照 0.824 模型）", dict(TARGET=1000.0, KAPPA=0.2, TAU_M=0.0, ISI=0, RESET=1)),
        ("LIF+ISI 未标定（0.501 模型）", dict(TARGET=1000.0, KAPPA=0.2, TAU_M=0.5, ISI=50, RESET=0)),
        ("LIF+ISI 重标定基线（0.877 模型）", dict(TARGET=5000.0, KAPPA=1.0, TAU_M=0.5, ISI=50, RESET=0)),
    ]
    print("\n## 四协议同测量协议对照\n")
    print("| 协议 | loss | frozen_acc | align_ALL | corrW1/W2/W3 | biasW1 | snrW3 | effW3 | signW3 |")
    print("|---|---|---|---|---|---|---|---|---|")
    found_any = False
    for name, spec in specs:
        r = find(spec)
        if r is None:
            print(f"| {name} | （不在本批数据中，用历史值：loss 0.xxx / acc 0.xxx） |")
            continue
        found_any = True
        print(f"| {name} | {r['loss_plateau']:.4f} | {r['frozen_acc']:.4f} | {r['align_all']:+.3f} | "
              f"{r['corr_W1']:+.3f}/{r['corr_W2']:+.3f}/{r['corr_W3']:+.3f} | {r['bias_W1']:.3f} | "
              f"{r['snr_W3']:.3f} | {r['eff_W3']:.3f} | {r['sign_W3']:.3f} |")
    if not found_any:
        print("（协议对照数据待 batch-2 完成后出现）")

    # ---- single-variable response ----
    print("\n## 单变量响应（RESET=0，其余参数 == 基线）\n")
    for p in PARAMS:
        vals = PARAM_RANGES[p]
        sub = [r for r in rows
               if r["RESET"] == 0 and
               all(abs(r[q] - BASELINE[q]) < 1e-9 for q in PARAMS if q != p)]
        sub = sorted(sub, key=lambda r: r[p])
        print(f"### {p}  ({len(sub)} runs)\n")
        print("| {0} | loss_plateau | loss_std | frozen_acc | align_ALL | corrW1/W2/W3 | biasW1/W2/W3 | snrW1/W2/W3 | effW1/W2/W3 | signW1/W2/W3 |".format(p))
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in sub:
            print(f"| {r[p]} | {r['loss_plateau']:.4f} | {r['loss_std']:.4f} | "
                  f"{r['frozen_acc']:.4f} | {r['align_all']:+.3f} | "
                  f"{r['corr_W1']:+.3f}/{r['corr_W2']:+.3f}/{r['corr_W3']:+.3f} | "
                  f"{r['bias_W1']:.3f}/{r['bias_W2']:.3f}/{r['bias_W3']:.3f} | "
                  f"{r['snr_W1']:.3f}/{r['snr_W2']:.3f}/{r['snr_W3']:.3f} | "
                  f"{r['eff_W1']:.3f}/{r['eff_W2']:.3f}/{r['eff_W3']:.3f} | "
                  f"{r['sign_W1']:.3f}/{r['sign_W2']:.3f}/{r['sign_W3']:.3f} |")
        print()

    # ---- cross-metric correlations ----
    print("\n## 跨指标相关（全部 runs）\n")
    keys = ["loss_plateau", "frozen_acc", "align_all", "align_std",
            "corr_W1", "corr_W2", "corr_W3", "snr_W1", "snr_W3",
            "eff_W1", "eff_W3", "sign_W1", "sign_W3", "bias_W1"]
    pairs = [
        ("align_all", "loss_plateau"), ("align_all", "frozen_acc"),
        ("align_all", "corr_W1"), ("align_all", "corr_W3"),
        ("corr_W1", "loss_plateau"), ("corr_W3", "loss_plateau"),
        ("sign_W3", "loss_plateau"), ("snr_W3", "loss_plateau"),
        ("eff_W3", "loss_plateau"), ("bias_W1", "loss_plateau"),
        ("corr_W3", "align_all"), ("snr_W3", "align_all"),
        ("align_std", "loss_plateau"), ("align_std", "frozen_acc"),
        ("snr_W1", "loss_plateau"), ("sign_W1", "frozen_acc"),
    ]
    print("| X | Y | Spearman ρ | Pearson r | n |")
    print("|---|---|---|---|---|")
    for x, y in pairs:
        xs = [r[x] for r in rows]
        ys = [r[y] for r in rows]
        print(f"| {x} | {y} | {spearman(xs, ys):+.3f} | {pearson(xs, ys):+.3f} | {len(rows)} |")

    # ---- parameter effects on key metrics (effect size vs baseline) ----
    print("\n## 参数 vs 关键指标（相对基线变化）\n")
    print("| 参数组合 | Δloss | Δacc | Δalign | ΔcorrW3 | ΔsnrW3 | ΔeffW3 | ΔsignW3 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if key_eq(r, BASELINE):
            continue
        dl = r["loss_plateau"] - base[0]["loss_plateau"] if base else float("nan")
        da = r["frozen_acc"] - base[0]["frozen_acc"] if base else float("nan")
        dg = r["align_all"] - base[0]["align_all"] if base else float("nan")
        dc = r["corr_W3"] - base[0]["corr_W3"] if base else float("nan")
        ds = r["snr_W3"] - base[0]["snr_W3"] if base else float("nan")
        de = r["eff_W3"] - base[0]["eff_W3"] if base else float("nan")
        dsign = r["sign_W3"] - base[0]["sign_W3"] if base else float("nan")
        tag = "T{0}/K{1}/TM{2}/ISI{3}{4}".format(
            int(r["TARGET"]), r["KAPPA"], r["TAU_M"], r["ISI"],
            "/RESET" if r["RESET"] else "")
        print(f"| {tag} | {dl:+.4f} | {da:+.4f} | {dg:+.3f} | {dc:+.3f} | "
              f"{ds:+.3f} | {de:+.3f} | {dsign:+.3f} |")

    print("\n原始全量数据：`mnist_lif_table_results.csv`；逐 run 日志 `logs/table_lif_*.log`")


if __name__ == "__main__":
    sys.exit(main())