#!/usr/bin/env python3
"""论文排图：X2 学习曲线+参数扫描 / X6 噪声-性能 / X3 单实现vs多实现分布。
数据：run_log.txt（学习曲线）、mnist_lif_table_results.csv（扫描/噪声-性能）、ms_eval.log（seed 级分布）。
输出：docs/paper/figures/*.png（英文标签，出版级）。"""
import csv
import json
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "docs/paper/figures")
os.makedirs(FIG, exist_ok=True)
CSV = os.path.join(ROOT, "mnist_lif_table_results.csv")
RUNLOG = os.path.join(ROOT, "exp4/lif5_t5000_k1_isi50_a15_20k/run_log.txt")

# ---------- 数据 ----------
def parse_runlog(path):
    samples, loss, acc, tpts = [], [], [], []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"train (\d+)/\d+  loss\(roll100\) ([\d.e-]+)  acc\(roll100\) ([\d.e-]+)", line)
        if m:
            samples.append(int(m.group(1))); loss.append(float(m.group(2))); acc.append(float(m.group(3)))
        m2 = re.search(r">>> test acc @ (\d+): ([\d.e-]+)", line)
        if m2:
            tpts.append((int(m2.group(1)), float(m2.group(2))))
    return np.array(samples), np.array(loss), np.array(acc), tpts

def read_csv():
    rows = []
    for r in csv.DictReader(open(CSV, newline="")):
        rows.append({k: (float(v) if k not in ("which", "seed") else v) for k, v in r.items()})
    return rows

def seed_vals(logpath):
    """从 eval_multiseed 日志解析 seed 级 acc 列表。"""
    for line in open(logpath, encoding="utf-8", errors="replace"):
        m = re.search(r"seeds\[123,1,2,3,4\]: \[([^\]]*)\]", line)
        if m:
            return [float(x) for x in m.group(1).replace("'", "").replace("'", "").replace("[", "").replace("]", "").split(",")]
    return None

# ---------- Fig 1: 学习曲线（X2）----------
def fig_learning_curves():
    samples, loss, acc, tpts = parse_runlog(RUNLOG)
    fig, ax1 = plt.subplots(figsize=(6.4, 3.6))
    ax1.set_xlabel("Training samples"); ax1.set_ylabel("Normalized MSE  (roll-100)", color="#c44")
    ax1.plot(samples, loss, color="#c44", lw=1.4)
    ax1.tick_params(axis="y", labelcolor="#c44"); ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy (roll-100 / frozen test)", color="#283")
    ax2.plot(samples, acc, color="#47a", lw=1.4, label="train acc (in-sample)")
    if tpts:
        ts = [t[0] for t in tpts]; ta = [t[1] for t in tpts]
        ax2.plot(ts, ta, "o-", color="#283", ms=3, lw=0.8, label="frozen test acc (n=1000)")
    ax2.axhline(0.877, color="#888", ls="--", lw=0.8)
    ax2.text(samples[-1], 0.885, "long-run peak 0.877", ha="right", fontsize=8, color="#888")
    ax2.set_ylim(0, 1.02)
    ax2.legend(loc="lower right", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="#283")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_learning_curves.png"), dpi=150); plt.close(fig)
    print("fig1_learning_curves.png")

# ---------- Fig 2: 参数扫描（X2）----------
def fig_param_sweep():
    rows = read_csv()
    base = {}  # 基线
    for r in rows:
        if abs(r["TARGET"] - 5000) < 1e-9 and abs(r["KAPPA"] - 1.0) < 1e-9 and abs(r["TAU_M"] - 0.5) < 1e-9 and int(r["ISI"]) == 50 and r["RESET"] == 0:
            base = r
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    sweeps = [
        ("TARGET", [3000, 5000, 7000], "TARGET", axes[0, 0]),
        ("KAPPA", [0.2, 0.5, 1.0], "KAPPA (κ)", axes[0, 1]),
        ("TAU_M", [0.2, 0.5, 1.0], "τ_m", axes[1, 0]),
        ("ISI", [0, 50, 100], "ISI steps", axes[1, 1]),
    ]
    for k, vals, xl, ax in sweeps:
        xs, ys = [], []
        for v in vals:
            for r in rows:
                ok = int(r["RESET"]) == 0
                oth = {kk: r[kk] for kk in ("TARGET", "KAPPA", "TAU_M") if kk != k}
                oth["ISI"] = r["ISI"]
                baseoth = {"TARGET": 5000.0, "KAPPA": 1.0, "TAU_M": 0.5, "ISI": 50}
                if ok and abs(r[k] - v) < 1e-9 and all(abs(oth[kk] - baseoth[kk]) < 1e-9 for kk in oth):
                    xs.append(v); ys.append(float(r["frozen_acc"]))
        ax.plot(xs, ys, "o-", color="#283", ms=5, lw=1.6)
        ax.axhline(float(base["frozen_acc"]), color="#c44", ls="--", lw=0.8)
        ax.set_xlabel(xl); ax.set_ylabel("frozen acc (1k)");
        ax.set_ylim(0, 0.8); ax.grid(alpha=0.3)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    fig.suptitle("Single-variable ablation (1k snapshot; dashed = baseline 0.648)")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(os.path.join(FIG, "fig2_param_sweep.png"), dpi=150); plt.close(fig)
    print("fig2_param_sweep.png")

# ---------- Fig 3: 噪声-性能（X6）----------
def fig_noise_perf():
    rows = read_csv()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (xcol, xl) in zip(axes, [
        ("snr_W3", "update SNR (W3)  —  higher = less noise"),
        ("eff_W3", "expected efficiency (W3)  —  higher = more signal lands"),
        ("sign_W3", "sign consistency (W3)  —  higher = more aligned direction"),
    ]):
        xs = np.array([float(r[xcol]) for r in rows]); ys = np.array([float(r["frozen_acc"]) for r in rows])
        ax.scatter(xs, ys, s=30, color="#283", zorder=3)
        # 线性拟合
        b, a = np.polyfit(xs, ys, 1)
        xx = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(xx, a + b * xx, color="#c44", lw=1.2, ls="--")
        r = np.corrcoef(xs, ys)[0, 1]
        ax.set_xlabel(xl); ax.set_ylabel("frozen acc (1k)"); ax.grid(alpha=0.3)
        ax.set_title(f"Pearson r = {r:.2f}", fontsize=10)
        # 标出对齐悖论点（低SNR高acc = 主结果重标定）
        for i, rr in enumerate(rows):
            if abs(float(rr["TARGET"]) - 5000) < 1e-9 and abs(float(rr["KAPPA"]) - 1.0) < 1e-9 and abs(float(rr["TAU_M"]) - 0.5) < 1e-9 and int(rr["ISI"]) == 50:
                ax.annotate("recalibrated\n(main result)", (xs[i], ys[i]), textcoords="offset points",
                            xytext=(-8, 10), fontsize=7, color="#c44")
    fig.suptitle("Noise (SNR / efficiency / sign) vs accuracy  —  16 configs, 1k snapshot")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(os.path.join(FIG, "fig3_noise_perf.png"), dpi=150); plt.close(fig)
    print("fig3_noise_perf.png")

# ---------- Fig 4: 单实现 vs 多实现分布（X3）----------
def fig_impl_dist():
    s1 = seed_vals(os.path.join(ROOT, "exp4/lif5_s1_t5000_k1_isi50_a15_14k/ms_eval.log"))
    s2 = seed_vals(os.path.join(ROOT, "exp4/lif5_s2_t5000_k1_isi50_a15_14k/ms_eval.log"))
    main = seed_vals(os.path.join(ROOT, "logs/x3_mainrun_ms.log"))
    groups = [("main run", main), ("seed-1 rerun", s1), ("seed-2 rerun", s2)]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i, (name, vals) in enumerate(groups):
        if not vals:
            continue
        x = i + 1
        ax.scatter(np.full(len(vals), x) + np.random.default_rng(i).uniform(-0.12, 0.12, len(vals)),
                   vals, s=26, color="#47a", zorder=3, alpha=0.85)
        ax.errorbar(x, np.mean(vals), yerr=np.std(vals), fmt="o", color="#283", ms=5, capsize=4, lw=1.4)
        ax.text(x, max(vals) + 0.015, f"{np.mean(vals):.3f}±{np.std(vals):.3f}", ha="center", fontsize=8)
    ax.axhline(0.8, color="#c44", ls="--", lw=0.9)
    ax.text(0.05, 0.805, "acceptance bar 0.8", fontsize=8, color="#c44", transform=ax.get_yaxis_transform())
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("frozen test acc (5 seeds × n=500)")
    ax.set_ylim(0.75, 0.92); ax.grid(alpha=0.3)
    fig.suptitle("Single-implementation spread vs multi-seed mean")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(os.path.join(FIG, "fig4_impl_dist.png"), dpi=150); plt.close(fig)
    print("fig4_impl_dist.png")

if __name__ == "__main__":
    fig_learning_curves()
    fig_param_sweep()
    fig_noise_perf()
    fig_impl_dist()
    print("done ->", FIG)
