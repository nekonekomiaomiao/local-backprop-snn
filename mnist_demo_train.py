#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST 局部在线 BP 概率突触 SNN —— 论文演示训练器
==================================================
单一可执行文件：双击即开始训练，实时可视化训练进度。

协议（与论文正文一致）：
  LIF 膜泄漏（τ_m = 0.5，论文旧版 §4.5）+ 样本间静默间隔（ISI = 50 步）
  + 输出层重标定（TARGET=5000, κ=1.0 全局抑制池）—— 多 seed 冻结验收 0.877±0.007 的旗舰配置。

用法（单文件可执行，无 Python 环境依赖）：
  ./mnist_train_demo                  # 默认：seed 0，N=3000 样本（约 25 分钟）
  ./mnist_train_demo --samples 1000   # 缩短演示
  ./mnist_train_demo --config uncal   # 对照协议：未重标定（TARGET=1000, κ=0.2）
  ./mnist_train_demo --config reset   # 对照协议：硬清零 reset
  ./mnist_train_demo --seed 1 --out /path/to/dir --no-gui

输出（写入 --out，默认当前目录）：
  demo_checkpoint.npz      训练好的权重（P/SIGN，与 exp4/*.npz 同格式，可用 eval_multiseed.py 验收）
  demo_training_curves.png 训练曲线（loss / train-acc / frozen test acc 三板）
  demo_summary.txt         配置 + 最终多 seed 冻结 acc + 结果说明
"""
import argparse
import gzip
import os
import sys
import time
import urllib.request

import numpy as np

# ---------- 数据自给（文件夹随包自带 MNIST，点击即用零下载；--download 为逃生通道） ----------
_MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_MNIST_FILES = {
    "train-images-idx3-ubyte.gz": 9912422,
    "train-labels-idx1-ubyte.gz": 28881,
    "t10k-images-idx3-ubyte.gz": 1648877,
    "t10k-labels-idx1-ubyte.gz": 4542,
}
_DATA_DIR_HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_data")


def _ensure_data():
    """冻结打包模式下数据已内置；源码模式要求脚本同目录的 mnist_data/ 随包提供。"""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return
    missing = [f for f in _MNIST_FILES
               if not os.path.isfile(os.path.join(_DATA_DIR_HERE, f))]
    if not missing:
        return
    if ARGS.download:
        print("[data] 正在下载 MNIST 数据集到", _DATA_DIR_HERE)
        os.makedirs(_DATA_DIR_HERE, exist_ok=True)
        for name, want in _MNIST_FILES.items():
            dest = os.path.join(_DATA_DIR_HERE, name)
            if os.path.isfile(dest) and os.path.getsize(dest) == want:
                continue
            print(f"[data]   {name}  ({want / 1e6:.1f} MB) ...", flush=True)
            urllib.request.urlretrieve(_MNIST_URL + name, dest)
            got = os.path.getsize(dest)
            if got != want:
                raise RuntimeError(f"{name} 下载校验失败：期望 {want} 字节，实际 {got}")
        print("[data] 完成。", flush=True)
    else:
        print("[data] 错误：缺少 MNIST 数据文件：", ", ".join(missing))
        print("       请把整个随包文件夹一并拷贝（含 mnist_data/ 子文件夹），脚本与数据必须同目录；")
        print("       或者在联网机器上改用 --download 让脚本自动下载。")
        sys.exit(2)

# ---------- 参数 ----------
CONFIGS = {
    "recal":  dict(TARGET=5000.0, KAPPA=1.0, TAU_M=0.5, ISI=50, RESET=0,
                   name="LIF+ISI 重标定（论文旗舰）", name_en="Recalibrated LIF+ISI (paper flagship)"),
    "uncal":  dict(TARGET=1000.0, KAPPA=0.2, TAU_M=0.5, ISI=50, RESET=0,
                   name="LIF+ISI 未标定（对照 0.501）", name_en="Uncalibrated LIF+ISI (control)"),
    "reset":  dict(TARGET=1000.0, KAPPA=0.2, TAU_M=0.0, ISI=0,  RESET=1,
                   name="硬清零 reset（对照 0.824）", name_en="Hard-reset protocol (control)"),
    "if":     dict(TARGET=1000.0, KAPPA=0.2, TAU_M=0.0, ISI=0,  RESET=0,
                   name="IF 无泄漏（对照 0.348）", name_en="IF leak-free (control)"),
}


def parse_args():
    ap = argparse.ArgumentParser(description="MNIST 局部在线 BP SNN 论文演示训练器")
    ap.add_argument("--samples", type=int, default=3000, help="训练样本数（默认 3000，约 25 分钟；1000 ≈ 8 分钟）")
    ap.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    ap.add_argument("--config", choices=list(CONFIGS), default="recal",
                    help="协议配置（默认 recal=论文旗舰 0.877 配置）")
    ap.add_argument("--out", default=".", help="输出目录（checkpoint/png/summary）")
    ap.add_argument("--eval-every", type=int, default=500, help="周期冻结评估间隔（默认 500 样本）")
    ap.add_argument("--no-gui", action="store_true", help="强制无窗口模式（仅终端进度 + 结尾 PNG）")
    ap.add_argument("--download", action="store_true",
                    help="逃生通道：仅在随包数据缺失且网络可用时，自动下载 MNIST 到脚本同目录 mnist_data/")
    return ap.parse_args()


ARGS = parse_args()
CFG = CONFIGS[ARGS.config]
ARGS.out = os.path.abspath(ARGS.out)
os.makedirs(ARGS.out, exist_ok=True)

# 数据自检：随包 mnist_data/ 必须在脚本同目录（缺失则提示，--download 仅逃生用）
_ensure_data()

# ---------- 导入训练内核（argv 先行，mnist_shared 在 import 时读参数并加载数据） ----------
sys.argv = ["mnist_shared.py", str(ARGS.seed), str(ARGS.samples), "1.0", "200", "1.5e-8",
            "3000", "30", str(CFG["TARGET"]), "100", "0.02", str(CFG["KAPPA"]), "0",
            str(CFG["RESET"]), str(CFG["ISI"]), str(CFG["TAU_M"]), "0.2"]
import mnist_shared as m  # noqa: E402

# ---------- 可视化（有显示则开窗口，否则 Agg 存图） ----------
import matplotlib
_GUI = False
if not ARGS.no_gui and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    try:
        import tkinter  # noqa: F401
        matplotlib.use("TkAgg")
        _GUI = True
    except Exception:
        matplotlib.use("Agg")
        _GUI = False
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_EVAL = 200          # 周期/最终冻结评估的测试样本数
EVAL_SEEDS = (123, 1, 2)   # 最终多 seed 冻结验收

print("=" * 74)
print(" MNIST 脉冲 SNN —— 局部在线反向传播（论文演示）")
print("=" * 74)
print(f" 配置   : {CFG['name']}   (TARGET={CFG['TARGET']:g}, κ={CFG['KAPPA']}, τ_m={CFG['TAU_M']}, ISI={CFG['ISI']})")
print(f" 数据   : MNIST 784 -> 共享卷积(5x5x32, str2 64 参数) -> FC128 -> 10   随机初始化，无权重注入")
print(f" 训练   : {ARGS.samples} 样本 × {m.steps} 步/样本, α=1.5e-8, 在线局部学习（论文 SDE 框架）")
print(f" 输出   : {ARGS.out}/  (checkpoint / 训练曲线 PNG / summary)")
print("-" * 74)

# ---------- 状态初始化 ----------
m.u[:] = 0; m.r_est[:] = 0; m.f_est[:] = 0
m.E1[:] = 0; m.E2[:] = 0; m.E3[:] = 0
m.P[:] = np.full(m.G3, m.P_INIT)
m.P[m.G2 + np.arange(m.NOUT) * (m.NFC + 1)] = 0.6

losses = np.zeros(ARGS.samples)
accs = np.zeros(ARGS.samples)
steps_plt = []
loss_plt, acc_plt = [], []
eval_steps, test_plt = [], []   # 冻结评估独立计时，与 200 块进度解耦


def frozen_test(seed, n=None):
    """冻结权重、多 seed、τ_m/ISI 读出协议下的测试准确率（项目验收协议的精简版）。"""
    n = n or N_EVAL
    rng = np.random.default_rng(seed)
    m.u[:] = 0; m.r_est[:] = 0; m.f_est[:] = 0
    m.E1[:] = 0; m.E2[:] = 0; m.E3[:] = 0
    idx = rng.choice(10000, n, replace=False)
    hits = 0
    for ii in idx:
        if CFG["RESET"]:
            m.reset_state()
        x = m.te_flat[ii]; y = m.tel[ii]
        for _ in range(m.steps):
            m.spiking_step(x, m.y_onehot[y], learn=False)
        if m.ISI_STEPS > 0 and not CFG["RESET"]:
            zx = np.zeros(784); zy = np.zeros(m.NOUT)
            for _ in range(m.ISI_STEPS):
                m.spiking_step(zx, zy, learn=False)
        hits += int(np.argmax(m.f_est / m.TAU_F) == y)
    return hits / n


def redraw(title):
    plt.clf()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    fig.suptitle(title, fontsize=11)
    ax = axes[0]
    ax.plot(steps_plt, loss_plt, "-", color="#c44", lw=1.6)
    ax.set_title("Normalized MSE  (last-100 rolling)"); ax.set_xlabel("sample"); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(steps_plt, acc_plt, "-", color="#47a", lw=1.6)
    ax.set_ylim(0, 1.02); ax.set_title("Train accuracy  (last-100 rolling)"); ax.set_xlabel("sample"); ax.grid(alpha=0.3)
    ax = axes[2]
    if eval_steps:
        ax.plot(eval_steps, test_plt, "-o", color="#283", lw=1.6, ms=3)
    else:
        ax.text(0.5, 0.5, "no frozen eval yet\n(needs samples >= eval-every)",
                ha="center", va="center", transform=ax.transAxes, color="#888")
    ax.set_ylim(0, 1.02); ax.set_title(f"Frozen test accuracy  (n={N_EVAL}, seed 123)"); ax.set_xlabel("sample"); ax.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.pause(0.01) if _GUI else None


t0 = time.time()
for smp in range(ARGS.samples):
    if CFG["RESET"]:
        m.reset_state()
    ii = m.order[smp]
    x = m.tr_flat[ii]; y = m.trl[ii]; yv = m.y_onehot[y]
    for _ in range(m.steps):
        m.spiking_step(x, yv, learn=True)
    if m.ISI_STEPS > 0:
        zx = np.zeros(784); zy = np.zeros(m.NOUT)
        for _ in range(m.ISI_STEPS):
            m.spiking_step(zx, zy, learn=False)
    f = m.f_est / m.TAU_F
    losses[smp] = float(np.mean(0.5 * ((f - m.TARGET * yv) / m.TARGET) ** 2))
    accs[smp] = float(np.argmax(f) == y)
    if (smp + 1) % ARGS.eval_every == 0:
        ta = frozen_test(123)
        eval_steps.append(smp + 1)
        test_plt.append(ta)
        print(f"  [eval @{smp + 1:>5}] frozen test_acc(n={N_EVAL}) = {ta:.3f}   "
              f"({time.time() - t0:>6.0f}s)", flush=True)
    if (smp + 1) % 200 == 0:
        steps_plt.append(smp + 1)
        loss_plt.append(losses[max(0, smp - 99):smp + 1].mean())
        acc_plt.append(accs[max(0, smp - 99):smp + 1].mean())
        bar = "#" * int(round(acc_plt[-1] * 40))
        print(f"  {smp + 1:>6}/{ARGS.samples}  loss={loss_plt[-1]:.4f}  train_acc={acc_plt[-1]:.3f}  [{bar}]  "
              f"({time.time() - t0:>6.0f}s)", flush=True)
        redraw(f"{CFG['name_en']}   |   sample {smp + 1}/{ARGS.samples}")

# ---------- 最终评估 ----------
print("-" * 74)
print(" 最终多 seed 冻结验收（项目验收协议精简版：3 seed × n=%d，τ_m=0.5/ISI=100 读出）" % N_EVAL)
seeds = EVAL_SEEDS if not CFG["RESET"] else (123,)
res = [frozen_test(s) for s in seeds]
mean = float(np.mean(res)); std = float(np.std(res))
for s, r in zip(seeds, res):
    print(f"   seed {s:>4}: {r:.3f}")
print(f"   mean = {mean:.3f} ± {std:.3f}")

# ---------- 保存 ----------
ckpt = os.path.join(ARGS.out, "demo_checkpoint.npz")
np.savez(ckpt, P=m.P, SIGN=m.SIGN, R_IN=m.R_IN, ALPHA=m.ALPHA, SEED=m.SEED,
         PRE1=m.PRE1, POST1=m.POST1, KIDX1=m.KIDX1, PRE2=m.PRE2, POST2=m.POST2,
         PRE3=m.PRE3, POST3=m.POST3)
png = os.path.join(ARGS.out, "demo_training_curves.png")
redraw(f"{CFG['name_en']}   |   final frozen test acc = {mean:.3f} ± {std:.3f}   |   {ARGS.samples} samples")
if not _GUI:
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.close()
summary = os.path.join(ARGS.out, "demo_summary.txt")
with open(summary, "w", encoding="utf-8") as fh:
    fh.write(f"config      : {CFG['name']}\n")
    fh.write(f"seed        : {ARGS.seed}   samples: {ARGS.samples}   elapsed: {time.time() - t0:.0f}s\n")
    fh.write(f"protocol    : LIF tau_m={m.TAU_M}, ISI={m.ISI_STEPS}, TARGET={m.TARGET}, kappa={m.KAPPA}, "
             f"reset={CFG['RESET']}, alpha={m.ALPHA}\n")
    fh.write(f"frozen-test : mean={mean:.3f} std={std:.3f}  seeds{list(seeds)} n={N_EVAL}\n")
    fh.write(f"checkpoint  : {ckpt}\n")
    fh.write(f"curve figure: {png}\n")
    fh.write("full-acceptance protocol (paper): eval_multiseed.py (5 seeds x n=500, tau_m=0.5, ISI=100)\n")
print("-" * 74)
print(f" 完成！{ARGS.samples} 样本用时 {time.time() - t0:.0f}s")
print(f" 权重   : {ckpt}")
print(f" 曲线图 : {png}")
print(f" 摘要   : {summary}")
print("=" * 74)