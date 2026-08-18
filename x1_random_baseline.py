#!/usr/bin/env python3
"""X1: 纯随机权重控制基线 —— 未训练的随机初始化权重，跑标准多 seed 冻结评估。
意义：证明 0.877 完全来自学习，而非初始化先验（基线应 ~0.10 随机猜测）。"""
import sys, numpy as np

# 旗舰配置 argv（与主跑同拓扑/同协议，只是不训练）
sys.argv = ["mnist_shared.py", "0", "14000", "1.0", "200", "1.5e-8", "3000", "30",
            "5000", "1000", "0.02", "1.0", "0", "0", "50", "0.5"]
import mnist_shared as m

print(f"随机初始化权重：P=±{m.P_INIT}（SIGN 随机 ±1, 数值常数 {m.P_INIT}），未训练")
print("评估协议：5 seed × n=500 冻结（τ_m=0.5/ISI=100 读出）\n")
res = []
for seed in [123, 1, 2, 3, 4]:
    m.u[:] = 0; m.r_est[:] = 0; m.f_est[:] = 0; m.E1[:] = 0; m.E2[:] = 0; m.E3[:] = 0
    idx = np.random.default_rng(seed).choice(10000, 300, replace=False)
    hits = 0
    for ii in idx:
        x = m.te_flat[ii]; y = m.tel[ii]
        for _ in range(m.steps):
            m.spiking_step(x, m.y_onehot[y], learn=False)
        hits += int(np.argmax(m.f_est) == y)
        zx = np.zeros(784); zy = np.zeros(m.NOUT)
        for _ in range(100):
            m.spiking_step(zx, zy, learn=False)
    res.append(hits / 300)
    print(f"  seed {seed:>3}: {res[-1]:.3f}", flush=True)
print(f"\nX1 随机权重基线: mean={np.mean(res):.3f} ± {np.std(res):.3f}")
