#!/usr/bin/env python3
"""X10: event-driven 事件数度量 —— 对训练好的主跑权重(0.877)统计推理时每样本事件数、
各层分解、稀疏性、与稠密等效计算的"事件驱动节省"。
事件 = 突触传递脉冲(逐边)；发放 = 神经元输出脉冲(逐神经元)。"""
import sys
import numpy as np

sys.argv = ["mnist_shared.py", "0", "14000", "1.0", "200", "1.5e-8", "3000", "30",
            "5000", "1000", "0.02", "1.0", "0", "0", "50", "0.5"]
import mnist_shared as m

CKPT = "exp4/lif5_ckpts/ckpt_14000.npz"
z = np.load(CKPT)
m.P[:] = z["P"].copy()
if "SIGN" in z:
    m.SIGN[:] = z["SIGN"]
print(f"已加载主跑 ckpt：P len={len(m.P)}，SIGN random={np.any(m.SIGN==-1)}")

# 等价稠密 MAC / 样本（供稀疏对比）
MAC_DENSE = 784 * (m.N_F) + m.N_F * m.NFC + m.NFC * m.NOUT
print(f"等价稠密每样本 MAC 数：{MAC_DENSE:,}")

N_SAMPLE = 500
per_sample_ev = []   # 突触事件/样本 (k1+k2+k3)
per_sample_sp = []   # 神经元发放/样本 (n1+n2+n_out)
per_sample_in = []   # 输入事件/样本
per_class = {c: [] for c in range(10)}
active_edges = []

def count_step(x_vec):
    """复刻 mnist_shared.spiking_step 的推理路径并计数（learn=False）。"""
    if m.TAU_M > 0.0:
        m.u *= np.exp(-m.DT / m.TAU_M)
    pre_spikes = np.zeros(m.N_NEURONS)
    pre_spikes[m.OFF_IN:m.OFF_IN + 784] = m.RNG.poisson(m.R_IN * x_vec * m.DT)
    pre_spikes[m.OFF_B1] = m.RNG.poisson(m.BIAS_RATE * m.DT)
    pre_spikes[m.OFF_B2] = m.RNG.poisson(m.BIAS_RATE * m.DT)
    pre_spikes[m.OFF_B3] = m.RNG.poisson(m.BIAS_RATE * m.DT)
    ev_in = int(pre_spikes[m.OFF_IN:m.OFF_IN + 784].sum())

    k1 = m.RNG.binomial(pre_spikes[m.PRE1].astype(np.int64), m.P[m.KIDX1])
    m.u[m.F] += np.bincount(m.POST1 - m.OFF_F, m.SIGN[m.KIDX1] * k1, minlength=m.N_F)
    m.E1 += -m.E1 * m.DT / m.TAU_E + pre_spikes[m.PRE1]
    n1 = np.floor(m.u[m.F] / m.THETA).clip(0.0, None).astype(np.int64)
    m.u[m.F] -= n1 * m.THETA
    m.r_est[m.F] += -m.r_est[m.F] * m.DT / m.TAU_R + n1
    ev_k1 = int(k1.sum()); sp_n1 = int(n1.sum())

    k2 = m.RNG.binomial(pre_spikes[m.PRE2].astype(np.int64), m.P[m.G1:m.G2])
    m.u[m.FC] += np.bincount(m.POST2 - m.OFF_FC, m.SIGN[m.G1:m.G2] * k2, minlength=m.NFC)
    m.E2 += -m.E2 * m.DT / m.TAU_E + pre_spikes[m.PRE2]
    n2 = np.floor(m.u[m.FC] / m.THETA).clip(0.0, None).astype(np.int64)
    m.u[m.FC] -= n2 * m.THETA
    m.r_est[m.FC] += -m.r_est[m.FC] * m.DT / m.TAU_R + n2
    ev_k2 = int(k2.sum()); sp_n2 = int(n2.sum())

    k3 = m.RNG.binomial(pre_spikes[m.PRE3].astype(np.int64), m.P[m.G2:m.G3])
    if m.KAPPA > 0.0:
        out_rate = m.f_est / m.TAU_F
        m.u[m.OUT] -= m.KAPPA * np.sum(out_rate) * m.DT
    m.u[m.OUT] += np.bincount(m.POST3 - m.OFF_OUT, m.SIGN[m.G2:m.G3] * k3, minlength=m.NOUT)
    m.E3 += -m.E3 * m.DT / m.TAU_E + pre_spikes[m.PRE3]
    n_out = np.floor(m.u[m.OUT] / m.THETA).clip(0.0, None).astype(np.int64)
    m.u[m.OUT] -= n_out * m.THETA
    m.r_est[m.OUT] += -m.r_est[m.OUT] * m.DT / m.TAU_R + n_out
    m.f_est += -m.f_est * m.DT / m.TAU_F + n_out
    ev_k3 = int(k3.sum()); sp_n3 = int(n_out.sum())
    return ev_in, ev_k1, ev_k2, ev_k3, sp_n1, sp_n2, sp_n3, (k1 > 0), (k2 > 0), (k3 > 0)

rng = np.random.default_rng(0)
idx = rng.choice(10000, N_SAMPLE, replace=False)
for n, ii in enumerate(idx):
    x = m.te_flat[ii]; y = int(m.tel[ii])
    m.u[:] = 0; m.r_est[:] = 0; m.f_est[:] = 0; m.E1[:] = 0; m.E2[:] = 0; m.E3[:] = 0
    ev_in = ev_k1 = ev_k2 = ev_k3 = sp1 = sp2 = sp3 = 0
    n_total_edges = len(m.PRE1) + len(m.PRE2) + len(m.PRE3)
    act1 = np.zeros(len(m.PRE1), dtype=bool); act2 = np.zeros(len(m.PRE2), dtype=bool); act3 = np.zeros(len(m.PRE3), dtype=bool)
    for _ in range(m.steps):
        a, b, c, d, e_, f_, g, A1, A2, A3 = count_step(x)
        ev_in += a; ev_k1 += b; ev_k2 += c; ev_k3 += d; sp1 += e_; sp2 += f_; sp3 += g
        act1 |= A1; act2 |= A2; act3 |= A3
    zx = np.zeros(784)
    for _ in range(m.ISI_STEPS):
        a, b, c, d, e_, f_, g, A1, A2, A3 = count_step(zx)
        ev_in += a; ev_k1 += b; ev_k2 += c; ev_k3 += d; sp1 += e_; sp2 += f_; sp3 += g
        act1 |= A1; act2 |= A2; act3 |= A3
    per_sample_ev.append(ev_k1 + ev_k2 + ev_k3)
    per_sample_sp.append(sp1 + sp2 + sp3)
    per_sample_in.append(ev_in)
    per_class[y].append(ev_k1 + ev_k2 + ev_k3)
    active_edges.append(int(act1.sum()) + int(act2.sum()) + int(act3.sum()))

pev = np.array(per_sample_ev); psp = np.array(per_sample_sp); pin = np.array(per_sample_in)
pae = np.array(active_edges); n_edges = len(m.PRE1) + len(m.PRE2) + len(m.PRE3)
print(f"\n每样本 突触事件(传递脉冲): mean={pev.mean():.0f} ± {pev.std():.0f}  [min {pev.min()} max {pev.max()}]")
print(f"每样本 神经元发放:        mean={psp.mean():.0f} ± {psp.std():.0f}")
print(f"每样本 输入事件:          mean={pin.mean():.0f} ± {pin.std():.0f}")
print(f"\n每样本 激活突触(唯一活跃边): mean={pae.mean():.0f} ± {pae.std():.0f}  (占总边 {n_edges} 的 {pae.mean()/n_edges*100:.1f}%)")
print(f"每激活边平均事件数: {pev.mean()/pae.mean():.1f} (反映率编码的时间冗余)")
# 有效省算力：事件驱动只需算激活边（而非全连接稠密 MAC）
dmt = MAC_DENSE * N_SAMPLE
print(f"\n有效省算力: 稠密每样本 {MAC_DENSE:,} MAC vs 激活边 {pae.mean():.0f}/样本 -> 只算激活边 {dmt/(pae.mean()*N_SAMPLE):.1f}x 更省")
print(f"\n按类别 突触事件/样本:")
for c in range(10):
    arr = np.array(per_class[c])
    print(f"  类{c}: {arr.mean():.0f} ± {arr.std():.0f} (n={len(arr)})")
