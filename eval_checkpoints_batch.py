"""Batch-evaluate saved mnist_checkpoint.npz files: frozen pulse acc + mean-field acc with KAPPA.
Usage: python3 eval_checkpoints_batch.py <ckpt1> [ckpt2 ...]
"""
import sys
import numpy as np

CKPTS = sys.argv[1:] if len(sys.argv) > 1 else ["mnist_checkpoint.npz"]

sys.argv = ["mnist_shared.py", "0", "500", "1.0", "200", "3e-8", "3000", "30", "1000", "100", "0.02", "0.2", "0"]
import mnist_shared as m

TH = m.THETA


def rates_of(P, x, KAPPA):
    a_in = np.zeros(m.N_NEURONS)
    a_in[m.OFF_IN:m.OFF_IN + 784] = m.R_IN * x
    a_in[m.OFF_B1] = m.BIAS_RATE
    a_in[m.OFF_B2] = m.BIAS_RATE
    a_in[m.OFF_B3] = m.BIAS_RATE
    z1 = np.bincount(m.POST1 - m.OFF_F, m.SIGN[m.KIDX1] * P[m.KIDX1] / TH * a_in[m.PRE1], minlength=m.N_F)
    a1 = np.maximum(z1, 0)
    p1v = np.zeros(m.N_NEURONS)
    p1v[m.F] = a1
    p1v[m.OFF_B2] = m.BIAS_RATE
    z2 = np.bincount(m.POST2 - m.OFF_FC, m.SIGN[m.G1:m.G2] * P[m.G1:m.G2] / TH * p1v[m.PRE2], minlength=m.NFC)
    a2 = np.maximum(z2, 0)
    a2b = np.concatenate([a2, [m.BIAS_RATE]])
    z3 = np.bincount(m.POST3 - m.OFF_OUT, m.SIGN[m.G2:m.G3] * P[m.G2:m.G3] / TH * a2b[m.PRE3 - m.OFF_FC], minlength=10)
    a3 = np.maximum(z3, 0)
    if KAPPA > 0:
        S = a3.sum()
        for _ in range(8):
            a3 = np.maximum(z3 - KAPPA * S, 0)
            S = a3.sum()
    return a3


def frozen_pulse_acc(P, n_img=1000, seed=123):
    m.P[:] = P
    idx = np.random.default_rng(seed).choice(10000, n_img, replace=False)
    hits = 0
    for ii in idx:
        x = m.te_flat[ii]
        y = m.tel[ii]
        for _ in range(m.steps):
            m.spiking_step(x, m.y_onehot[y], learn=False)
        hits += int(np.argmax(m.f_est / m.TAU_F) == y)
    return hits / n_img


def meanfield_acc(P, n_img=1000, seed=123, KAPPA=0.0):
    idx = np.random.default_rng(seed).choice(10000, n_img, replace=False)
    hits = 0
    for ii in idx:
        a3 = rates_of(P, m.te_flat[ii], KAPPA)
        hits += int(np.argmax(a3) == m.tel[ii])
    return hits / n_img


def infer_kappa(ckpt):
    name = ckpt.lower()
    if "k0p4" in name: return 0.4
    if "k0p3" in name: return 0.3
    if "k0p2" in name: return 0.2
    if "k0p15" in name: return 0.15
    if "k0p1" in name: return 0.1
    if "k0p05" in name: return 0.05
    return 0.0


for ckpt in CKPTS:
    z = np.load(ckpt)
    P = z["P"].copy()
    if "SIGN" in z:
        m.SIGN[:] = z["SIGN"]
    m.u[:] = 0; m.r_est[:] = 0; m.f_est[:] = 0; m.E1[:] = 0; m.E2[:] = 0; m.E3[:] = 0
    kappa = infer_kappa(ckpt)
    m.KAPPA = kappa
    try:
        fp = frozen_pulse_acc(P)
        m.u[:] = 0; m.r_est[:] = 0; m.f_est[:] = 0; m.E1[:] = 0; m.E2[:] = 0; m.E3[:] = 0
        mf = meanfield_acc(P, KAPPA=kappa)
    except Exception as e:
        print(f"{ckpt}: ERROR {e}")
        continue
    print(f"{ckpt}  KAPPA={kappa}  frozen_pulse={fp:.4f}  meanfield={mf:.4f}", flush=True)
