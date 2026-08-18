"""Multi-readout frozen evaluation (LIF-capable), appends to mnist_mf_pulse_eval.csv.
Usage: python3 eval_mf_batch.py [--tau_m=0.5] [--isi_steps=100] [--kappa=1.0] [--sample_t=1.0] [--n=1000] <ckpt1> [ckpt2 ...]
Readouts (all frozen, same protocol as training):
  pulse_final : argmax(f_est/TAU_F) at end of sample
  pulse_avg   : argmax(mean f_est/TAU_F over second half of sample)
  pulse_spike : argmax of spike counts over sample
  meanfield   : analytic rates with KAPPA self-consistency (8 iters)
"""
import sys
import numpy as np

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
OPTS = {}
for a in sys.argv[1:]:
    if a.startswith("--"):
        k, v = a[2:].split("=")
        OPTS[k] = float(v)
TAU_M = OPTS.get("tau_m", 0.5)
ISI = int(OPTS.get("isi_steps", 100))
KAPPA = OPTS.get("kappa", 1.0)
SAMPLE_T = OPTS.get("sample_t", 1.0)
N_IMG = int(OPTS.get("n", 1000))
CKPTS = ARGS if ARGS else ["mnist_checkpoint.npz"]

sys.argv = ["mnist_shared.py", "0", "500", str(SAMPLE_T), "200", "3e-8", "3000", "30", "1000",
            str(N_IMG), "0.02", str(KAPPA), "0", "0", str(ISI), str(TAU_M)]
import mnist_shared as m

TH = m.THETA
LAM = 1.0 / TAU_M if TAU_M > 0 else 0.0


def phi(x, lam):
    if lam <= 0:
        return np.maximum(x, 0)
    return np.where(x > 0, lam / np.log(1.0 + lam / np.maximum(x, 1e-9)), 0.0)


def rates_of(P, x, KAPPA):
    a_in = np.zeros(m.N_NEURONS)
    a_in[m.OFF_IN:m.OFF_IN + 784] = m.R_IN * x
    a_in[m.OFF_B1] = m.BIAS_RATE
    a_in[m.OFF_B2] = m.BIAS_RATE
    a_in[m.OFF_B3] = m.BIAS_RATE
    z1 = np.bincount(m.POST1 - m.OFF_F, m.SIGN[m.KIDX1] * P[m.KIDX1] / TH * a_in[m.PRE1], minlength=m.N_F)
    a1 = phi(z1 - LAM, LAM)
    p1v = np.zeros(m.N_NEURONS)
    p1v[m.F] = a1
    p1v[m.OFF_B2] = m.BIAS_RATE
    z2 = np.bincount(m.POST2 - m.OFF_FC, m.SIGN[m.G1:m.G2] * P[m.G1:m.G2] / TH * p1v[m.PRE2], minlength=m.NFC)
    a2 = phi(z2 - LAM, LAM)
    a2b = np.concatenate([a2, [m.BIAS_RATE]])
    z3 = np.bincount(m.POST3 - m.OFF_OUT, m.SIGN[m.G2:m.G3] * P[m.G2:m.G3] / TH * a2b[m.PRE3 - m.OFF_FC], minlength=10)
    a3 = phi(z3 - LAM, LAM)
    if KAPPA > 0:
        S = a3.sum()
        for _ in range(8):
            a3 = np.maximum(phi(z3 - KAPPA * S, LAM), 0)
            S = a3.sum()
    return a3


def frozen_acc(P, n_img, seed=123):
    m.P[:] = P
    idx = np.random.default_rng(seed).choice(10000, n_img, replace=False)
    hits_f, hits_a, hits_s = 0, 0, 0
    half = m.steps // 2
    for ii in idx:
        x = m.te_flat[ii]
        y = m.tel[ii]
        spk = np.zeros(m.NOUT)
        f_avg = np.zeros(m.NOUT)
        for st in range(m.steps):
            m.spiking_step(x, m.y_onehot[y], learn=False)
            spk += m.n_out_last
            if st >= half:
                f_avg += m.f_est / m.TAU_F
        hits_f += int(np.argmax(m.f_est / m.TAU_F) == y)
        hits_a += int(np.argmax(f_avg) == y)
        hits_s += int(np.argmax(spk) == y)
        if ISI > 0:
            zx = np.zeros(784)
            zy = np.zeros(m.NOUT)
            for _ in range(ISI):
                m.spiking_step(zx, zy, learn=False)
    return hits_f / n_img, hits_a / n_img, hits_s / n_img


import csv
import os

OUT = "mnist_mf_pulse_eval.csv"
new = not os.path.exists(OUT)
f = open(OUT, "a", newline="")
w = csv.writer(f)
if new:
    w.writerow(["ckpt", "tau_m", "isi_steps", "kappa", "pulse_final", "pulse_avg", "pulse_spike", "meanfield"])

for ckpt in CKPTS:
    z = np.load(ckpt)
    P = z["P"].copy()
    if "SIGN" in z:
        m.SIGN[:] = z["SIGN"]
    m.u[:] = 0; m.r_est[:] = 0; m.f_est[:] = 0; m.E1[:] = 0; m.E2[:] = 0; m.E3[:] = 0
    m.KAPPA = KAPPA
    pf, pa, ps = frozen_acc(P, N_IMG)
    m.u[:] = 0; m.r_est[:] = 0; m.f_est[:] = 0; m.E1[:] = 0; m.E2[:] = 0; m.E3[:] = 0
    idx = np.random.default_rng(123).choice(10000, N_IMG, replace=False)
    hits = 0
    for ii in idx:
        a3 = rates_of(P, m.te_flat[ii], KAPPA)
        hits += int(np.argmax(a3) == m.tel[ii])
    mf = hits / N_IMG
    print(f"{ckpt}  TAU_M={TAU_M}  ISI={ISI}  KAPPA={KAPPA}  pulse_final={pf:.4f}  "
          f"pulse_avg={pa:.4f}  pulse_spike={ps:.4f}  meanfield={mf:.4f}", flush=True)
    w.writerow([ckpt, TAU_M, ISI, KAPPA, round(pf, 4), round(pa, 4), round(ps, 4), round(mf, 4)])
    f.flush()
f.close()