import sys
import time
import numpy as np

LR = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-8
N = int(sys.argv[2]) if len(sys.argv) > 2 else 50000
KAPPA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
sys.argv = ["mnist_shared.py", str(SEED), "2000", "1.0", "200", "3e-8", "3000", "30", "1000"]
import mnist_shared as m

TH = m.THETA


def rates_of(P, x):
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
    if KAPPA > 0.0:
        S = a3.sum()
        for _ in range(8):
            a3 = np.maximum(z3 - KAPPA * S, 0)
            S = a3.sum()
    return a1, a2, a3, a_in, p1v


def ana_grad(P, x, yv):
    a1, a2, a3, a_in, p1v = rates_of(P, x)
    d3 = a3 - m.TARGET * yv
    w3 = (m.SIGN[m.G2:m.G3] * P[m.G2:m.G3] / TH).reshape(10, m.NFC + 1)
    d2 = (a2 > 0) * (w3[:, 1:].T @ d3)
    d_f = np.bincount(m.PRE2 - m.OFF_F, m.SIGN[m.G1:m.G2] * P[m.G1:m.G2] / TH * d2[m.POST2 - m.OFF_FC], minlength=m.N_F + 1)[:m.N_F]
    d1 = (a1 > 0)[m.POST1 - m.OFF_F] * d_f[m.POST1 - m.OFF_F]
    s = m.SIGN
    g = np.zeros(m.G3)
    g[:m.G1] = np.bincount(m.KIDX1, s[m.KIDX1] * a_in[m.PRE1] * d1, minlength=m.N_S1) / TH
    g[m.G1:m.G2] = s[m.G1:m.G2] * p1v[m.PRE2] * d2[m.POST2 - m.OFF_FC] / TH
    a2b = np.concatenate([a2, [m.BIAS_RATE]])
    g[m.G2:m.G3] = s[m.G2:m.G3] * a2b[m.PRE3 - m.OFF_FC] * d3[m.POST3 - m.OFF_OUT] / TH
    return g


def eval_acc(P, n=300):
    idx = np.random.default_rng(SEED + 1).choice(10000, n, replace=False)
    hits = 0
    for ii in idx:
        a3 = rates_of(P, m.te_flat[ii])[2]
        hits += int(np.argmax(a3) == m.tel[ii])
    return hits / n


rng = np.random.default_rng(SEED)
perm = rng.permutation(60000)
order = np.tile(perm, N // 60000 + 1)[:N]
P = np.full(m.G3, m.P_INIT)
P[m.G2 + np.arange(m.NOUT) * (m.NFC + 1)] = 0.6

print(f"mean-field GD  shared  LR={LR}  N={N}  KAPPA={KAPPA}  seed={SEED}", flush=True)
t0 = time.time()
for s in range(N):
    x = m.tr_flat[order[s]]
    yv = m.y_onehot[m.trl[order[s]]]
    g = ana_grad(P, x, yv)
    P -= LR * g
    P = np.clip(P, 1e-6, 1 - 1e-6)
    if (s + 1) % 2500 == 0:
        a = eval_acc(P)
        print(f"  {s + 1}/{N}  test {a:.3f}  ({time.time() - t0:.0f} s)", flush=True)
    if (s + 1) % 25000 == 0:
        np.savez("meanfield_checkpoint.npz", P=P, SIGN=m.SIGN, R_IN=m.R_IN, KAPPA=KAPPA, LR=LR, SEED=SEED, step=s + 1)
        print(f"  [meanfield checkpoint saved @ {s + 1}]", flush=True)
a = eval_acc(P, 1000)
np.savez("meanfield_checkpoint.npz", P=P, SIGN=m.SIGN, R_IN=m.R_IN, KAPPA=KAPPA, LR=LR, SEED=SEED)
print(f"mean-field GD done in {time.time() - t0:.0f} s; final test acc (n=1000) = {a:.4f}", flush=True)
print("saved meanfield_checkpoint.npz", flush=True)