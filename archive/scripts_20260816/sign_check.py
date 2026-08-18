"""Sign sanity check: per-sample dot of update vs gradient per layer.
Answer: in the working IF protocol, is E[dP] parallel (+) or anti-parallel (-) to E[g]?
"""
import sys
import numpy as np

TAU_M = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
ISI = int(sys.argv[2]) if len(sys.argv) > 2 else 0
N = int(sys.argv[3]) if len(sys.argv) > 3 else 30
SAMPLE_T = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
sys.argv = ["mnist_shared.py", "0", str(N), str(SAMPLE_T), "200", "1.5e-8", "3000", "30", "1000",
            "1000", "0.02", "0.2", "0", "0", str(ISI), str(TAU_M)]
import mnist_shared as m

z = np.load("exp4/reset_a15_cont/mnist_checkpoint.npz")
m.P[:] = z["P"]
m.SIGN[:] = z["SIGN"]
m.u[:] = 0
m.r_est[:] = 0
m.f_est[:] = 0
m.E1[:] = 0
m.E2[:] = 0
m.E3[:] = 0

TH = m.THETA

def rates_if(P, x):
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
    return a1, a2, a3, a_in, p1v

def ana_grad(P, x, yv):
    a1, a2, a3, a_in, p1v = rates_if(P, x)
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

sig_dP = np.zeros(m.G3)
sig_g = np.zeros(m.G3)
n_samples = 0
for smp in range(N):
    ii = m.order[smp]
    x = m.tr_flat[ii]
    y = m.trl[ii]
    yv = m.y_onehot[y]
    P0 = m.P.copy()
    g = ana_grad(m.P, x, yv)
    for st in range(m.steps):
        m.spiking_step(x, yv, learn=True)
    if ISI > 0:
        zx = np.zeros(784)
        zy = np.zeros(m.NOUT)
        for _ in range(ISI):
            m.spiking_step(zx, zy, learn=False)
    dP = m.P - P0
    sig_dP += dP
    sig_g += g
    n_samples += 1

sl1 = slice(0, m.G1)
sl2 = slice(m.G1, m.G2)
sl3 = slice(m.G2, m.G3)
for name, sl in (("W1", sl1), ("W2", sl2), ("W3", sl3)):
    sd = sig_dP[sl]
    sg = sig_g[sl]
    c = float(np.dot(sd, sg) / (np.linalg.norm(sd) * np.linalg.norm(sg) + 1e-12))
    print(f"TAU_M={TAU_M} ISI={ISI} N={n_samples}  {name}: cos(E[dP], E[g])={c:+.3f}  "
          f"|E[dP]|/|E[g]|={np.linalg.norm(sd) / (np.linalg.norm(sg) + 1e-12):.3e}")