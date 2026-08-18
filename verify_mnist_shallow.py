import sys
import numpy as np

mode = sys.argv[1] if len(sys.argv) > 1 else "stride"
loss_mode = sys.argv[2] if len(sys.argv) > 2 else "mse"
sys.argv = ["mnist_shallow.py", "0", "2000", "1.0", "200", "2e-6", "1000", "30", mode, "0", "100", "0.02", "1.0", loss_mode]
import mnist_shallow as m

x = m.tr_flat[5]
yv = m.y_onehot[m.trl[5]]
TH = m.THETA
TP = m.THETA_POOL


def rates(P):
    a_in = np.zeros(m.N_NEURONS)
    a_in[m.OFF_IN:m.OFF_IN + 784] = m.R_IN * x
    a_in[m.OFF_B1] = m.BIAS_RATE
    a_in[m.OFF_B2] = m.BIAS_RATE
    a_in[m.OFF_B3] = m.BIAS_RATE
    if m.MODE == "stride":
        z1 = np.bincount(m.POST1 - m.OFF_F, m.SIGN[:m.G1] * P[:m.G1] / TH * a_in[m.PRE1], minlength=m.N_F)
        a1 = np.maximum(z1, 0)
        p1v = np.zeros(m.N_NEURONS)
        p1v[m.F] = a1
        p1v[m.OFF_B2] = m.BIAS_RATE
    elif m.MODE == "avg":
        z1 = np.bincount(m.POST1 - m.OFF_L1, m.SIGN[:m.G1] * P[:m.G1] / TH * a_in[m.PRE1], minlength=m.N_L1)
        a1 = np.maximum(z1, 0)
        pf = a1[m.PREP1 - m.OFF_L1].reshape(-1, 4).sum(axis=1) / TP
        p1v = np.zeros(m.N_NEURONS)
        p1v[m.F] = pf
        p1v[m.OFF_B2] = m.BIAS_RATE
    else:  # max
        z1 = np.bincount(m.POST1 - m.OFF_L1, m.SIGN[:m.G1] * P[:m.G1] / TH * a_in[m.PRE1], minlength=m.N_L1)
        a1 = np.maximum(z1, 0)
        pf = a1[m.PREP1 - m.OFF_L1].reshape(-1, 4).max(axis=1)
        p1v = np.zeros(m.N_NEURONS)
        p1v[m.F] = pf
        p1v[m.OFF_B2] = m.BIAS_RATE
    z2 = np.bincount(m.POST2 - m.OFF_FC, m.SIGN[m.G1:m.G2] * P[m.G1:m.G2] / TH * p1v[m.PRE2], minlength=m.NFC)
    a2 = np.maximum(z2, 0)
    a2b = np.concatenate([a2, [m.BIAS_RATE]])
    z3 = np.bincount(m.POST3 - m.OFF_OUT, m.SIGN[m.G2:m.G3] * P[m.G2:m.G3] / TH * a2b[m.PRE3 - m.OFF_FC], minlength=10)
    a3 = np.maximum(z3, 0)
    return a1, a2, a3, a_in, p1v


def lossf(a3):
    if m.LOSS == "ce":
        p = np.exp((a3 - a3.max()) / m.TAU_SM)
        p = p / p.sum()
        return -np.log(p[np.argmax(yv)] + 1e-12)
    return float(np.mean(0.5 * ((a3 - m.TARGET * yv) / m.TARGET) ** 2))


def ana_grad(P):
    a1, a2, a3, a_in, p1v = rates(P)
    if m.LOSS == "ce":
        p = np.exp((a3 - a3.max()) / m.TAU_SM)
        p = p / p.sum()
        d3 = m.TARGET * (p - yv)
    else:
        d3 = a3 - m.TARGET * yv
    w3 = (m.SIGN[m.G2:m.G3] * P[m.G2:m.G3] / TH).reshape(10, m.NFC + 1)
    d2 = (a2 > 0) * (w3[:, 1:].T @ d3)
    d_f = np.bincount(m.PRE2 - m.OFF_F, m.SIGN[m.G1:m.G2] * P[m.G1:m.G2] / TH * d2[m.POST2 - m.OFF_FC], minlength=m.N_F + 1)[:m.N_F]
    if m.MODE == "stride":
        d1n = (a1 > 0) * d_f
        d1 = d1n[m.POST1 - m.OFF_F]
        off1 = m.OFF_F
    elif m.MODE == "avg":
        d_l1 = np.zeros(m.N_L1)
        d_l1[m.PREP1 - m.OFF_L1] = d_f[m.POSTP1 - m.OFF_F] / TP
        d1 = (a1 > 0)[m.POST1 - m.OFF_L1] * d_l1[m.POST1 - m.OFF_L1]
        off1 = m.OFF_L1
    else:
        d_l1 = np.zeros(m.N_L1)
        win = np.argmax(a1[m.PREP1_4 - m.OFF_L1], axis=1)
        flat = m.PREP1_4[np.arange(m.NP1), win]
        d_l1[flat - m.OFF_L1] = d_f
        d1 = (a1 > 0)[m.POST1 - m.OFF_L1] * d_l1[m.POST1 - m.OFF_L1]
        off1 = m.OFF_L1
    s = m.SIGN
    g = np.zeros(m.G3)
    g[:m.G1] = s[:m.G1] * a_in[m.PRE1] * d1 / TH
    g[m.G1:m.G2] = s[m.G1:m.G2] * p1v[m.PRE2] * d2[m.POST2 - m.OFF_FC] / TH
    a2b = np.concatenate([a2, [m.BIAS_RATE]])
    g[m.G2:m.G3] = s[m.G2:m.G3] * a2b[m.PRE3 - m.OFF_FC] * d3[m.POST3 - m.OFF_OUT] / TH
    return g


P0 = m.P.copy()
a3 = rates(P0)[2]
print(f"mode={m.MODE}  out rates:", np.round(a3, 1), " loss:", round(lossf(a3), 4))
g = ana_grad(P0)
for name, sl in (("W1", slice(0, m.G1)), ("W2", slice(m.G1, m.G2)), ("W3", slice(m.G2, m.G3))):
    gs = g[sl]
    print(f"  {name}: max|g|={np.abs(gs).max():.1f} nonzero={int((np.abs(gs) > 1e-6).sum())}/{gs.size}")

eps = 1e-4
idxs = []
for sl in (slice(0, m.G1), slice(m.G1, m.G2), slice(m.G2, m.G3)):
    gs = np.abs(g[sl])
    k = min(4, int((gs > 1e-3).sum()))
    idxs += [sl.start + i for i in np.argsort(gs)[-k:]]
g_num = np.zeros(m.G3)
for gi in idxs:
    Pp = P0.copy(); Pp[gi] += eps
    Pm = P0.copy(); Pm[gi] -= eps
    g_num[gi] = (lossf(rates(Pp)[2]) - lossf(rates(Pm)[2])) / (2 * eps)
signs = np.sign(g[idxs]) == np.sign(g_num[idxs])
print(f"FD check on {len(idxs)} high-gradient synapses: sign match {int(signs.sum())}/{len(idxs)}")
