import sys
import numpy as np

sys.argv = ["mnist_shared.py", "0", "500", "10.0", "200", "3e-9", "3000", "30", "0", "100"]
import mnist_shared as m

x = m.tr_flat[5]
yv = m.y_onehot[m.trl[5]]
TH = m.THETA


def rates(P):
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


def lossf(a3):
    return float(np.mean(0.5 * ((a3 - m.TARGET * yv) / m.TARGET) ** 2))


def ana_grad(P):
    a1, a2, a3, a_in, p1v = rates(P)
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


P0 = m.P.copy()
a3 = rates(P0)[2]
print("out rates:", np.round(a3, 1), " loss:", round(lossf(a3), 4))
g = ana_grad(P0)
for name, sl in (("CONV1(共享)", slice(0, m.G1)), ("FC", slice(m.G1, m.G2)), ("OUT", slice(m.G2, m.G3))):
    gs = g[sl]
    print(f"  {name}: max|g|={np.abs(gs).max():.1f} nonzero={int((np.abs(gs) > 1e-6).sum())}/{gs.size}")

eps = 1e-5
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
print(f"FD check on {len(idxs)} high-gradient params: sign match {int(signs.sum())}/{len(idxs)}")
print("num:", np.round(g_num[idxs], 4))
print("ana:", np.round(g[idxs], 1))
