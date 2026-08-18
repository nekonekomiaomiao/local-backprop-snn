import numpy as np
import mnist_conv_snn as m

rng = np.random.default_rng(7)
x = m.tr_flat[5].copy()
yv = m.y_onehot[m.trl[5]]

TH = m.THETA
TP = m.THETA_POOL


def mean_field_rates(P):
    """确定性速率域前向（均值场），返回各层速率向量。"""
    a_in = np.zeros(m.N_NEURONS)
    a_in[m.OFF_IN:m.OFF_IN + 784] = m.R_IN * x
    a_in[m.OFF_B1] = m.BIAS_RATE
    a_in[m.OFF_B2] = m.BIAS_RATE
    a_in[m.OFF_B3] = m.BIAS_RATE
    a_in[m.OFF_B4] = m.BIAS_RATE

    z1 = np.bincount(m.POST1 - m.OFF_L1, m.SIGN[:m.G1] * P[:m.G1] / TH * a_in[m.PRE1], minlength=m.N_L1)
    a1 = np.maximum(z1, 0)
    p1 = a1[m.PREP1 - m.OFF_L1].reshape(-1, 4).sum(axis=1) / TP
    p1v = np.zeros(m.N_NEURONS)
    p1v[m.P1] = p1
    p1v[m.OFF_B2] = m.BIAS_RATE

    z2 = np.bincount(m.POST2 - m.OFF_L2, m.SIGN[m.G1:m.G2] * P[m.G1:m.G2] / TH * p1v[m.PRE2], minlength=m.N_L2)
    a2 = np.maximum(z2, 0)
    p2 = a2[m.PREP2 - m.OFF_L2].reshape(-1, 4).sum(axis=1) / TP
    p2v = np.zeros(m.N_NEURONS)
    p2v[m.P2] = p2
    p2v[m.OFF_B3] = m.BIAS_RATE

    z3 = np.bincount(m.POST3 - m.OFF_FC, m.SIGN[m.G2:m.G3] * P[m.G2:m.G3] / TH * p2v[m.PRE3], minlength=m.NFC)
    a3 = np.maximum(z3, 0)

    a3b = np.concatenate([a3, [m.BIAS_RATE]])
    z4 = np.bincount(m.POST4 - m.OFF_OUT, m.SIGN[m.G3:m.G4] * P[m.G3:m.G4] / TH * a3b[m.PRE4 - m.OFF_FC], minlength=10)
    a4 = np.maximum(z4, 0)
    return a1, p1, p1v, a2, p2, p2v, a3, a4, a_in


def loss_from(a4):
    return float(np.mean(0.5 * ((a4 - m.TARGET * yv) / m.TARGET) ** 2))


def analytic_grad(P):
    a1, p1, p1v, a2, p2, p2v, a3, a4, a_in = mean_field_rates(P)
    d4 = a4 - m.TARGET * yv
    w4 = (m.SIGN[m.G3:m.G4] * P[m.G3:m.G4] / TH).reshape(10, m.NFC + 1)
    d3 = (a3 > 0) * (w4[:, 1:].T @ d4)
    d_p2 = np.bincount(m.PRE3 - m.OFF_P2, m.SIGN[m.G2:m.G3] * P[m.G2:m.G3] / TH * d3[m.POST3 - m.OFF_FC], minlength=m.N_P2 + 1)[:m.NP2]
    d2 = (a2 > 0) * np.bincount(m.PREP2 - m.OFF_L2, d_p2[m.POSTP2 - m.OFF_P2] / TP, minlength=m.N_L2)
    w2 = m.SIGN[m.G1:m.G2] * P[m.G1:m.G2] / TH
    d_p1 = np.bincount(m.PRE2 - m.OFF_P1, w2 * d2[m.POST2 - m.OFF_L2], minlength=m.N_P1 + 1)[:m.NP1]
    d1 = (a1 > 0) * np.bincount(m.PREP1 - m.OFF_L1, d_p1[m.POSTP1 - m.OFF_P1] / TP, minlength=m.N_L1)
    g = np.zeros(m.G4)
    g[:m.G1] = a_in[m.PRE1] * d1[m.POST1 - m.OFF_L1]
    g[m.G1:m.G2] = p1v[m.PRE2] * d2[m.POST2 - m.OFF_L2]
    g[m.G2:m.G3] = p2v[m.PRE3] * d3[m.POST3 - m.OFF_FC]
    a3b = np.concatenate([a3, [m.BIAS_RATE]])
    g[m.G3:m.G4] = a3b[m.PRE4 - m.OFF_FC] * d4[m.POST4 - m.OFF_OUT]
    return g


P0 = m.P.copy()
a1, p1, p1v, a2, p2, p2v, a3, a4, a_in = mean_field_rates(P0)
print("mean-field out rates:", np.round(a4, 1))
print("loss:", loss_from(a4))

g_ana = analytic_grad(P0)
picks = []
for sl in (slice(0, m.G1), slice(m.G1, m.G2), slice(m.G2, m.G3), slice(m.G3, m.G4)):
    gs = np.abs(g_ana[sl])
    k = min(3, int((gs > 1e-3).sum()))
    picks += [sl.start + i for i in np.argsort(gs)[-k:]]
idxs = picks

eps = 1e-4
g_num = np.zeros(m.G4)
for gi in idxs:
    Pp = P0.copy(); Pp[gi] += eps
    Pm = P0.copy(); Pm[gi] -= eps
    g_num[gi] = (loss_from(mean_field_rates(Pp)[7]) - loss_from(mean_field_rates(Pm)[7])) / (2 * eps)

print("indices   :", idxs)
print("num grad  :", np.round(g_num[idxs], 4))
print("ana grad  :", np.round(g_ana[idxs], 4))
cos = float(np.dot(g_ana[idxs], g_num[idxs]) / (np.linalg.norm(g_ana[idxs]) * np.linalg.norm(g_num[idxs]) + 1e-12))
print("cos(num, ana) at sampled indices:", round(cos, 6))
for name, sl in (("W1", slice(0, m.G1)), ("W2", slice(m.G1, m.G2)), ("W3", slice(m.G2, m.G3)), ("W4", slice(m.G3, m.G4))):
    gs = g_ana[sl]
    print(f"  {name}: max|g|={np.abs(gs).max():.2f}  nonzero={int((np.abs(gs) > 1e-6).sum())}/{gs.size}")
