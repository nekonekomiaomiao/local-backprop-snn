import sys
import numpy as np
import mnist_shallow as m

z = np.load("mnist_checkpoint.npz")
m.P[:] = z["P"]


def mean_field_out(x_vec, P):
    a_in = np.zeros(m.N_NEURONS)
    a_in[m.OFF_IN:m.OFF_IN + 784] = m.R_IN * x_vec
    a_in[m.OFF_B1] = m.BIAS_RATE
    a_in[m.OFF_B2] = m.BIAS_RATE
    a_in[m.OFF_B3] = m.BIAS_RATE
    if m.MODE == "stride":
        z1 = np.bincount(m.POST1 - m.OFF_F, m.SIGN[:m.G1] * P[:m.G1] / m.THETA * a_in[m.PRE1], minlength=m.N_F)
        a1 = np.maximum(z1, 0)
        p1v = np.zeros(m.N_NEURONS)
        p1v[m.F] = a1
        p1v[m.OFF_B2] = m.BIAS_RATE
    elif m.MODE == "avg":
        z1 = np.bincount(m.POST1 - m.OFF_L1, m.SIGN[:m.G1] * P[:m.G1] / m.THETA * a_in[m.PRE1], minlength=m.N_L1)
        a1 = np.maximum(z1, 0)
        p1v = np.zeros(m.N_NEURONS)
        p1v[m.F] = a1[m.PREP1 - m.OFF_L1].reshape(-1, 4).sum(axis=1) / m.THETA_POOL
        p1v[m.OFF_B2] = m.BIAS_RATE
    else:
        z1 = np.bincount(m.POST1 - m.OFF_L1, m.SIGN[:m.G1] * P[:m.G1] / m.THETA * a_in[m.PRE1], minlength=m.N_L1)
        a1 = np.maximum(z1, 0)
        p1v = np.zeros(m.N_NEURONS)
        p1v[m.F] = a1[m.PREP1 - m.OFF_L1].reshape(-1, 4).max(axis=1)
        p1v[m.OFF_B2] = m.BIAS_RATE
    z2 = np.bincount(m.POST2 - m.OFF_FC, m.SIGN[m.G1:m.G2] * P[m.G1:m.G2] / m.THETA * p1v[m.PRE2], minlength=m.NFC)
    a2 = np.maximum(z2, 0)
    a2b = np.concatenate([a2, [m.BIAS_RATE]])
    z3 = np.bincount(m.POST3 - m.OFF_OUT, m.SIGN[m.G2:m.G3] * P[m.G2:m.G3] / m.THETA * a2b[m.PRE3 - m.OFF_FC], minlength=10)
    a3 = np.maximum(z3, 0)
    return a3


rng = np.random.default_rng(2)
idx = rng.choice(10000, 1000, replace=False)
hits = 0
for ii in idx:
    a3 = mean_field_out(m.te_flat[ii], m.P)
    hits += int(np.argmax(a3) == m.tel[ii])
print(f"mean-field (noise-free) eval acc on 1000 test: {hits / 1000:.4f}")

idx = rng.choice(60000, 1000, replace=False)
hits = 0
for ii in idx:
    a3 = mean_field_out(m.tr_flat[ii], m.P)
    hits += int(np.argmax(a3) == m.trl[ii])
print(f"mean-field (noise-free) eval acc on 1000 train: {hits / 1000:.4f}")
