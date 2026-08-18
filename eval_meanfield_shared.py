import sys
import numpy as np

ckpt = sys.argv[1] if len(sys.argv) > 1 else "mnist_checkpoint.npz"
sys.argv = ["mnist_shared.py", "0", "500", "1.0", "200", "3e-8", "3000", "30", "1000", "100"]
import mnist_shared as m

z = np.load(ckpt)
m.P[:] = z["P"]
TH = m.THETA


def mean_field_out(x_vec, P):
    a_in = np.zeros(m.N_NEURONS)
    a_in[m.OFF_IN:m.OFF_IN + 784] = m.R_IN * x_vec
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
    return a3


rng = np.random.default_rng(2)
idx = rng.choice(10000, 2000, replace=False)
hits = 0
for ii in idx:
    a3 = mean_field_out(m.te_flat[ii], m.P)
    hits += int(np.argmax(a3) == m.tel[ii])
print(f"mean-field (noise-free) eval acc on 2000 test: {hits / 2000:.4f}")

idx = rng.choice(60000, 2000, replace=False)
hits = 0
for ii in idx:
    a3 = mean_field_out(m.tr_flat[ii], m.P)
    hits += int(np.argmax(a3) == m.trl[ii])
print(f"mean-field (noise-free) eval acc on 2000 train: {hits / 2000:.4f}")