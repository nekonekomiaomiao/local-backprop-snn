"""Decompose W3 update vs gradient by output class on one sample.
Prints per-class: sum of update components (kk-C*dt)*E*s and gradient components, with signs.
"""
import sys
import numpy as np

TAU_M = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
ISI = int(sys.argv[2]) if len(sys.argv) > 2 else 0
sys.argv = ["mnist_shared.py", "0", "1", "1.0", "200", "1.5e-8", "3000", "30", "1000",
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
    return a3, a2b

x = m.tr_flat[m.order[0]]
y = m.trl[m.order[0]]
yv = m.y_onehot[y]
a3, a2b = rates_if(m.P, x)
d3_mf = a3 - m.TARGET * yv
g3 = m.SIGN[m.G2:m.G3] * a2b[m.PRE3 - m.OFF_FC] * d3_mf[m.POST3 - m.OFF_OUT] / TH

# per-step actual update components for W3
upd3 = np.zeros(m.G3 - m.G2)
mean_d3_step = np.zeros(10)
mean_kk3 = np.zeros(m.G3 - m.G2)
for st in range(m.steps):
    m.spiking_step(x, yv, learn=True)
    d_out = m.f_est / m.TAU_F - m.TARGET * yv
    d3 = d_out[m.POST3 - m.OFF_OUT]
    kk3 = np.random.poisson(np.clip(m.FDA - d3, 0.0, None) * m.DT)
    upd3 += m.SIGN[m.G2:m.G3] * m.E3 * (kk3 - m.C * m.DT)
    mean_d3_step += d_out
    mean_kk3 += kk3
mean_d3_step /= m.steps
mean_kk3 /= m.steps

# per-class decomposition
print(f"TAU_M={TAU_M} ISI={ISI}  sample label y={y}")
print("per-class (10 outputs):  mean_d3_step  mean_kk3  g3_sum  upd3_sum")
for k in range(10):
    sel = m.POST3 - m.OFF_OUT == k
    print(f"  class {k}: d3={mean_d3_step[k]:+8.1f}  kk3={mean_kk3[sel].mean():6.2f}  "
          f"g3_sum={g3[sel].sum():+10.1f}  upd3_sum={upd3[sel].sum():+.3e}")
print(f"correct class = {y}, correct-class d3 should be < 0")
print(f"total: dot(upd3, g3) = {np.dot(upd3, g3):+.3e}   |upd3|={np.linalg.norm(upd3):.2e}  |g3|={np.linalg.norm(g3):.1f}")