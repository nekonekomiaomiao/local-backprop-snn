import sys
import numpy as np

sys.argv = ["mnist_shallow.py", "0", "500", "1.0", "200", "1e-7", "8000", "30", "stride"]
import mnist_shallow as m

m.P[:] = np.full(m.G3, m.P_INIT)
m.P[m.G2 + np.arange(m.NOUT) * (m.NFC + 1)] = 0.6
m.u[:] = 0
m.r_est[:] = 0
m.f_est[:] = 0
m.E1[:] = 0
m.E2[:] = 0
m.E3[:] = 0

N = 40
for s in range(N):
    x = m.tr_flat[m.order[s]]
    y = m.trl[m.order[s]]
    for _ in range(m.steps):
        m.spiking_step(x, y, learn=True)

rates = m.r_est / m.TAU_R

def meanfield(xv):
    a_in = np.zeros(m.N_NEURONS)
    a_in[m.OFF_IN:m.OFF_IN + 784] = m.R_IN * xv
    a_in[m.OFF_B1] = m.BIAS_RATE
    a_in[m.OFF_B2] = m.BIAS_RATE
    a_in[m.OFF_B3] = m.BIAS_RATE
    z1 = np.bincount(m.POST1 - m.OFF_F, m.SIGN[:m.G1] * m.P[:m.G1] / m.THETA * a_in[m.PRE1], minlength=m.N_F)
    a1 = np.maximum(z1, 0)
    p1v = np.zeros(m.N_NEURONS)
    p1v[m.F] = a1
    p1v[m.OFF_B2] = m.BIAS_RATE
    z2 = np.bincount(m.POST2 - m.OFF_FC, m.SIGN[m.G1:m.G2] * m.P[m.G1:m.G2] / m.THETA * p1v[m.PRE2], minlength=m.NFC)
    a2 = np.maximum(z2, 0)
    a2b = np.concatenate([a2, [m.BIAS_RATE]])
    z3 = np.bincount(m.POST3 - m.OFF_OUT, m.SIGN[m.G2:m.G3] * m.P[m.G2:m.G3] / m.THETA * a2b[m.PRE3 - m.OFF_FC], minlength=10)
    a3 = np.maximum(z3, 0)
    return a1, a2, a3

mf = meanfield(m.tr_flat[m.order[0]])
print(f"layer      spiking(mean over 40 samples)   mean-field")
print(f"F(conv)    {rates[m.F].mean():9.1f}  vs  {mf[0].mean():9.1f}")
print(f"FC         {rates[m.FC].mean():9.1f}  vs  {mf[1].mean():9.1f}")
print(f"OUT        {np.round(rates[m.OUT],1)}")
print(f"OUT mf     {np.round(mf[2],1)}")
print(f"spike ratio F: {rates[m.F].mean() / (mf[0].mean() + 1e-9):.3f}")
print(f"spike ratio FC: {rates[m.FC].mean() / (mf[1].mean() + 1e-9):.3f}")
print(f"OUT total spiking {rates[m.OUT].sum():.1f} vs mf {mf[2].sum():.1f}")
