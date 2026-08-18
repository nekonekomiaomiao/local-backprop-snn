import sys
import numpy as np

sys.argv = ["mnist_shallow.py", "0", "500", "1.0", "200", "5e-8", "4000", "30", "stride"]
import mnist_shallow as m

m.P[:] = np.full(m.G3, m.P_INIT)
m.P[m.G2 + np.arange(m.NOUT) * (m.NFC + 1)] = 0.6
m.u[:] = 0
m.r_est[:] = 0
m.f_est[:] = 0
m.E1[:] = 0
m.E2[:] = 0
m.E3[:] = 0

N_SAMPLES = 80
acc = np.zeros((3, 5))   # W1/W2/W3 × (sum e, sum d, sum ed, sum ee, sum dd)
n_steps = N_SAMPLES * m.steps

for smp in range(N_SAMPLES):
    x = m.tr_flat[m.order[smp]]
    yv = m.y_onehot[m.trl[m.order[smp]]]
    for _ in range(m.steps):
        m.spiking_step(x, yv, learn=True)
        # 重算当前 e 与 d（与学习路径一致）
        w2 = m.SIGN[m.G1:m.G2] * m.P[m.G1:m.G2] / m.THETA
        w3 = m.SIGN[m.G2:m.G3] * m.P[m.G2:m.G3] / m.THETA
        gateFC = (m.r_est[m.FC] / m.TAU_R) > m.R_GATE
        d_out = m.f_est / m.TAU_F - m.TARGET * yv
        d_fc = gateFC * np.bincount(m.PRE3 - m.OFF_FC, w3 * d_out[m.POST3 - m.OFF_OUT], minlength=m.NFC + 1)[:m.NFC]
        d_f = np.bincount(m.PRE2 - m.OFF_F, w2 * d_fc[m.POST2 - m.OFF_FC], minlength=m.N_F + 1)[:m.N_F]
        d1 = d_f[m.POST1 - m.OFF_F]
        d2 = d_fc[m.POST2 - m.OFF_FC]
        d3 = d_out[m.POST3 - m.OFF_OUT]
        for gi, (e, d) in enumerate(((m.E1, d1), (m.E2, d2), (m.E3, d3))):
            acc[gi, 0] += e.sum()
            acc[gi, 1] += d.sum()
            acc[gi, 2] += (e * d).sum()
            acc[gi, 3] += (e * e).sum()
            acc[gi, 4] += (d * d).sum()

for gi, name in enumerate(("W1", "W2", "W3")):
    n = n_steps * (m.G1 if gi == 0 else (m.G2 - m.G1 if gi == 1 else m.G3 - m.G2))
    me = acc[gi, 0] / n
    md = acc[gi, 1] / n
    cov = acc[gi, 2] / n - me * md
    vare = acc[gi, 3] / n - me * me
    vard = acc[gi, 4] / n - md * md
    corr = cov / np.sqrt(vare * vard + 1e-12)
    print(f"{name}: corr(e,d)={corr:.3f}  E[e]={me:.2f}  E[d]={md:.2f}  cov={cov:.1f}")
    bias_frac = cov / (np.abs(me * md) + 1e-12)
    print(f"   covariance bias |cov(e,d)| / |E[e]E[d]| = {bias_frac:.3f}")
