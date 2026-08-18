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

N = 30
S = np.zeros((3, 5))   # ΣE, Σδ, ΣE·δ, ΣE·(k-Cdt), 步数
for s in range(N):
    ii = m.order[s]
    x = m.tr_flat[ii]
    y = m.trl[ii]
    yv = m.y_onehot[y]
    for st in range(m.steps):
        # 记录步前状态
        E_before = (m.E1.copy(), m.E2.copy(), m.E3.copy())
        m.spiking_step(x, yv, learn=True)
        # 重算步内 δ（同学习路径）
        w2 = m.SIGN[m.G1:m.G2] * m.P[m.G1:m.G2] / m.THETA
        w3 = m.SIGN[m.G2:m.G3] * m.P[m.G2:m.G3] / m.THETA
        gateFC = (m.r_est[m.FC] / m.TAU_R) > m.R_GATE
        gateF = (m.r_est[m.F] / m.TAU_R) > m.R_GATE
        d_out = m.f_est / m.TAU_F - m.TARGET * yv
        d_fc = gateFC * np.bincount(m.PRE3 - m.OFF_FC, w3 * d_out[m.POST3 - m.OFF_OUT], minlength=m.NFC + 1)[:m.NFC]
        d_f = np.bincount(m.PRE2 - m.OFF_F, w2 * d_fc[m.POST2 - m.OFF_FC], minlength=m.N_F + 1)[:m.N_F]
        ds = (d_f[m.POST1 - m.OFF_F], d_fc[m.POST2 - m.OFF_FC], d_out[m.POST3 - m.OFF_OUT])
        for gi in range(3):
            e = E_before[gi]
            d = ds[gi]
            S[gi, 0] += e.sum()
            S[gi, 1] += d.sum()
            S[gi, 2] += (e * d).sum()

for gi, name in enumerate(("W1", "W2", "W3")):
    n_syn = (m.G1, m.G2 - m.G1, m.G3 - m.G2)[gi]
    n = N * m.steps * n_syn
    Ee = S[gi, 0] / n
    Ed = S[gi, 1] / n
    Eed = S[gi, 2] / n
    n_e_steps = N * m.steps
    print(f"{name}: E[e]={Ee:.3f}  E[d]={Ed:.3f}  E[e*d]={Eed:.3f}  E[e]E[d]={Ee * Ed:.3f}  ratio E[ed]/(E[e]E[d])={Eed / (Ee * Ed + 1e-12):.3f}")
