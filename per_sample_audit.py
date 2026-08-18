import sys
import numpy as np

sys.argv = ["mnist_shallow.py", "0", "500", "1.0", "200", "1e-7", "8000", "30", "stride"]
import mnist_shallow as m
import verify_mnist_shallow as v

m.P[:] = np.full(m.G3, m.P_INIT)
m.P[m.G2 + np.arange(m.NOUT) * (m.NFC + 1)] = 0.6
m.u[:] = 0
m.r_est[:] = 0
m.f_est[:] = 0
m.E1[:] = 0
m.E2[:] = 0
m.E3[:] = 0

lr = m.ALPHA * m.TAU_E * m.THETA * m.SAMPLE_T
for s in range(3):
    ii = m.order[s]
    x = m.tr_flat[ii]
    y = m.trl[ii]
    yv = m.y_onehot[y]
    P0 = m.P.copy()
    v.x = x
    v.yv = yv
    g = v.ana_grad(P0)
    sum_Ed = np.zeros(3)
    sum_Ek = np.zeros(3)
    sum_E = np.zeros(3)
    sum_d = np.zeros(3)
    for st in range(m.steps):
        E_b = (m.E1.copy(), m.E2.copy(), m.E3.copy())
        # 步内 δ（与更新一致）
        w2 = m.SIGN[m.G1:m.G2] * m.P[m.G1:m.G2] / m.THETA
        w3 = m.SIGN[m.G2:m.G3] * m.P[m.G2:m.G3] / m.THETA
        gateFC = (m.r_est[m.FC] / m.TAU_R) > m.R_GATE
        gateF = (m.r_est[m.F] / m.TAU_R) > m.R_GATE
        d_out = m.f_est / m.TAU_F - m.TARGET * yv
        d_fc = gateFC * np.bincount(m.PRE3 - m.OFF_FC, w3 * d_out[m.POST3 - m.OFF_OUT], minlength=m.NFC + 1)[:m.NFC]
        d_f = np.bincount(m.PRE2 - m.OFF_F, w2 * d_fc[m.POST2 - m.OFF_FC], minlength=m.N_F + 1)[:m.N_F]
        ds = (d_f[m.POST1 - m.OFF_F], d_fc[m.POST2 - m.OFF_FC], d_out[m.POST3 - m.OFF_OUT])
        for gi in range(3):
            sum_E[gi] += E_b[gi].sum()
            sum_d[gi] += ds[gi].sum()
            sum_Ed[gi] += (E_b[gi] * ds[gi]).sum()
        m.spiking_step(x, yv, learn=True)
        for gi in range(3):
            ferr = np.clip(m.FDA - ds[gi], 0, None)
            # E[k] 用理论值 (ferr*DT) 代替，测 E[e]·E[k-Cdt] 的期望
    dP = m.P - P0
    gth = -lr * g
    n_syns = (m.G1, m.G2 - m.G1, m.G3 - m.G2)
    print(f"--- sample {s} label {m.trl[ii]}")
    for gi, name in enumerate(("W1", "W2", "W3")):
        ns = n_syns[gi]
        act = dP[slice(0, m.G1) if gi == 0 else slice(m.G1, m.G2) if gi == 1 else slice(m.G2, m.G3)]
        th = gth[slice(0, m.G1) if gi == 0 else slice(m.G1, m.G2) if gi == 1 else slice(m.G2, m.G3)]
        Ee = sum_E[gi] / (m.steps * ns)
        dd = sum_d[gi] / (m.steps * ns)
        Eed = sum_Ed[gi] / (m.steps * ns)
        sig = float(np.dot(act, th)) / (np.linalg.norm(act) * np.linalg.norm(th) + 1e-12)
        mag = np.linalg.norm(act) / (np.linalg.norm(th) + 1e-12)
        print(f"  {name}: E[e]={Ee:.2f} (tau_e*a) E[d]={dd:.2f} E[e*d]={Eed:.1f} | E[e]E[d]={Ee * dd:.1f}")
        print(f"         actual/expected: cos={sig:+.3f} mag_ratio={mag:.3f}")
