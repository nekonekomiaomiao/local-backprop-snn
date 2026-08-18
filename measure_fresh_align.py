import sys
import numpy as np

sys.argv = ["mnist_shallow.py", "0", "500", "1.0", "200", "5e-6", "8000", "30", "stride"]
import mnist_shallow as m
import verify_mnist_shallow as v

m.P[:] = np.full(m.G3, m.P_INIT)
m.P[m.G2 + np.arange(m.NOUT) * (m.NFC + 1)] = 0.6

N = 40
for mode in ("continuous", "fresh"):
    m.u[:] = 0
    m.r_est[:] = 0
    m.f_est[:] = 0
    m.E1[:] = 0
    m.E2[:] = 0
    m.E3[:] = 0
    m.P[:] = np.full(m.G3, m.P_INIT)
    m.P[m.G2 + np.arange(m.NOUT) * (m.NFC + 1)] = 0.6
    cos = np.zeros(N)
    for s in range(N):
        ii = m.order[s]
        x = m.tr_flat[ii]
        y = m.trl[ii]
        yv = m.y_onehot[y]
        if mode == "fresh":
            m.u[:] = 0
            m.r_est[:] = 0
            m.f_est[:] = 0
            m.E1[:] = 0
            m.E2[:] = 0
            m.E3[:] = 0
        P0 = m.P.copy()
        v.x = x
        v.yv = yv
        g = v.ana_grad(P0)
        for _ in range(m.steps):
            m.spiking_step(x, yv, learn=True)
        dP = m.P - P0
        ng, nd = np.linalg.norm(g), np.linalg.norm(dP)
        cos[s] = float(np.dot(g, -dP) / (ng * nd + 1e-12)) if (ng > 0 and nd > 0) else 0.0
    print(f"mode={mode}: mean cos = {cos.mean():+.4f}, sign>0 frac = {float((cos > 0).mean()):.3f}")
