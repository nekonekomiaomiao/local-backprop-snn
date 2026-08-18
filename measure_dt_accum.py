import sys
import numpy as np

dT = float(sys.argv[1])
alpha = float(sys.argv[2])
N = int(sys.argv[3])
sys.argv = ["mnist_shallow.py", "0", "500", str(dT), "200", str(alpha), "8000", "30", "stride"]
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

lr = alpha * m.TAU_E * m.THETA * dT
acc_dP = np.zeros(m.G3)
acc_g = np.zeros(m.G3)
for s in range(N):
    ii = m.order[s]
    x = m.tr_flat[ii]
    y = m.trl[ii]
    yv = m.y_onehot[y]
    P0 = m.P.copy()
    v.x = x
    v.yv = yv
    g = v.ana_grad(P0)
    for _ in range(m.steps):
        m.spiking_step(x, yv, learn=True)
    acc_dP += m.P - P0
    acc_g += g
cum_cos = float(np.dot(acc_g, -acc_dP) / (np.linalg.norm(acc_g) * np.linalg.norm(acc_dP) + 1e-12))
mag = np.linalg.norm(acc_dP) / (lr * np.linalg.norm(acc_g) + 1e-12)
print(f"dT={dT} alpha={alpha}: cumulative cos = {cum_cos:+.4f}, |ΣΔP|/(lr|Σg|) = {mag:.3f}")
