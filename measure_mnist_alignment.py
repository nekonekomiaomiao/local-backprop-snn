import sys
import numpy as np
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

N = 60
cos_acc = np.zeros(N)
for s in range(N):
    ii = m.order[s]
    x = m.tr_flat[ii]
    y = m.trl[ii]
    yv = m.y_onehot[y]
    P0 = m.P.copy()
    # 解析梯度（当前样本的均值场）
    v.x = x
    v.yv = yv
    g = v.ana_grad(P0)
    for _ in range(m.steps):
        m.spiking_step(x, yv, learn=True)
    dP = m.P - P0
    # 有效梯度方向 = -g (GD)，但 g 是 ∂L/∂P
    n_g = np.linalg.norm(g)
    n_d = np.linalg.norm(dP)
    cos_acc[s] = float(np.dot(g, -dP) / (n_g * n_d + 1e-12)) if (n_g > 0 and n_d > 0) else 0.0

print(f"per-sample cos(actual update, -gradient): mean={cos_acc.mean():+.4f}  "
      f"median={np.median(cos_acc):+.4f}  sign>0 fraction={float((cos_acc > 0).mean()):.3f}")
g1 = np.abs(v.ana_grad(m.P)[:m.G1])
g2 = np.abs(v.ana_grad(m.P)[m.G1:m.G2])
g3 = np.abs(v.ana_grad(m.P)[m.G2:m.G3])
print(f"gradient magnitude (last sample): W1 {g1.mean():.1f}  W2 {g2.mean():.1f}  W3 {g3.mean():.1f}")
