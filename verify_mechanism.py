import numpy as np

rng = np.random.default_rng(0)

R = 100.0
BIAS_RATE = 15.0
THETA = 1.0
TAU_E = 0.05
FDA = 120.0
C = FDA
DT = 0.02
SAMPLE_T = 2.0
ALPHA = 2e-5
P_INIT = 0.3
N_H = 16

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = np.array([0.0, 1.0, 1.0, 0.0])

IDX_BIAS = np.arange(2, 5)
IDX_H1 = np.arange(5, 5 + N_H)
IDX_H2 = np.arange(5 + N_H, 5 + 2 * N_H)
IDX_OUT = 5 + 2 * N_H

pre1 = np.tile(np.arange(3), N_H)
post1 = np.repeat(IDX_H1, 3)
pre2 = np.tile(np.concatenate([[IDX_BIAS[1]], IDX_H1]), N_H)
post2 = np.repeat(IDX_H2, N_H + 1)
pre3 = np.concatenate([[IDX_BIAS[2]], IDX_H2])
post3 = np.repeat(IDX_OUT, N_H + 1)
PRE = np.concatenate([pre1, pre2, pre3])
POST = np.concatenate([post1, post2, post3])
G1 = len(pre1)
G2 = G1 + len(pre2)
N_SYN = len(PRE)

SIGN = rng.choice([-1.0, 1.0], N_SYN)
P = np.full(N_SYN, P_INIT)
P[G2] = 0.6


def analytic_grad(Pv):
    w1 = (SIGN[:G1] * Pv[:G1] / THETA).reshape(N_H, 3)
    w2 = (SIGN[G1:G2] * Pv[G1:G2] / THETA).reshape(N_H, N_H + 1)
    w3 = SIGN[G2:] * Pv[G2:] / THETA
    grad = np.zeros(N_SYN)
    for i in range(4):
        x, y = X[i], Y[i]
        f_in = np.concatenate([R * x, [BIAS_RATE]])
        z1 = w1 @ f_in
        a1 = np.maximum(z1, 0.0)
        z2 = w2 @ np.concatenate([[BIAS_RATE], a1])
        a2 = np.maximum(z2 + a1 / THETA, 0.0)
        z3 = w3 @ np.concatenate([[BIAS_RATE], a2])
        a3 = np.maximum(z3, 0.0)
        d3 = a3 - R * y
        da2 = d3 * w3[1:]
        d2 = (a2 > 0) * da2
        da1 = w2[:, 1:].T @ d2 + d2 / THETA
        d1 = (a1 > 0) * da1
        g3 = d3 * np.concatenate([[BIAS_RATE], a2])
        g2 = d2[:, None] * np.concatenate([[BIAS_RATE], a1])[None, :]
        g1 = d1[:, None] * f_in[None, :]
        grad[:G1] += g1.ravel()
        grad[G1:G2] += g2.ravel()
        grad[G2:] += g3.ravel()
    return grad / 4.0


def construction_update(Pv):
    w1 = (SIGN[:G1] * Pv[:G1] / THETA).reshape(N_H, 3)
    w2 = (SIGN[G1:G2] * Pv[G1:G2] / THETA).reshape(N_H, N_H + 1)
    w3 = SIGN[G2:] * Pv[G2:] / THETA
    upd = np.zeros(N_SYN)
    for i in range(4):
        x, y = X[i], Y[i]
        f_in = np.concatenate([R * x, [BIAS_RATE]])
        z1 = w1 @ f_in
        a1 = np.maximum(z1, 0.0)
        z2 = w2 @ np.concatenate([[BIAS_RATE], a1])
        a2 = np.maximum(z2 + a1 / THETA, 0.0)
        a3 = np.maximum(w3 @ np.concatenate([[BIAS_RATE], a2]), 0.0)
        d3 = a3 - R * y
        da2 = d3 * w3[1:]
        d2 = (a2 > 0) * da2
        d1 = (a1 > 0) * (w2[:, 1:].T @ d2 + d2 / THETA)
        d = np.zeros(6 + 2 * N_H)
        d[IDX_H1] = d1
        d[IDX_H2] = d2
        d[IDX_OUT] = d3
        e = np.zeros(N_SYN)
        e[:G1] = TAU_E * np.tile(np.concatenate([f_in[:2], [BIAS_RATE]]), N_H)
        e[G1:G2] = TAU_E * np.tile(np.concatenate([[BIAS_RATE], a1]), N_H)
        e[G2:] = TAU_E * np.concatenate([[BIAS_RATE], a2])
        upd += SIGN * ALPHA * e * (FDA - d[POST] - C) * SAMPLE_T
    return upd / 4.0


g_ana = analytic_grad(P)
g_con = SIGN * construction_update(P) / THETA
cos = np.dot(g_ana, -g_con) / (np.linalg.norm(g_ana) * np.linalg.norm(g_con) + 1e-12)
print(f"cosine between gradient and -construction update (expect ~+1): {cos:.4f}")
mask = (np.abs(g_ana) > 1e-6) & (np.abs(g_con) > 1e-9)
print(f"sign agreement on nonzero entries: {np.mean(np.sign(g_ana[mask]) == np.sign(-g_con[mask])):.3f}  ({mask.sum()} entries)")
print(f"|grad| exact={np.linalg.norm(g_ana):.4f}  construction={np.linalg.norm(g_con):.4f}")
