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
N_SAMPLES = 8000

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = np.array([0.0, 1.0, 1.0, 0.0])

IDX_BIAS = np.arange(2, 5)
IDX_H1 = np.arange(5, 5 + N_H)
IDX_H2 = np.arange(5 + N_H, 5 + 2 * N_H)
IDX_OUT = 5 + 2 * N_H

pre1 = np.tile(np.arange(3), N_H)
post1 = np.repeat(IDX_H1, 3)
pre2 = np.repeat(np.concatenate([[IDX_BIAS[1]], IDX_H1]), N_H)
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

ETA = ALPHA * TAU_E / THETA
schedule = rng.integers(0, 4, N_SAMPLES)


def grad_w(Pv):
    w1 = (SIGN[:G1] * Pv[:G1] / THETA).reshape(N_H, 3)
    w2 = (SIGN[G1:G2] * Pv[G1:G2] / THETA).reshape(N_H, N_H + 1)
    w3 = SIGN[G2:] * Pv[G2:] / THETA
    g = np.zeros(N_SYN)
    for i in range(4):
        x, y = X[i], Y[i]
        f_in = np.concatenate([R * x, [BIAS_RATE]])
        z1 = w1 @ f_in
        a1 = np.maximum(z1, 0.0)
        z2 = w2 @ np.concatenate([[BIAS_RATE], a1])
        a2 = np.maximum(z2, 0.0) + a1
        z3 = w3 @ np.concatenate([[BIAS_RATE], a2])
        a3 = np.maximum(z3, 0.0)
        d3 = a3 - R * y
        da2 = d3 * w3[1:]
        d2 = (z2 > 0) * da2
        da1 = w2[:, 1:].T @ d2 + da2
        d1 = (z1 > 0) * da1
        g[:G1] += (d1[:, None] * f_in[None, :]).ravel()
        g[G1:G2] += (d2[:, None] * np.concatenate([[BIAS_RATE], a1])[None, :]).ravel()
        g[G2:] += d3 * np.concatenate([[BIAS_RATE], a2])
    return g / 4.0


losses = np.zeros(N_SAMPLES)
accs = np.zeros(N_SAMPLES)
for smp in range(N_SAMPLES):
    x, y = X[schedule[smp]], Y[schedule[smp]]
    g = grad_w(P)
    dP = -SIGN * ETA * g * SAMPLE_T
    P = np.clip(P + dP, 1e-6, 1 - 1e-6)
    w1 = (SIGN[:G1] * P[:G1] / THETA).reshape(N_H, 3)
    w2 = (SIGN[G1:G2] * P[G1:G2] / THETA).reshape(N_H, N_H + 1)
    w3 = SIGN[G2:] * P[G2:] / THETA
    a1 = np.maximum(w1 @ np.concatenate([R * x, [BIAS_RATE]]), 0.0)
    a2 = np.maximum(w2 @ np.concatenate([[BIAS_RATE], a1]), 0.0) + a1
    a3 = np.maximum(w3 @ np.concatenate([[BIAS_RATE], a2]), 0.0)
    losses[smp] = 0.5 * ((a3 - R * y) / R) ** 2
    accs[smp] = float((a3 > R / 2.0) == (y == 1.0))
    if (smp + 1) % 1000 == 0:
        print(f"sample {smp + 1:6d}  loss(roll) {np.mean(losses[smp-99:smp+1]):.4f}  acc(roll) {np.mean(accs[smp-99:smp+1]):.3f}")

print("evaluation:")
for i in range(4):
    x, y = X[i], Y[i]
    w1 = (SIGN[:G1] * P[:G1] / THETA).reshape(N_H, 3)
    w2 = (SIGN[G1:G2] * P[G1:G2] / THETA).reshape(N_H, N_H + 1)
    w3 = SIGN[G2:] * P[G2:] / THETA
    a1 = np.maximum(w1 @ np.concatenate([R * x, [BIAS_RATE]]), 0.0)
    a2 = np.maximum(w2 @ np.concatenate([[BIAS_RATE], a1]), 0.0) + a1
    a3 = np.maximum(w3 @ np.concatenate([[BIAS_RATE], a2]), 0.0)
    pred = a3 > R / 2.0
    print(f"  x = ({int(x[0])},{int(x[1])})  y = {int(y)}  out = {a3:6.1f} Hz  pred = {int(pred)}  {'OK' if pred == (y == 1.0) else 'FAIL'}")
w1m = (SIGN[:G1] * P[:G1] / THETA).reshape(N_H, 3)
print("W1 differential units:", np.sum(np.sign(w1m[:, 0]) != np.sign(w1m[:, 1])), "of", N_H)
