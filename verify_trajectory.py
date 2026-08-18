import sys
import time
import numpy as np

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
RNG = np.random.default_rng(SEED)

R = 200.0
BIAS_RATE = 30.0
THETA = 1.0
TARGET = 200.0
TAU_E = 0.2
TAU_R = 0.1
TAU_F = 0.1
FDA = 500.0
C = FDA
DT = 0.02
SAMPLE_T = 2.0
ALPHA = 2.5e-6
P_INIT = 0.3
N_SAMPLES = 6000
WIN = 300

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = np.array([0.0, 1.0, 1.0, 0.0])

N_H = 16
N_NEURONS = 6 + 2 * N_H
IDX_IN = np.arange(2)
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
N_SYN = len(PRE)
G1 = len(pre1)
G2 = G1 + len(pre2)

SIGN = RNG.choice([-1.0, 1.0], N_SYN)
P = np.full(N_SYN, P_INIT)
P[G2] = 0.6

STEPS_PER_SAMPLE = int(SAMPLE_T / DT)


def weights(Pv):
    return ((SIGN[:G1] * Pv[:G1] / THETA).reshape(N_H, 3),
            (SIGN[G1:G2] * Pv[G1:G2] / THETA).reshape(N_H, N_H + 1),
            SIGN[G2:] * Pv[G2:] / THETA)


def grad_w(Pv):
    w1, w2, w3 = weights(Pv)
    g = np.zeros(N_SYN)
    for i in range(4):
        x, y = X[i], Y[i]
        f_in = np.concatenate([R * x, [BIAS_RATE]])
        a1 = np.maximum(w1 @ f_in, 0.0)
        a2 = np.maximum(w2 @ np.concatenate([[BIAS_RATE], a1]) + a1 / THETA, 0.0)
        a3 = np.maximum(w3 @ np.concatenate([[BIAS_RATE], a2]), 0.0)
        d3 = a3 - TARGET * y
        da2 = d3 * w3[1:]
        d2 = (a2 > 0) * da2
        d1 = (a1 > 0) * (w2[:, 1:].T @ d2 + d2 / THETA)
        g[:G1] += (d1[:, None] * f_in[None, :]).ravel()
        g[G1:G2] += (d2[:, None] * np.concatenate([[BIAS_RATE], a1])[None, :]).ravel()
        g[G2:] += d3 * np.concatenate([[BIAS_RATE], a2])
    return g / 4.0


u = np.zeros(N_NEURONS)
r_est = np.zeros(N_NEURONS)
f_est = 0.0
E_trace = np.zeros(N_SYN)
schedule = RNG.integers(0, 4, N_SAMPLES)
acc_dw = np.zeros(N_SYN)
acc_g = np.zeros(N_SYN)
losses = np.zeros(N_SAMPLES)
cos_hist = []

t0 = time.time()
for smp in range(N_SAMPLES):
    x = X[schedule[smp]]
    y = Y[schedule[smp]]
    g_now = grad_w(P)
    acc_g += g_now
    for _ in range(STEPS_PER_SAMPLE):
        pre_spikes = np.zeros(N_NEURONS)
        pre_spikes[IDX_IN] = RNG.poisson(R * x * DT)
        pre_spikes[IDX_BIAS] = RNG.poisson(BIAS_RATE * DT)

        k1 = RNG.binomial(pre_spikes[pre1].astype(np.int64), P[:G1])
        u[IDX_H1] += np.bincount(post1 - IDX_H1[0], SIGN[:G1] * k1, minlength=N_H)
        E_trace[:G1] += -E_trace[:G1] * DT / TAU_E + pre_spikes[pre1]
        n1 = np.floor(u[IDX_H1] / THETA).clip(0.0, None).astype(np.int64)
        u[IDX_H1] -= n1 * THETA
        r_est[IDX_H1] += -r_est[IDX_H1] * DT / TAU_R + n1

        pre2_spikes = np.tile(np.concatenate([[pre_spikes[IDX_BIAS[1]]], n1]), N_H).astype(np.int64)
        k2 = RNG.binomial(pre2_spikes, P[G1:G2])
        u[IDX_H2] += n1
        u[IDX_H2] += np.bincount(post2 - IDX_H2[0], SIGN[G1:G2] * k2, minlength=N_H)
        E_trace[G1:G2] += -E_trace[G1:G2] * DT / TAU_E + pre2_spikes
        n2 = np.floor(u[IDX_H2] / THETA).clip(0.0, None).astype(np.int64)
        u[IDX_H2] -= n2 * THETA
        r_est[IDX_H2] += -r_est[IDX_H2] * DT / TAU_R + n2

        pre3_spikes = np.concatenate([[pre_spikes[IDX_BIAS[2]]], n2]).astype(np.int64)
        k3 = RNG.binomial(pre3_spikes, P[G2:])
        u[IDX_OUT] += float(np.sum(SIGN[G2:] * k3))
        E_trace[G2:] += -E_trace[G2:] * DT / TAU_E + pre3_spikes
        n_out = max(int(np.floor(u[IDX_OUT] / THETA)), 0)
        u[IDX_OUT] -= n_out * THETA
        r_est[IDX_OUT] += -r_est[IDX_OUT] * DT / TAU_R + n_out
        f_est += -f_est * DT / TAU_F + n_out

        w3 = SIGN[G2:] * P[G2:] / THETA
        w2 = (SIGN[G1:G2] * P[G1:G2] / THETA).reshape(N_H, -1)
        w1 = (SIGN[:G1] * P[:G1] / THETA).reshape(N_H, 3)
        a1_mf = np.maximum(w1 @ np.concatenate([R * x, [BIAS_RATE]]), 0.0)
        a2_mf = np.maximum(w2 @ np.concatenate([[BIAS_RATE], a1_mf]) + a1_mf / THETA, 0.0)
        a3_mf = np.maximum(w3 @ np.concatenate([[BIAS_RATE], a2_mf]), 0.0)
        delta_out = a3_mf - TARGET * y
        delta_h2 = (a2_mf > 0) * w3[1:] * delta_out
        delta_h1 = (a1_mf > 0) * (w2[:, 1:].T @ delta_h2 + delta_h2 / THETA)

        delta_full = np.zeros(N_NEURONS)
        delta_full[IDX_H1] = delta_h1
        delta_full[IDX_H2] = delta_h2
        delta_full[IDX_OUT] = delta_out

        k = RNG.poisson(np.clip(FDA - delta_full[POST], 0.0, None) * DT)
        dp = SIGN * ALPHA * E_trace * (k - C * DT)
        acc_dw += dp / THETA
        P += dp
        P = np.clip(P, 1e-6, 1.0 - 1e-6)
    losses[smp] = 0.5 * ((f_est / TAU_F - TARGET * y) / TARGET) ** 2
    if (smp + 1) % WIN == 0:
        ng = np.linalg.norm(acc_g)
        ndw = np.linalg.norm(acc_dw)
        cos = float(np.dot(acc_g, -acc_dw) / (ng * ndw + 1e-30))
        cos_hist.append((smp + 1, cos, ng / WIN, ndw / WIN))
        print(f"window {smp + 1 - WIN + 1:5d}-{smp + 1:5d}  cos(sum g, -dW) = {cos:+.3f}"
              f"  mean|g| = {ng / WIN:8.1f}  mean|dW/sample| = {ndw / WIN:.5f}  loss = {np.mean(losses[smp - WIN + 1:smp + 1]):.4f}")
        acc_dw = np.zeros(N_SYN)
        acc_g = np.zeros(N_SYN)

print(f"done in {time.time() - t0:.1f} s")

cos_vals = np.array([c for _, c, _, _ in cos_hist])
print(f"cosine over windows: mean {cos_vals.mean():.3f}, min {cos_vals.min():.3f}, max {cos_vals.max():.3f}, "
      f"last 5: {[f'{c:+.2f}' for c in cos_vals[-5:]]}")

g_all = np.zeros(N_SYN)
dw_all = np.zeros(N_SYN)
u = np.zeros(N_NEURONS)
r_est = np.zeros(N_NEURONS)
f_est = 0.0
E_trace = np.zeros(N_SYN)
schedule2 = RNG.integers(0, 4, N_SAMPLES)
for smp in range(N_SAMPLES):
    x = X[schedule2[smp]]
    y = Y[schedule2[smp]]
    g_all += grad_w(P)
    for _ in range(STEPS_PER_SAMPLE):
        pre_spikes = np.zeros(N_NEURONS)
        pre_spikes[IDX_IN] = RNG.poisson(R * x * DT)
        pre_spikes[IDX_BIAS] = RNG.poisson(BIAS_RATE * DT)
        k1 = RNG.binomial(pre_spikes[pre1].astype(np.int64), P[:G1])
        u[IDX_H1] += np.bincount(post1 - IDX_H1[0], SIGN[:G1] * k1, minlength=N_H)
        E_trace[:G1] += -E_trace[:G1] * DT / TAU_E + pre_spikes[pre1]
        n1 = np.floor(u[IDX_H1] / THETA).clip(0.0, None).astype(np.int64)
        u[IDX_H1] -= n1 * THETA
        pre2_spikes = np.tile(np.concatenate([[pre_spikes[IDX_BIAS[1]]], n1]), N_H).astype(np.int64)
        k2 = RNG.binomial(pre2_spikes, P[G1:G2])
        u[IDX_H2] += n1
        u[IDX_H2] += np.bincount(post2 - IDX_H2[0], SIGN[G1:G2] * k2, minlength=N_H)
        E_trace[G1:G2] += -E_trace[G1:G2] * DT / TAU_E + pre2_spikes
        n2 = np.floor(u[IDX_H2] / THETA).clip(0.0, None).astype(np.int64)
        u[IDX_H2] -= n2 * THETA
        pre3_spikes = np.concatenate([[pre_spikes[IDX_BIAS[2]]], n2]).astype(np.int64)
        k3 = RNG.binomial(pre3_spikes, P[G2:])
        u[IDX_OUT] += float(np.sum(SIGN[G2:] * k3))
        E_trace[G2:] += -E_trace[G2:] * DT / TAU_E + pre3_spikes
        n_out = max(int(np.floor(u[IDX_OUT] / THETA)), 0)
        u[IDX_OUT] -= n_out * THETA
        f_est += -f_est * DT / TAU_F + n_out
        w3 = SIGN[G2:] * P[G2:] / THETA
        w2 = (SIGN[G1:G2] * P[G1:G2] / THETA).reshape(N_H, -1)
        w1 = (SIGN[:G1] * P[:G1] / THETA).reshape(N_H, 3)
        a1_mf = np.maximum(w1 @ np.concatenate([R * x, [BIAS_RATE]]), 0.0)
        a2_mf = np.maximum(w2 @ np.concatenate([[BIAS_RATE], a1_mf]) + a1_mf / THETA, 0.0)
        a3_mf = np.maximum(w3 @ np.concatenate([[BIAS_RATE], a2_mf]), 0.0)
        delta_out = a3_mf - TARGET * y
        delta_h2 = (a2_mf > 0) * w3[1:] * delta_out
        delta_h1 = (a1_mf > 0) * (w2[:, 1:].T @ delta_h2 + delta_h2 / THETA)
        delta_full = np.zeros(N_NEURONS)
        delta_full[IDX_H1] = delta_h1
        delta_full[IDX_H2] = delta_h2
        delta_full[IDX_OUT] = delta_out
        k = RNG.poisson(np.clip(FDA - delta_full[POST], 0.0, None) * DT)
        dp = SIGN * ALPHA * E_trace * (k - C * DT)
        dw_all += dp / THETA
        P += dp
        P = np.clip(P, 1e-6, 1.0 - 1e-6)
    _ = f_est
cos_full = float(np.dot(g_all, -dw_all) / (np.linalg.norm(g_all) * np.linalg.norm(dw_all) + 1e-30))
print(f"whole-trajectory ({N_SAMPLES} samples): cos(sum g, -sum dW) = {cos_full:+.3f}")
g1 = g_all[:G1]
g2 = g_all[G1:G2]
g3 = g_all[G2:]
d1 = dw_all[:G1]
d2 = dw_all[G1:G2]
d3 = dw_all[G2:]
for name, gx, dx in (("W1", g1, d1), ("W2", g2, d2), ("W3", g3, d3)):
    c = float(np.dot(gx, -dx) / (np.linalg.norm(gx) * np.linalg.norm(dx) + 1e-30))
    print(f"  layer {name}: cos = {c:+.3f}  |g| = {np.linalg.norm(gx):9.1f}  |dW| = {np.linalg.norm(dx):.4f}")
