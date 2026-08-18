import sys
import numpy as np

rng = np.random.default_rng(0)

R = 200.0
BIAS_RATE = 30.0
THETA = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
DT = 0.02
N_H = 16
N_SYN = 337
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
G1 = len(pre1)
G2 = G1 + len(pre2)

SIGN = rng.choice([-1.0, 1.0], N_SYN)
P = np.full(N_SYN, float(sys.argv[2]) if len(sys.argv) > 2 else 0.3)
P[G2] = 0.6

w1 = (SIGN[:G1] * P[:G1] / THETA).reshape(N_H, 3)
w2 = (SIGN[G1:G2] * P[G1:G2] / THETA).reshape(N_H, N_H + 1)
w3 = SIGN[G2:] * P[G2:] / THETA

x = np.array([0.0, 1.0])
f_in = np.concatenate([R * x, [BIAS_RATE]])
a1_mf = np.maximum(w1 @ f_in, 0.0)
a2_mf = np.maximum(w2 @ np.concatenate([[BIAS_RATE], a1_mf]) + a1_mf / THETA, 0.0)
a3_mf = np.maximum(w3 @ np.concatenate([[BIAS_RATE], a2_mf]), 0.0)
print("mean-field rates (Hz):")
print("  a1:", np.round(a1_mf, 1))
z2_mf = w2 @ np.concatenate([[BIAS_RATE], a1_mf]) + a1_mf / THETA
print("  z2+ a1/th:", np.round(z2_mf, 2))
print("  a2:", np.round(a2_mf, 1))
print("  a3:", round(float(a3_mf), 1))

u = np.zeros(N_NEURONS)
n_spk = np.zeros(N_NEURONS)
w2acc = np.zeros(N_H)
k2acc = np.zeros(G2 - G1)
pre2acc = np.zeros(G2 - G1)
STEPS = 5000
for _ in range(STEPS):
    pre_spikes = np.zeros(N_NEURONS)
    pre_spikes[IDX_IN] = rng.poisson(R * x * DT)
    pre_spikes[IDX_BIAS] = rng.poisson(BIAS_RATE * DT)
    k1 = rng.binomial(pre_spikes[pre1].astype(np.int64), P[:G1])
    u[IDX_H1] += np.bincount(post1 - IDX_H1[0], SIGN[:G1] * k1, minlength=N_H)
    n1 = np.floor(u[IDX_H1] / THETA).clip(0.0, None).astype(np.int64)
    u[IDX_H1] -= n1 * THETA
    pre2_spikes = np.tile(np.concatenate([[pre_spikes[IDX_BIAS[1]]], n1]), N_H).astype(np.int64)
    k2 = rng.binomial(pre2_spikes, P[G1:G2])
    k2acc += k2
    pre2acc += pre2_spikes
    if _ == 1234:
        print("DEBUG step 1234:")
        print("  pre2_spikes[0:17]:", pre2_spikes[0:17])
        print("  k2[0:17]:", k2[0:17])
        print("  SIGN[G1:G1+17]:", SIGN[G1:G1 + 17].astype(int))
        print("  n1:", n1)
        print("  u[IDX_H2] before:", u[IDX_H2].copy())
    u[IDX_H2] += n1
    u[IDX_H2] += np.bincount(post2 - IDX_H2[0], SIGN[G1:G2] * k2, minlength=N_H)
    n2 = np.floor(u[IDX_H2] / THETA).clip(0.0, None).astype(np.int64)
    u[IDX_H2] -= n2 * THETA
    pre3_spikes = np.concatenate([[pre_spikes[IDX_BIAS[2]]], n2]).astype(np.int64)
    k3 = rng.binomial(pre3_spikes, P[G2:])
    u[IDX_OUT] += float(np.sum(SIGN[G2:] * k3))
    n_out = max(int(np.floor(u[IDX_OUT] / THETA)), 0)
    u[IDX_OUT] -= n_out * THETA
    n_spk[IDX_H1] += n1
    n_spk[IDX_H2] += n2
    n_spk[IDX_OUT] += n_out

T = STEPS * DT
print("spiking rates over", T, "s:")
print("  h1:", np.round(n_spk[IDX_H1] / T, 1))
print("  W2-path accumulation:", np.round(w2acc, 2))
print("  h2:", np.round(n_spk[IDX_H2] / T, 1))
print("  out:", round(float(n_spk[IDX_OUT] / T), 1))
j = 0
sl = slice(G1 + j * (N_H + 1), G1 + (j + 1) * (N_H + 1))
print("  post-0 k2 mean/step:", np.round(k2acc[sl] / STEPS, 4))
print("  post-0 pre2 mean/step:", np.round(pre2acc[sl] / STEPS, 4))
print("  post-0 SIGN:", SIGN[sl].astype(int))
print("  post-0 expected s*p*f*dt:", np.round(SIGN[sl] * P[sl] * np.concatenate([[BIAS_RATE], a1_mf]) * DT, 4))
