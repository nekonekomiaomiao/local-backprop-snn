import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
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
R_GATE = 1.0

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


def grad_w(Pv):
    w1 = (SIGN[:G1] * Pv[:G1] / THETA).reshape(N_H, 3)
    w2 = (SIGN[G1:G2] * Pv[G1:G2] / THETA).reshape(N_H, N_H + 1)
    w3 = SIGN[G2:] * Pv[G2:] / THETA
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


print("Local online backpropagation on a probabilistic-synapse spiking network (residual topology, XOR task)")
print(f"theta={THETA}  f_da=C={FDA} Hz  tau_e={TAU_E} s  tau_f={TAU_F} s  alpha={ALPHA}  dt={DT} s  sample={SAMPLE_T} s")
print(f"topology: in(2) -> dense({N_H}) -> residual({N_H}) -> out(1), 3 bias inputs; {N_SYN} plastic synapses; seed {SEED}")

t0 = time.time()
u = np.zeros(N_NEURONS)
r_est = np.zeros(N_NEURONS)
f_est = 0.0
E_trace = np.zeros(N_SYN)
schedule = RNG.integers(0, 4, N_SAMPLES)
losses = np.zeros(N_SAMPLES)
accs = np.zeros(N_SAMPLES)
cosines = np.zeros(N_SAMPLES)
sign_ok = np.zeros(N_SAMPLES)
cum_dw = np.zeros(N_SYN)
cum_g = np.zeros(N_SYN)
win_dw = np.zeros(N_SYN)
win_g = np.zeros(N_SYN)
cum_cos = np.zeros(N_SAMPLES)
win_cos_hist = []

for smp in range(N_SAMPLES):
    x = X[schedule[smp]]
    y = Y[schedule[smp]]
    P_before = P.copy()
    g_ex = grad_w(P_before)
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

        gate_h2 = (r_est[IDX_H2] / TAU_R) > R_GATE
        gate_h1 = (r_est[IDX_H1] / TAU_R) > R_GATE
        w3 = SIGN[G2:] * P[G2:] / THETA
        w2 = (SIGN[G1:G2] * P[G1:G2] / THETA).reshape(N_H, -1)
        delta_out = f_est / TAU_F - TARGET * y
        delta_h2 = gate_h2 * w3[1:] * delta_out
        delta_h1 = gate_h1 * (w2[:, 1:].T @ delta_h2 + delta_h2 / THETA)

        delta_full = np.zeros(N_NEURONS)
        delta_full[IDX_H1] = delta_h1
        delta_full[IDX_H2] = delta_h2
        delta_full[IDX_OUT] = delta_out

        k = RNG.poisson(np.clip(FDA - delta_full[POST], 0.0, None) * DT)
        P += SIGN * ALPHA * E_trace * (k - C * DT)
        P = np.clip(P, 1e-6, 1.0 - 1e-6)

    losses[smp] = 0.5 * ((f_est / TAU_F - TARGET * y) / TARGET) ** 2
    accs[smp] = float(((f_est / TAU_F) > TARGET / 2.0) == (y == 1.0))
    g_emp = SIGN * (P - P_before) / THETA
    cosines[smp] = float(np.dot(g_emp, -g_ex)) / (np.linalg.norm(g_emp) * np.linalg.norm(g_ex) + 1e-12)
    exp_sign = np.sign(-SIGN * delta_full[POST])
    mask = np.abs(delta_full[POST]) > 1.0
    if mask.sum() > 0:
        sign_ok[smp] = np.mean(np.sign(P - P_before)[mask] == exp_sign[mask])
    else:
        sign_ok[smp] = 0.5
    cum_dw += g_emp
    cum_g += g_ex
    win_dw += g_emp
    win_g += g_ex
    cum_cos[smp] = float(np.dot(cum_g, -cum_dw)) / (np.linalg.norm(cum_g) * np.linalg.norm(cum_dw) + 1e-12)
    if (smp + 1) % 500 == 0:
        win_cos = float(np.dot(win_g, -win_dw)) / (np.linalg.norm(win_g) * np.linalg.norm(win_dw) + 1e-12)
        print(f"sample {smp + 1:6d}  loss(roll) {np.mean(losses[smp-99:smp+1]):.4f}  acc(roll) {np.mean(accs[smp-99:smp+1]):.3f}"
              f"  cos/sample(roll) {np.mean(cosines[smp-99:smp+1]):+.3f}  cos/window(500) {win_cos:+.3f}"
              f"  cos/run {cum_cos[smp]:+.3f}  p-sign(roll) {np.mean(sign_ok[smp-99:smp+1]):.3f}")
        win_dw = np.zeros(N_SYN)
        win_g = np.zeros(N_SYN)
        win_cos_hist.append(win_cos)

train_time = time.time() - t0
print(f"training done in {train_time:.1f} s;  release probabilities p in [{P.min():.3f}, {P.max():.3f}]")

print("evaluation on the four XOR patterns (learning frozen):")
for i in range(4):
    x, y = X[i], Y[i]
    u = np.zeros(N_NEURONS)
    r_est = np.zeros(N_NEURONS)
    f_est = 0.0
    for _ in range(STEPS_PER_SAMPLE):
        pre_spikes = np.zeros(N_NEURONS)
        pre_spikes[IDX_IN] = RNG.poisson(R * x * DT)
        pre_spikes[IDX_BIAS] = RNG.poisson(BIAS_RATE * DT)
        k1 = RNG.binomial(pre_spikes[pre1].astype(np.int64), P[:G1])
        u[IDX_H1] += np.bincount(post1 - IDX_H1[0], SIGN[:G1] * k1, minlength=N_H)
        n1 = np.floor(u[IDX_H1] / THETA).clip(0.0, None).astype(np.int64)
        u[IDX_H1] -= n1 * THETA
        pre2_spikes = np.tile(np.concatenate([[pre_spikes[IDX_BIAS[1]]], n1]), N_H).astype(np.int64)
        k2 = RNG.binomial(pre2_spikes, P[G1:G2])
        u[IDX_H2] += n1
        u[IDX_H2] += np.bincount(post2 - IDX_H2[0], SIGN[G1:G2] * k2, minlength=N_H)
        n2 = np.floor(u[IDX_H2] / THETA).clip(0.0, None).astype(np.int64)
        u[IDX_H2] -= n2 * THETA
        pre3_spikes = np.concatenate([[pre_spikes[IDX_BIAS[2]]], n2]).astype(np.int64)
        k3 = RNG.binomial(pre3_spikes, P[G2:])
        u[IDX_OUT] += float(np.sum(SIGN[G2:] * k3))
        n_out = max(int(np.floor(u[IDX_OUT] / THETA)), 0)
        u[IDX_OUT] -= n_out * THETA
        r_est[IDX_H1] += -r_est[IDX_H1] * DT / TAU_R + n1
        r_est[IDX_H2] += -r_est[IDX_H2] * DT / TAU_R + n2
        r_est[IDX_OUT] += -r_est[IDX_OUT] * DT / TAU_R + n_out
        f_est += -f_est * DT / TAU_F + n_out
    pred = (f_est / TAU_F) > TARGET / 2.0
    print(f"  x = ({int(x[0])},{int(x[1])})  y = {int(y)}  f_out = {f_est / TAU_F:6.1f} Hz  pred = {int(pred)}  {'OK' if pred == (y == 1.0) else 'FAIL'}")

EPOCH = 4
n_epochs = N_SAMPLES // EPOCH
e_loss = losses[: n_epochs * EPOCH].reshape(n_epochs, EPOCH).mean(axis=1)
e_acc = accs[: n_epochs * EPOCH].reshape(n_epochs, EPOCH).mean(axis=1)
e_cos = cosines[: n_epochs * EPOCH].reshape(n_epochs, EPOCH).mean(axis=1)
e_sign = sign_ok[: n_epochs * EPOCH].reshape(n_epochs, EPOCH).mean(axis=1)
e_cum = cum_cos[: n_epochs * EPOCH:EPOCH]
win = 25
roll_loss = np.convolve(e_loss, np.ones(win) / win, mode="valid")
roll_acc = np.convolve(e_acc, np.ones(win) / win, mode="valid")
roll_cos = np.convolve(e_cos, np.ones(win) / win, mode="valid")
roll_sign = np.convolve(e_sign, np.ones(win) / win, mode="valid")
win_cos_hist = np.array(win_cos_hist[: len(roll_loss)])
wincos_x = np.arange(len(win_cos_hist)) * 500 + 250

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].plot(np.arange(len(roll_loss)) + win - 1, roll_loss, lw=1.5, color="tab:blue")
axes[0, 0].set_xlabel("epoch (4 samples)")
axes[0, 0].set_ylabel("loss")
axes[0, 0].set_title("loss (rolling mean)")
axes[0, 1].plot(np.arange(len(roll_acc)) + win - 1, roll_acc, lw=1.5, color="tab:green")
axes[0, 1].set_ylim(0.0, 1.05)
axes[0, 1].set_xlabel("epoch")
axes[0, 1].set_ylabel("accuracy")
axes[0, 1].set_title("XOR accuracy (rolling mean)")
axes[1, 0].plot(np.arange(len(roll_cos)) + win - 1, roll_cos, lw=1.2, color="tab:red", label="cos(sample update, -gradient), rolling")
axes[1, 0].plot(wincos_x, win_cos_hist, lw=1.8, color="tab:purple", label="cos(500-sample cumulative update, -gradient)")
axes[1, 0].axhline(0.0, color="gray", lw=0.8, ls="--")
axes[1, 0].set_ylim(-1.0, 1.0)
axes[1, 0].set_xlabel("epoch")
axes[1, 0].set_ylabel("cosine")
axes[1, 0].set_title("gradient alignment")
axes[1, 0].legend(loc="upper right", fontsize=8)
axes[1, 1].plot(np.arange(len(roll_sign)) + win - 1, roll_sign, lw=1.5, color="tab:orange")
axes[1, 1].axhline(0.5, color="gray", lw=0.8, ls="--")
axes[1, 1].set_ylim(0.0, 1.0)
axes[1, 1].set_xlabel("epoch")
axes[1, 1].set_ylabel("fraction")
axes[1, 1].set_title("p-update sign agreement with SDE expectation")
plt.tight_layout()
plt.savefig("xor_local_bp_result.png", dpi=150)
print("figure saved to xor_local_bp_result.png")
print(f"final: cos(sample, -grad) rolling {np.mean(cosines[-100:]):+.3f}, "
      f"cos(cumulative, -grad) {cum_cos[-1]:+.3f}, p-sign rolling {np.mean(sign_ok[-100:]):.3f}")
