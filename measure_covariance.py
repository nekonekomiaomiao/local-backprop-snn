import numpy as np

RNG = np.random.default_rng(0)

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
ALPHA = 2.5e-6
P_INIT = 0.3
R_GATE = 1.0
N_SAMPLES = 1500

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

schedule = RNG.integers(0, 4, N_SAMPLES)


def run(sample_t):
    global P
    steps = int(sample_t / DT)
    cut = max(1, int(0.25 * steps))
    u = np.zeros(N_NEURONS)
    r_est = np.zeros(N_NEURONS)
    f_est = 0.0
    E_trace = np.zeros(N_SYN)
    acc0 = np.zeros((2, N_SYN, 5))
    n_steps = np.array([cut * N_SAMPLES, (steps - cut) * N_SAMPLES])
    sign_ok = np.zeros(N_SAMPLES)
    losses = np.zeros(N_SAMPLES)
    for smp in range(N_SAMPLES):
        x = X[schedule[smp]]
        y = Y[schedule[smp]]
        P_before = P.copy()
        for s in range(steps):
            ph = 0 if s < cut else 1
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

            d = delta_full[POST]
            e = E_trace
            acc0[ph, :, 0] += e
            acc0[ph, :, 1] += d
            acc0[ph, :, 2] += e * d
            acc0[ph, :, 3] += e * e
            acc0[ph, :, 4] += d * d

            k = RNG.poisson(np.clip(FDA - d, 0.0, None) * DT)
            P += SIGN * ALPHA * e * (k - C * DT)
            P = np.clip(P, 1e-6, 1.0 - 1e-6)
        losses[smp] = 0.5 * ((f_est / TAU_F - TARGET * y) / TARGET) ** 2
        exp_sign = np.sign(-SIGN * delta_full[POST])
        mask = np.abs(delta_full[POST]) > 1.0
        if mask.sum() > 0:
            sign_ok[smp] = np.mean(np.sign(P - P_before)[mask] == exp_sign[mask])
        else:
            sign_ok[smp] = 0.5
    return acc0, losses, sign_ok, n_steps


for sample_t in [0.5, 1.0, 2.0, 4.0]:
    P = np.full(N_SYN, P_INIT)
    P[G2] = 0.6
    acc, losses, sign_ok, n_steps = run(sample_t)
    print(f"sample duration = {sample_t} s")
    for gname, g in (("W1", slice(0, G1)), ("W2", slice(G1, G2)), ("W3", slice(G2, N_SYN))):
        line = f"  {gname}: "
        for ph, pname in ((0, "first25%"), (1, "last75%")):
            n = n_steps[ph]
            cov = acc[ph, g, 2] / n - (acc[ph, g, 0] / n) * (acc[ph, g, 1] / n)
            var_e = acc[ph, g, 3] / n - (acc[ph, g, 0] / n) ** 2
            var_d = acc[ph, g, 4] / n - (acc[ph, g, 1] / n) ** 2
            corr = cov / (np.sqrt(var_e * var_d) + 1e-12)
            line += f"{pname}: corr(e,d)={np.mean(corr):+.3f} "
        print(line)
    print(f"  loss(last 300): {np.mean(losses[-300:]):.4f}   p-sign(last 300): {np.mean(sign_ok[-300:]):.3f}")
