import sys
import numpy as np

rng = np.random.default_rng(0)
rng2 = np.random.default_rng(1)

R = float(sys.argv[1]) if len(sys.argv) > 1 else 200.0
BIAS_RATE = R * 0.15
THETA = 1.0
TAU_E = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
TAU_F = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
TAU_DELAY = float(sys.argv[4]) if len(sys.argv) > 4 else 0.4
FDA = 2.5 * R
C = FDA
DT = 0.02
SAMPLE_T = 8.0
ALPHA = float(sys.argv[5]) if len(sys.argv) > 5 else 6e-7
P_INIT = 0.3
N_H = 16
N_SAMPLES = 3000

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

schedule = rng.integers(0, 4, N_SAMPLES)
STEPS = int(SAMPLE_T / DT)
DELAY_STEPS = max(1, int(round(TAU_DELAY / DT)))


def weights(Pv):
    return ((SIGN[:G1] * Pv[:G1] / THETA).reshape(N_H, 3),
            (SIGN[G1:G2] * Pv[G1:G2] / THETA).reshape(N_H, N_H + 1),
            SIGN[G2:] * Pv[G2:] / THETA)


def forward(Pv, x):
    w1, w2, w3 = weights(Pv)
    a1 = np.maximum(w1 @ np.concatenate([R * x, [BIAS_RATE]]), 0.0)
    a2 = np.maximum(w2 @ np.concatenate([[BIAS_RATE], a1]), 0.0) + a1
    a3 = np.maximum(w3 @ np.concatenate([[BIAS_RATE], a2]), 0.0)
    return a1, a2, a3


def delta_full(Pv, x, y, a1, a2, a3, d3_out=None):
    w1, w2, w3 = weights(Pv)
    d3 = a3 - R * y if d3_out is None else d3_out
    da2 = d3 * w3[1:]
    d2 = (a2 - a1 > 0) * da2
    d1 = (a1 > 0) * (w2[:, 1:].T @ d2 + da2)
    d = np.zeros(6 + 2 * N_H)
    d[IDX_H1] = d1
    d[IDX_H2] = d2
    d[IDX_OUT] = d3
    return d


def run(mode):
    global P
    P = np.full(N_SYN, P_INIT)
    P[G2] = 0.6
    f_est = 0.0
    delta_buf = np.zeros(DELAY_STEPS)
    E_trace = np.zeros(N_SYN)
    t_step = 0
    acc_roll = 0.0
    for smp in range(N_SAMPLES):
        x, y = X[schedule[smp]], Y[schedule[smp]]
        a1, a2, a3 = forward(P, x)

        if "e" in mode:
            for _ in range(STEPS):
                pre = np.zeros(6 + 2 * N_H)
                pre[IDX_BIAS] = rng2.poisson(BIAS_RATE * DT)
                pre[IDX_H1] = rng2.poisson(a1 * DT)
                pre[IDX_H2] = rng2.poisson(a2 * DT)
                pre[IDX_OUT] = rng2.poisson(a3 * DT)
                E_trace += -E_trace * DT / TAU_E + pre[PRE]
            e_mean = E_trace
        else:
            e_mean = np.zeros(N_SYN)
            e_mean[:G1] = TAU_E * np.tile(np.concatenate([R * x, [BIAS_RATE]]), N_H)
            e_mean[G1:G2] = TAU_E * np.tile(np.concatenate([[BIAS_RATE], a1]), N_H)
            e_mean[G2:] = TAU_E * np.concatenate([[BIAS_RATE], a2])

        if "f" in mode:
            for _ in range(STEPS):
                delta_out_now = f_est - R * y
                delta_out = delta_buf[t_step % DELAY_STEPS]
                delta_buf[t_step % DELAY_STEPS] = delta_out_now
                t_step += 1
                f_est += -f_est * DT / TAU_F + rng2.poisson(a3 * DT)
            d = delta_full(P, x, y, a1, a2, a3, d3_out=delta_out)
        else:
            t_step += STEPS
            d = delta_full(P, x, y, a1, a2, a3)

        if "k" in mode:
            tot = np.zeros(N_SYN)
            for _ in range(STEPS):
                k = rng2.poisson(np.clip(FDA - d[POST], 0.0, None) * DT)
                tot += k
            eff = tot - C * SAMPLE_T
        else:
            eff = -d[POST] * SAMPLE_T

        P += SIGN * ALPHA * e_mean * eff
        P = np.clip(P, 1e-6, 1 - 1e-6)
        a1, a2, a3 = forward(P, x)
        acc_roll += float((a3 > R / 2.0) == (y == 1.0))
        if (smp + 1) % 2000 == 0:
            print(f"  [{mode}] sample {smp+1:6d}  acc(2000) {acc_roll/2000:.3f}")
            acc_roll = 0.0
    return [forward(P, X[i])[2] for i in range(4)]


for mode in ["clean", "kfe"]:
    out = run(mode)
    ok = all((out[i] > R / 2.0) == (Y[i] == 1.0) for i in range(4))
    print(f"mode={mode or 'clean':5s}  outputs: " + "  ".join(f"{v:6.1f}" for v in out) +
          f"  {'ALL-OK' if ok else 'FAIL'}")
