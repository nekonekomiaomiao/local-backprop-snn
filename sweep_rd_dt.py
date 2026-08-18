import sys
import os
import time
import csv
import numpy as np

DT = 0.02
THETA = 1.0
TAU_E = 0.2
TAU_R = 0.1
TAU_F = 0.1
ALPHA = 2.5e-6
P_INIT = 0.3

DT_LIST = [0.5, 1.0, 2.0, 4.0]

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
GNAMES = ["W1", "W2", "W3"]
GS = [slice(0, G1), slice(G1, G2), slice(G2, N_SYN)]

R_VALUE = float(sys.argv[1]) if len(sys.argv) > 1 else 200.0
MODE = sys.argv[2] if len(sys.argv) > 2 else "both"
N_SAMPLES = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
OUTDIR = "sweep_out"
os.makedirs(OUTDIR, exist_ok=True)

DOCSEED = "docseed" in sys.argv
if DOCSEED:
    OUTDIR = "sweep_out_doc"
    os.makedirs(OUTDIR, exist_ok=True)

rng_fixed = np.random.default_rng(2026)
SIGN = rng_fixed.choice([-1.0, 1.0], N_SYN)
SCHEDULE = rng_fixed.integers(0, 4, N_SAMPLES)

if DOCSEED:
    rng_fixed = np.random.default_rng(0)
    SIGN = rng_fixed.choice([-1.0, 1.0], N_SYN)
    SCHEDULE = rng_fixed.integers(0, 4, N_SAMPLES)
    N_SAMPLES = 1500


def group_stats(cov, i0, i1, g, phases, steps, cut):
    """per-synapse corr(cov)/cov mean over group, phases given (subset of (0,1))"""
    step_counts = {0: cut, 1: steps - cut}
    sub = cov[i0:i1, :, :][:, phases].sum(axis=(0, 1))
    n = (i1 - i0) * sum(step_counts[p] for p in phases)
    me = sub[0] / n
    md = sub[1] / n
    cov_ed = sub[2] / n - me * md
    vare = sub[3] / n - me * me
    vard = sub[4] / n - md * md
    corr = cov_ed / np.sqrt(vare * vard + 1e-12)
    return float(corr[g].mean()), float(cov_ed[g].mean()), float(me[g].mean()), float(md[g].mean())


def run_cfg(R, sample_t, mode, n_samples):
    k = R / 200.0
    BIAS_RATE = 30.0 * k if mode == "scaled" else 30.0
    TARGET = 200.0 * k if mode == "scaled" else 200.0
    FDA = 500.0 * k if mode == "scaled" else 500.0
    C = FDA
    R_GATE = 1.0 * k if mode == "scaled" else 1.0
    steps = int(sample_t / DT)
    cut = max(1, int(0.25 * steps))
    if DOCSEED:
        rng = rng_fixed
    else:
        seed = int(R) + int(round(sample_t * 100)) + (70000 if mode == "fixed" else 0)
        rng = np.random.default_rng(seed)

    P = np.full(N_SYN, P_INIT)
    P[G2] = 0.6
    u = np.zeros(N_NEURONS)
    r_est = np.zeros(N_NEURONS)
    f_est = 0.0
    E_trace = np.zeros(N_SYN)
    losses = np.zeros(n_samples)
    accs = np.zeros(n_samples)
    sign_ok = np.zeros(n_samples)
    cov = np.zeros((n_samples, 2, 5, N_SYN))

    t0 = time.time()
    for smp in range(n_samples):
        x = X[SCHEDULE[smp]]
        y = Y[SCHEDULE[smp]]
        P_before = P.copy()
        for s in range(steps):
            ph = 0 if s < cut else 1
            pre_spikes = np.zeros(N_NEURONS)
            pre_spikes[IDX_IN] = rng.poisson(R * x * DT)
            pre_spikes[IDX_BIAS] = rng.poisson(BIAS_RATE * DT)

            k1 = rng.binomial(pre_spikes[pre1].astype(np.int64), P[:G1])
            u[IDX_H1] += np.bincount(post1 - IDX_H1[0], SIGN[:G1] * k1, minlength=N_H)
            E_trace[:G1] += -E_trace[:G1] * DT / TAU_E + pre_spikes[pre1]
            n1 = np.floor(u[IDX_H1] / THETA).clip(0.0, None).astype(np.int64)
            u[IDX_H1] -= n1 * THETA
            r_est[IDX_H1] += -r_est[IDX_H1] * DT / TAU_R + n1

            pre2_spikes = np.tile(np.concatenate([[pre_spikes[IDX_BIAS[1]]], n1]), N_H).astype(np.int64)
            k2 = rng.binomial(pre2_spikes, P[G1:G2])
            u[IDX_H2] += n1
            u[IDX_H2] += np.bincount(post2 - IDX_H2[0], SIGN[G1:G2] * k2, minlength=N_H)
            E_trace[G1:G2] += -E_trace[G1:G2] * DT / TAU_E + pre2_spikes
            n2 = np.floor(u[IDX_H2] / THETA).clip(0.0, None).astype(np.int64)
            u[IDX_H2] -= n2 * THETA
            r_est[IDX_H2] += -r_est[IDX_H2] * DT / TAU_R + n2

            pre3_spikes = np.concatenate([[pre_spikes[IDX_BIAS[2]]], n2]).astype(np.int64)
            k3 = rng.binomial(pre3_spikes, P[G2:])
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
            cov[smp, ph, 0] += e
            cov[smp, ph, 1] += d
            cov[smp, ph, 2] += e * d
            cov[smp, ph, 3] += e * e
            cov[smp, ph, 4] += d * d

            kk = rng.poisson(np.clip(FDA - d, 0.0, None) * DT)
            P += SIGN * ALPHA * E_trace * (kk - C * DT)
            P = np.clip(P, 1e-6, 1.0 - 1e-6)

        losses[smp] = 0.5 * ((f_est / TAU_F - TARGET * y) / TARGET) ** 2
        accs[smp] = float(((f_est / TAU_F) > TARGET / 2.0) == (y == 1.0))
        exp_sign = np.sign(-SIGN * delta_full[POST])
        mask = np.abs(delta_full[POST]) > 1.0
        if mask.sum() > 0:
            sign_ok[smp] = np.mean(np.sign(P - P_before)[mask] == exp_sign[mask])
        else:
            sign_ok[smp] = 0.5
        if (smp + 1) % 250 == 0:
            print(f"  [R={R:g} dT={sample_t} {mode}] sample {smp + 1}/{n_samples}  "
                  f"loss(roll100) {np.mean(losses[smp - 99:smp + 1]):.4f}  ({time.time() - t0:.0f} s)", flush=True)
    wall = time.time() - t0

    n_w = 4
    win_lo = n_samples // n_w
    w_corr = np.zeros((n_w, 3))
    for wi in range(n_w):
        for gi in range(3):
            w_corr[wi, gi], _, _, _ = group_stats(cov, wi * win_lo, (wi + 1) * win_lo, GS[gi], (0, 1), steps, cut)

    row = {}
    for gi, gname in enumerate(GNAMES):
        row[f"corr_{gname}_all"], row[f"cov_{gname}_all"], _, _ = group_stats(cov, 0, n_samples, GS[gi], (0, 1), steps, cut)
        row[f"corr_{gname}_fin"], _, _, _ = group_stats(cov, n_samples - 800, n_samples, GS[gi], (0, 1), steps, cut)
    row["corr_W3_f25"], _, _, _ = group_stats(cov, n_samples - 800, n_samples, GS[2], (0,), steps, cut)
    row["corr_W3_l75"], _, _, _ = group_stats(cov, n_samples - 800, n_samples, GS[2], (1,), steps, cut)

    roll = np.convolve(losses, np.ones(100) / 100, mode="valid")
    row["loss_last300"] = float(np.mean(losses[-300:]))
    row["loss_last500"] = float(np.mean(losses[-500:]))
    row["loss_min_roll"] = float(roll[len(roll) // 2:].min())
    row["acc_last500"] = float(np.mean(accs[-500:]))
    row["p_sign_last500"] = float(np.mean(sign_ok[-500:]))

    print(f"[R={R:g} dT={sample_t} {mode}] done in {wall:.0f} s | "
          f"corr W1/W2/W3 (all) = {row['corr_W1_all']:+.3f} / {row['corr_W2_all']:+.3f} / {row['corr_W3_all']:+.3f} | "
          f"loss_last500 = {row['loss_last500']:.4f}, acc = {row['acc_last500']:.3f}", flush=True)

    np.savez(os.path.join(OUTDIR, f"cfg_{mode}_R{int(R)}_dT{sample_t}.npz"),
             losses=losses, accs=accs, sign_ok=sign_ok, w_corr=w_corr,
             R=R, dT=sample_t, mode=mode, TARGET=TARGET, FDA=FDA, steps=steps)
    return row


def main():
    cfgs = [(dT, "scaled") for dT in DT_LIST]
    if R_VALUE != 200.0:
        cfgs.append((2.0, "fixed"))
    if MODE == "scaled":
        cfgs = [(dT, "scaled") for dT in DT_LIST]
    elif MODE == "fixed":
        cfgs = [(2.0, "fixed")]

    summary = os.path.join(OUTDIR, "sweep_summary.csv")
    write_header = not os.path.exists(summary)
    f = open(summary, "a", newline="")
    w = csv.writer(f)
    if write_header:
        w.writerow(["R", "mode", "dT", "sim_time_s", "corr_W1_all", "corr_W2_all", "corr_W3_all",
                    "cov_W3_all", "corr_W1_fin", "corr_W2_fin", "corr_W3_fin", "corr_W3_f25", "corr_W3_l75",
                    "loss_last300", "loss_last500", "loss_min_roll", "acc_last500", "p_sign_last500", "norm_loss"])
    for dT, mode in cfgs:
        row = run_cfg(R_VALUE, dT, mode, N_SAMPLES)
        k = R_VALUE / 200.0
        w.writerow([f"{R_VALUE:g}", mode, f"{dT:g}", f"{N_SAMPLES * dT:g}",
                    f"{row['corr_W1_all']:.4f}", f"{row['corr_W2_all']:.4f}", f"{row['corr_W3_all']:.4f}",
                    f"{row['cov_W3_all']:.4f}", f"{row['corr_W1_fin']:.4f}", f"{row['corr_W2_fin']:.4f}",
                    f"{row['corr_W3_fin']:.4f}", f"{row['corr_W3_f25']:.4f}", f"{row['corr_W3_l75']:.4f}",
                    f"{row['loss_last300']:.4f}", f"{row['loss_last500']:.4f}", f"{row['loss_min_roll']:.4f}",
                    f"{row['acc_last500']:.4f}", f"{row['p_sign_last500']:.4f}",
                    f"{row['loss_last500'] * np.sqrt(k * dT / 2.0):.4f}"])
        f.flush()
    f.close()


if __name__ == "__main__":
    main()
