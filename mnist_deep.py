#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""X7: 深度架构 —— 共享卷积 + 双隐藏全连接（FC32 -> FC32 -> 10）。
机制与 mnist_shared.py 完全一致：概率突触 + 资格迹 + 局部误差（delta -> Poisson）。
argv = <seed> <N> <SAMPLE_T> <R> <alpha> <FDA> <BIAS> <TARGET> <final_eval_n> [DT] [KAPPA] [GAMMA] [RESET] [ISI] [TAU_M] [TAU_E]"""
import sys, time
import numpy as np

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
SAMPLE_T = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
R_IN = float(sys.argv[4]) if len(sys.argv) > 4 else 200.0
ALPHA_IN = float(sys.argv[5]) if len(sys.argv) > 5 else 3e-9
FDA_IN = float(sys.argv[6]) if len(sys.argv) > 6 else 3000.0
BIAS_IN = float(sys.argv[7]) if len(sys.argv) > 7 else 30.0
TARGET_IN = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0
N_FINAL_EVAL = int(sys.argv[9]) if len(sys.argv) > 9 else 1000
DT = float(sys.argv[10]) if len(sys.argv) > 10 else 0.02
KAPPA = float(sys.argv[11]) if len(sys.argv) > 11 else 0.0
GAMMA = float(sys.argv[12]) if len(sys.argv) > 12 else 0.0
RESET_PER_SAMPLE = int(sys.argv[13]) if len(sys.argv) > 13 else 0
ISI_STEPS = int(sys.argv[14]) if len(sys.argv) > 14 else 0
TAU_M = float(sys.argv[15]) if len(sys.argv) > 15 else 0.0
TAU_E = float(sys.argv[16]) if len(sys.argv) > 16 else 0.2

steps = int(round(SAMPLE_T / DT))
THETA = 1.0; FDA = FDA_IN * DT; C = FDA_IN * DT; BIAS_RATE = BIAS_IN * DT
R_GATE = 0.02; TAU_R = 0.02; TAU_F = 0.05; TARGET = TARGET_IN * DT; ALPHA = ALPHA_IN * DT
RNG = np.random.default_rng(SEED)
from mnist_loader import load_mnist
tr, trl = load_mnist(train=True)
te, tel = load_mnist(train=False)

C1, K1 = 4, 5; NFC1, NFC2 = 32, 32; NOUT = 10; N_F = C1 * 12 * 12
OFF_IN = 0; OFF_B1 = 784; OFF_F = 785; OFF_B2 = OFF_F + N_F; OFF_FC1 = OFF_B2 + 1
OFF_B3 = OFF_FC1 + NFC1; OFF_FC2 = OFF_B3 + 1; OFF_B4 = OFF_FC2 + NFC2; OFF_OUT = OFF_B4 + 1
N_NEURONS = OFF_OUT + NOUT
F = np.arange(OFF_F, OFF_F + N_F); FC1 = np.arange(OFF_FC1, OFF_FC1 + NFC1)
FC2 = np.arange(OFF_FC2, OFF_FC2 + NFC2); OUT = np.arange(OFF_OUT, OFF_OUT + NOUT)


def build_shared_conv(pre_grid, grid_h, grid_w, n_maps_in, n_maps_out, kernel, stride, out_offset, bias_idx):
    oh = (grid_h - kernel) // stride + 1; ow = (grid_w - kernel) // stride + 1
    n_out = n_maps_out * oh * ow; loc = np.arange(oh * ow); i = loc // ow; j = loc % ow
    pg = pre_grid.reshape(n_maps_in, grid_h, grid_w)
    patch = np.stack([pg[m, stride * i + di, stride * j + dj] for m in range(n_maps_in) for di in range(kernel) for dj in range(kernel)], axis=1)
    patch = np.tile(patch, (n_maps_out, 1))
    pre = np.column_stack([patch, np.full(n_out, bias_idx)])
    base = out_offset + np.repeat(np.arange(n_maps_out) * (oh * ow), oh * ow)
    post = np.repeat(base + np.tile(loc, n_maps_out), n_maps_in * kernel * kernel + 1)
    kpos = np.tile(np.arange(n_maps_in * kernel * kernel + 1), n_out)
    fmap = np.repeat(np.arange(n_maps_out), oh * ow * (n_maps_in * kernel * kernel + 1))
    kidx = fmap * (n_maps_in * kernel * kernel + 1) + kpos
    return pre.ravel().astype(np.int64), post.astype(np.int64), n_out, kidx.astype(np.int64)


in_grid = np.arange(784).reshape(28, 28)
PRE1, POST1, N1, KIDX1 = build_shared_conv(in_grid[None], 28, 28, 1, C1, K1, 2, OFF_F, OFF_B1)
N_S1 = C1 * (K1 * K1 + 1)
PRE2 = np.tile(np.concatenate([[OFF_B2], F]), NFC1).astype(np.int64); POST2 = np.repeat(FC1, N_F + 1)
PRE_FC2 = np.tile(np.concatenate([[OFF_B3], FC1]), NFC2).astype(np.int64); POST_FC2 = np.repeat(FC2, NFC1 + 1)
PRE3 = np.tile(np.concatenate([[OFF_B4], FC2]), NOUT).astype(np.int64); POST3 = np.repeat(OUT, NFC2 + 1)
G1 = N_S1; G2 = G1 + len(PRE2); G_FC2 = G2 + len(PRE_FC2); G3 = G_FC2 + len(PRE3)
SIGN = RNG.choice([-1.0, 1.0], G3); P = np.full(G3, 0.3)
P[G_FC2 + np.arange(NOUT) * (NFC2 + 1)] = 0.6

tr = tr.astype(np.float32) / 255.0; te = te.astype(np.float32) / 255.0
tr_flat = tr.reshape(60000, -1); te_flat = te.reshape(10000, -1)
order = RNG.permutation(60000)[:N_TRAIN]; y_onehot = np.eye(NOUT)
u = np.zeros(N_NEURONS); r_est = np.zeros(N_NEURONS); f_est = np.zeros(NOUT)
E1 = np.zeros(len(PRE1)); E2 = np.zeros(len(PRE2)); E_fc2 = np.zeros(len(PRE_FC2)); E3 = np.zeros(len(PRE3))


def spiking_step(x_vec, y_vec, learn):
    global u, r_est, f_est, P, E1, E2, E_fc2, E3
    if TAU_M > 0.0:
        u *= np.exp(-DT / TAU_M)
    pre_spikes = np.zeros(N_NEURONS)
    pre_spikes[OFF_IN:OFF_IN + 784] = RNG.poisson(R_IN * x_vec * DT)
    pre_spikes[OFF_B1] = RNG.poisson(BIAS_RATE * DT)
    pre_spikes[OFF_B2] = RNG.poisson(BIAS_RATE * DT)
    pre_spikes[OFF_B3] = RNG.poisson(BIAS_RATE * DT)
    pre_spikes[OFF_B4] = RNG.poisson(BIAS_RATE * DT)

    k1 = RNG.binomial(pre_spikes[PRE1].astype(np.int64), P[KIDX1])
    u[F] += np.bincount(POST1 - OFF_F, SIGN[KIDX1] * k1, minlength=N_F)
    E1 += -E1 * DT / TAU_E + pre_spikes[PRE1]
    n1 = np.floor(u[F] / THETA).clip(0.0, None).astype(np.int64)
    u[F] -= n1 * THETA; r_est[F] += -r_est[F] * DT / TAU_R + n1; pre_spikes[F] = n1

    k2 = RNG.binomial(pre_spikes[PRE2].astype(np.int64), P[G1:G2])
    u[FC1] += np.bincount(POST2 - OFF_FC1, SIGN[G1:G2] * k2, minlength=NFC1)
    E2 += -E2 * DT / TAU_E + pre_spikes[PRE2]
    n2 = np.floor(u[FC1] / THETA).clip(0.0, None).astype(np.int64)
    u[FC1] -= n2 * THETA; r_est[FC1] += -r_est[FC1] * DT / TAU_R + n2; pre_spikes[FC1] = n2

    k_fc2 = RNG.binomial(pre_spikes[PRE_FC2].astype(np.int64), P[G2:G_FC2])
    u[FC2] += np.bincount(POST_FC2 - OFF_FC2, SIGN[G2:G_FC2] * k_fc2, minlength=NFC2)
    E_fc2 += -E_fc2 * DT / TAU_E + pre_spikes[PRE_FC2]
    n_fc2 = np.floor(u[FC2] / THETA).clip(0.0, None).astype(np.int64)
    u[FC2] -= n_fc2 * THETA; r_est[FC2] += -r_est[FC2] * DT / TAU_R + n_fc2; pre_spikes[FC2] = n_fc2

    k3 = RNG.binomial(pre_spikes[PRE3].astype(np.int64), P[G_FC2:G3])
    if KAPPA > 0.0:
        out_rate = f_est / TAU_F
        u[OUT] -= KAPPA * np.sum(out_rate) * DT
    u[OUT] += np.bincount(POST3 - OFF_OUT, SIGN[G_FC2:G3] * k3, minlength=NOUT)
    E3 += -E3 * DT / TAU_E + pre_spikes[PRE3]
    n_out = np.floor(u[OUT] / THETA).clip(0.0, None).astype(np.int64)
    u[OUT] -= n_out * THETA; r_est[OUT] += -r_est[OUT] * DT / TAU_R + n_out
    f_est += -f_est * DT / TAU_F + n_out

    if learn:
        w_fc1 = SIGN[G1:G2] * P[G1:G2] / THETA
        w_fc2 = SIGN[G2:G_FC2] * P[G2:G_FC2] / THETA
        w_out = SIGN[G_FC2:G3] * P[G_FC2:G3] / THETA
        gateF = (r_est[F] / TAU_R) > R_GATE
        gateFC1 = (r_est[FC1] / TAU_R) > R_GATE
        gateFC2 = (r_est[FC2] / TAU_R) > R_GATE
        d_out = f_est / TAU_F - TARGET * y_vec
        d_fc2 = gateFC2 * np.bincount(PRE3 - OFF_FC2, w_out * d_out[POST3 - OFF_OUT], minlength=NFC2 + 1)[:NFC2]
        d_fc1 = gateFC1 * np.bincount(PRE_FC2 - OFF_FC1, w_fc2 * d_fc2[POST_FC2 - OFF_FC2], minlength=NFC1 + 1)[:NFC1]
        d_f = np.bincount(PRE2 - OFF_F, w_fc1 * d_fc1[POST2 - OFF_FC1], minlength=N_F + 1)[:N_F]
        d1 = gateF[POST1 - OFF_F] * d_f[POST1 - OFF_F]
        d2 = d_fc1[POST2 - OFF_FC1]; d_fc2_edge = d_fc2[POST_FC2 - OFF_FC2]; d3 = d_out[POST3 - OFF_OUT]
        kk1 = RNG.poisson(np.clip(FDA - d1, 0.0, None) * DT)
        kk2 = RNG.poisson(np.clip(FDA - d2, 0.0, None) * DT)
        kk_fc2 = RNG.poisson(np.clip(FDA - d_fc2_edge, 0.0, None) * DT)
        kk3 = RNG.poisson(np.clip(FDA - d3, 0.0, None) * DT)
        sum1 = np.bincount(KIDX1, E1 * (kk1 - C * DT), minlength=N_S1)
        P[:G1] += SIGN[:G1] * ALPHA * sum1
        P[G1:G2] += SIGN[G1:G2] * ALPHA * E2 * (kk2 - C * DT)
        P[G2:G_FC2] += SIGN[G2:G_FC2] * ALPHA * E_fc2 * (kk_fc2 - C * DT)
        P[G_FC2:G3] += SIGN[G_FC2:G3] * ALPHA * E3 * (kk3 - C * DT)
        P = np.clip(P, 1e-6, 1.0 - 1e-6)


def reset_state():
    global u, r_est, f_est
    u[:] = 0; r_est[:] = 0; f_est[:] = 0; E1[:] = 0; E2[:] = 0; E_fc2[:] = 0; E3[:] = 0


def evaluate(n_img):
    idx = RNG.choice(10000, n_img, replace=False)
    hits = 0
    for ii in idx:
        if RESET_PER_SAMPLE:
            reset_state()
        x = te_flat[ii]; y = tel[ii]
        for _ in range(steps):
            spiking_step(x, np.zeros(NOUT), learn=False)
        if ISI_STEPS > 0:
            zx = np.zeros(784)
            for _ in range(ISI_STEPS):
                spiking_step(zx, np.zeros(NOUT), learn=False)
        hits += int(np.argmax(f_est) == y)
    return hits / n_img


def run_training():
    print(f"deep topology: 784 -> CV5x5x{C1} str2 ({N_S1} shared) -> FC{NFC1} -> FC{NFC2} -> {NOUT}  "
          f"({N_NEURONS} neurons, {G3} params)", flush=True)
    t0 = time.time()
    for smp in range(N_TRAIN):
        ii = order[smp]; x = tr_flat[ii]; y = trl[ii]; yv = y_onehot[y]
        if RESET_PER_SAMPLE:
            reset_state()
        for _ in range(steps):
            spiking_step(x, yv, learn=True)
        if ISI_STEPS > 0:
            zx = np.zeros(784)
            for _ in range(ISI_STEPS):
                spiking_step(zx, np.zeros(NOUT), learn=False)
        if (smp + 1) % 250 == 0:
            f = f_est / TAU_F
            loss = float(np.mean(0.5 * ((f - TARGET * yv) / TARGET) ** 2)) if TARGET > 0 else float("nan")
            te_acc = evaluate(200)
            print(f"  train {smp + 1}/{N_TRAIN}  loss {loss:.4f}  test_acc(n=200) {te_acc:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    np.savez("mnist_deep_checkpoint.npz", P=P, SIGN=SIGN)
    print(f"  final test acc (n={N_FINAL_EVAL}) = {evaluate(N_FINAL_EVAL):.3f}")


if __name__ == "__main__":
    run_training()
