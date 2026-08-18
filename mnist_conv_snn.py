import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mnist_loader import load_mnist

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
SAMPLE_T = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
R_IN = float(sys.argv[4]) if len(sys.argv) > 4 else 200.0
ALPHA = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-6
SIZE = sys.argv[6] if len(sys.argv) > 6 else "small"   # full=6/12/32, small=4/8/24, tiny=2/6/16

RNG = np.random.default_rng(SEED)

# ---------------- hyperparameters ----------------
THETA = 1.0          # conv/FC 层 IF 阈值
THETA_POOL = 4.0     # 平均池化阈值（2x2 → 出率 = Σf/θ = 平均）
TAU_E = 0.2
TAU_R = 0.1
TAU_F = 0.1
DT = 0.02
BIAS_RATE = 30.0
TARGET = R_IN
FDA = 2000.0
C = FDA
R_GATE = 1.0
P_INIT = 0.3
N_EVAL = 200          # 周期评估的测试图像数
EVAL_EVERY = 500
N_FINAL_EVAL = 1000   # 结束时的测试图像数

# ---------------- topology ----------------
if SIZE == "full":
    C1, K1 = 6, 5         # conv1: 6 maps, 5x5  → 24x24
    C2, K2 = 12, 5        # conv2: 12 maps, 5x5 → 8x8
    NFC = 32
elif SIZE == "tiny":
    C1, K1 = 2, 5
    C2, K2 = 6, 5
    NFC = 16
else:                     # small (默认)
    C1, K1 = 4, 5
    C2, K2 = 8, 5
    NFC = 24

N_L1 = C1 * 24 * 24
N_P1 = C1 * 12 * 12
N_L2 = C2 * 8 * 8
N_P2 = C2 * 4 * 4
NOUT = 10

OFF_IN = 0                       # 784
OFF_B1 = 784                     # 1
OFF_L1 = 785                     # N_L1
OFF_P1 = OFF_L1 + N_L1           # N_P1
OFF_B2 = OFF_P1 + N_P1           # 1
OFF_L2 = OFF_B2 + 1              # N_L2
OFF_P2 = OFF_L2 + N_L2           # N_P2
OFF_B3 = OFF_P2 + N_P2           # 1
OFF_FC = OFF_B3 + 1              # NFC
OFF_B4 = OFF_FC + NFC            # 1
OFF_OUT = OFF_B4 + 1             # 10
N_NEURONS = OFF_OUT + NOUT

L1 = np.arange(OFF_L1, OFF_L1 + N_L1)
P1 = np.arange(OFF_P1, OFF_P1 + N_P1)
L2 = np.arange(OFF_L2, OFF_L2 + N_L2)
P2 = np.arange(OFF_P2, OFF_P2 + N_P2)
FC = np.arange(OFF_FC, OFF_FC + NFC)
OUT = np.arange(OFF_OUT, OFF_OUT + NOUT)


def build_conv(pre_grid, grid_h, grid_w, n_maps_in, n_maps_out, kernel, out_offset, bias_idx):
    """局部连接 conv：每个输出神经元接收全部输入图上的 kernel×kernel 感受野 + 1 bias。无权重共享。"""
    oh, ow = grid_h - kernel + 1, grid_w - kernel + 1
    n_out = n_maps_out * oh * ow
    loc = np.arange(oh * ow)
    i = loc // ow
    j = loc % ow
    pg = pre_grid.reshape(n_maps_in, grid_h, grid_w)
    patch = np.stack([pg[m, i + di, j + dj] for m in range(n_maps_in) for di in range(kernel) for dj in range(kernel)], axis=1)
    patch = np.tile(patch, (n_maps_out, 1))
    pre = np.column_stack([patch, np.full(n_out, bias_idx)])
    base = out_offset + np.repeat(np.arange(n_maps_out) * (oh * ow), oh * ow)
    post = np.repeat(base + np.tile(loc, n_maps_out), n_maps_in * kernel * kernel + 1)
    return pre.ravel().astype(np.int64), post.astype(np.int64), n_out


def build_pool(pre_offset, grid_h, grid_w, n_maps, out_offset):
    """平均池化（固定边，权重 1/θ）：2x2 块 → 1 个池化神经元。"""
    oh, ow = grid_h // 2, grid_w // 2
    n_out = n_maps * oh * ow
    loc = np.arange(oh * ow)
    i = loc // ow
    j = loc % ow
    pg = (pre_offset + np.arange(n_maps * grid_h * grid_w).reshape(n_maps, grid_h, grid_w))
    pre = np.stack([pg[:, 2 * i + di, 2 * j + dj] for di in range(2) for dj in range(2)], axis=1).reshape(-1)
    base = out_offset + np.repeat(np.arange(n_maps) * (oh * ow), oh * ow)
    post = np.repeat(base + np.tile(loc, n_maps), 4)
    return pre.astype(np.int64), post.astype(np.int64), n_out


in_grid = np.arange(784).reshape(28, 28)
PRE1, POST1, N1 = build_conv(in_grid[None], 28, 28, 1, C1, K1, OFF_L1, OFF_B1)
PREP1, POSTP1, NP1 = build_pool(OFF_L1, 24, 24, C1, OFF_P1)
p1_grid = np.arange(N_P1).reshape(C1, 12, 12) + OFF_P1
PRE2, POST2, N2 = build_conv(p1_grid, 12, 12, C1, C2, K2, OFF_L2, OFF_B2)
PREP2, POSTP2, NP2 = build_pool(OFF_L2, 8, 8, C2, OFF_P2)

PRE3 = np.tile(np.concatenate([[OFF_B3], np.arange(N_P2) + OFF_P2]), NFC).astype(np.int64)
POST3 = np.repeat(FC, N_P2 + 1)
PRE4 = np.tile(np.concatenate([[OFF_B4], FC]), NOUT).astype(np.int64)
POST4 = np.repeat(OUT, NFC + 1)

G1 = len(PRE1)
G2 = G1 + len(PRE2)
G3 = G2 + len(PRE3)
G4 = G3 + len(PRE4)
GROUPS = [("W1", 0, G1), ("W2", G1, G2), ("W3", G2, G3), ("W4", G3, G4)]
SIGN = RNG.choice([-1.0, 1.0], G4)

P = np.full(G4, P_INIT)
P[G3 + np.arange(NOUT) * (NFC + 1)] = 0.6   # 输出层 bias 突触

# ---------------- data ----------------
print(f"loading MNIST ...", flush=True)
tr, trl = load_mnist(train=True)
te, tel = load_mnist(train=False)
tr = tr.astype(np.float32) / 255.0
te = te.astype(np.float32) / 255.0
tr_flat = tr.reshape(60000, -1)
te_flat = te.reshape(10000, -1)

steps = int(SAMPLE_T / DT)
order = RNG.permutation(60000)[:N_TRAIN]

u = np.zeros(N_NEURONS)
r_est = np.zeros(N_NEURONS)
f_est = np.zeros(NOUT)
E1 = np.zeros(G1)
E2 = np.zeros(G2 - G1)
E3 = np.zeros(G3 - G2)
E4 = np.zeros(G4 - G3)
y_onehot = np.eye(NOUT)


def spiking_step(x_vec, y_vec, learn):
    global u, r_est, f_est, E1, E2, E3, E4, P
    pre_spikes = np.zeros(N_NEURONS)
    pre_spikes[OFF_IN:OFF_IN + 784] = RNG.poisson(R_IN * x_vec * DT)
    pre_spikes[OFF_B1] = RNG.poisson(BIAS_RATE * DT)
    pre_spikes[OFF_B2] = RNG.poisson(BIAS_RATE * DT)
    pre_spikes[OFF_B3] = RNG.poisson(BIAS_RATE * DT)
    pre_spikes[OFF_B4] = RNG.poisson(BIAS_RATE * DT)

    k1 = RNG.binomial(pre_spikes[PRE1].astype(np.int64), P[:G1])
    u[L1] += np.bincount(POST1 - OFF_L1, SIGN[:G1] * k1, minlength=N1)
    E1 += -E1 * DT / TAU_E + pre_spikes[PRE1]
    n1 = np.floor(u[L1] / THETA).clip(0.0, None).astype(np.int64)
    u[L1] -= n1 * THETA
    r_est[L1] += -r_est[L1] * DT / TAU_R + n1
    pre_spikes[L1] = n1

    u[P1] += np.bincount(PREP1 - OFF_L1, pre_spikes[PREP1].astype(np.float64), minlength=N1)[:NP1]
    nP1 = np.floor(u[P1] / THETA_POOL).clip(0.0, None).astype(np.int64)
    u[P1] -= nP1 * THETA_POOL
    r_est[P1] += -r_est[P1] * DT / TAU_R + nP1
    pre_spikes[P1] = nP1

    k2 = RNG.binomial(pre_spikes[PRE2].astype(np.int64), P[G1:G2])
    u[L2] += np.bincount(POST2 - OFF_L2, SIGN[G1:G2] * k2, minlength=N2)
    E2 += -E2 * DT / TAU_E + pre_spikes[PRE2]
    n2 = np.floor(u[L2] / THETA).clip(0.0, None).astype(np.int64)
    u[L2] -= n2 * THETA
    r_est[L2] += -r_est[L2] * DT / TAU_R + n2
    pre_spikes[L2] = n2

    u[P2] += np.bincount(PREP2 - OFF_L2, pre_spikes[PREP2].astype(np.float64), minlength=N2)[:NP2]
    nP2 = np.floor(u[P2] / THETA_POOL).clip(0.0, None).astype(np.int64)
    u[P2] -= nP2 * THETA_POOL
    r_est[P2] += -r_est[P2] * DT / TAU_R + nP2
    pre_spikes[P2] = nP2

    k3 = RNG.binomial(pre_spikes[PRE3].astype(np.int64), P[G2:G3])
    u[FC] += np.bincount(POST3 - OFF_FC, SIGN[G2:G3] * k3, minlength=NFC)
    E3 += -E3 * DT / TAU_E + pre_spikes[PRE3]
    n3 = np.floor(u[FC] / THETA).clip(0.0, None).astype(np.int64)
    u[FC] -= n3 * THETA
    r_est[FC] += -r_est[FC] * DT / TAU_R + n3
    pre_spikes[FC] = n3

    k4 = RNG.binomial(pre_spikes[PRE4].astype(np.int64), P[G3:G4])
    u[OUT] += np.bincount(POST4 - OFF_OUT, SIGN[G3:G4] * k4, minlength=NOUT)
    E4 += -E4 * DT / TAU_E + pre_spikes[PRE4]
    n_out = np.floor(u[OUT] / THETA).clip(0.0, None).astype(np.int64)
    u[OUT] -= n_out * THETA
    r_est[OUT] += -r_est[OUT] * DT / TAU_R + n_out
    f_est += -f_est * DT / TAU_F + n_out

    if learn:
        w1 = SIGN[:G1] * P[:G1] / THETA
        w2 = SIGN[G1:G2] * P[G1:G2] / THETA
        w3 = SIGN[G2:G3] * P[G2:G3] / THETA
        w4 = SIGN[G3:G4] * P[G3:G4] / THETA

        gateL1 = (r_est[L1] / TAU_R) > R_GATE
        gateL2 = (r_est[L2] / TAU_R) > R_GATE
        gateFC = (r_est[FC] / TAU_R) > R_GATE

        d_out = f_est / TAU_F - TARGET * y_vec
        d_fc = gateFC * np.bincount(PRE4 - OFF_FC, w4 * d_out[POST4 - OFF_OUT], minlength=NFC + 1)[:NFC]
        d_p2 = np.bincount(PRE3 - OFF_P2, w3 * d_fc[POST3 - OFF_FC], minlength=N_P2 + 1)[:NP2]
        d_l2 = gateL2 * np.bincount(PREP2 - OFF_L2, d_p2[POSTP2 - OFF_P2] / THETA_POOL, minlength=N2)
        d_p1 = np.bincount(PRE2 - OFF_P1, w2 * d_l2[POST2 - OFF_L2], minlength=N_P1 + 1)[:NP1]
        d_l1 = gateL1 * np.bincount(PREP1 - OFF_L1, d_p1[POSTP1 - OFF_P1] / THETA_POOL, minlength=N1)

        d1 = d_l1[POST1 - OFF_L1]
        d2 = d_l2[POST2 - OFF_L2]
        d3 = d_fc[POST3 - OFF_FC]
        d4 = d_out[POST4 - OFF_OUT]

        kk1 = RNG.poisson(np.clip(FDA - d1, 0.0, None) * DT)
        kk2 = RNG.poisson(np.clip(FDA - d2, 0.0, None) * DT)
        kk3 = RNG.poisson(np.clip(FDA - d3, 0.0, None) * DT)
        kk4 = RNG.poisson(np.clip(FDA - d4, 0.0, None) * DT)
        P[:G1] += SIGN[:G1] * ALPHA * E1 * (kk1 - C * DT)
        P[G1:G2] += SIGN[G1:G2] * ALPHA * E2 * (kk2 - C * DT)
        P[G2:G3] += SIGN[G2:G3] * ALPHA * E3 * (kk3 - C * DT)
        P[G3:G4] += SIGN[G3:G4] * ALPHA * E4 * (kk4 - C * DT)
        P = np.clip(P, 1e-6, 1.0 - 1e-6)


def evaluate(n_img):
    """冻结学习，连续呈现 n_img 张测试图，argmax(f_est) 判类。"""
    idx = RNG.choice(10000, n_img, replace=False)
    hits = 0
    for ii in idx:
        x = te_flat[ii]
        y = tel[ii]
        for _ in range(steps):
            spiking_step(x, y, learn=False)
        hits += int(np.argmax(f_est) == y)
    return hits / n_img


def run_training():
    global u, r_est, f_est, E1, E2, E3, E4, P
    print(f"topology: 784 -> conv{5}x{5}x{C1} -> avgpool2 -> conv{5}x{5}x{C2} -> avgpool2 -> FC{NFC} -> {NOUT}"
          f"  ({N_NEURONS} neurons, {G4} plastic synapses)  seed={SEED}", flush=True)
    print(f"R={R_IN}  f_da=C={FDA}  TARGET={TARGET}  alpha={ALPHA}  sample={SAMPLE_T}s  train={N_TRAIN}", flush=True)

    losses = np.zeros(N_TRAIN)
    accs = np.zeros(N_TRAIN)
    eval_pts = []
    t0 = time.time()
    for smp in range(N_TRAIN):
        ii = order[smp]
        x = tr_flat[ii]
        y = trl[ii]
        yv = y_onehot[y]
        for _ in range(steps):
            spiking_step(x, yv, learn=True)
        f = f_est / TAU_F
        losses[smp] = float(np.mean(0.5 * ((f - TARGET * yv) / TARGET) ** 2))
        accs[smp] = float(np.argmax(f) == y)
        if (smp + 1) % 250 == 0:
            print(f"  train {smp + 1}/{N_TRAIN}  loss(roll100) {np.mean(losses[smp - 99:smp + 1]):.4f}"
                  f"  acc(roll100) {np.mean(accs[smp - 99:smp + 1]):.3f}  ({time.time() - t0:.0f} s)", flush=True)
        if (smp + 1) % EVAL_EVERY == 0:
            a = evaluate(N_EVAL)
            eval_pts.append((smp + 1, a))
            print(f"  >>> test acc @ {smp + 1}: {a:.3f}", flush=True)

    train_time = time.time() - t0
    a_final = evaluate(N_FINAL_EVAL)
    print(f"training done in {train_time:.0f} s; final test acc (n={N_FINAL_EVAL}) = {a_final:.4f}", flush=True)

    np.savez("mnist_checkpoint.npz", P=P, SIGN=SIGN,
             PRE1=PRE1, POST1=POST1, PREP1=PREP1, POSTP1=POSTP1,
             PRE2=PRE2, POST2=POST2, PREP2=PREP2, POSTP2=POSTP2,
             PRE3=PRE3, POST3=POST3, PRE4=PRE4, POST4=POST4,
             R_IN=R_IN, ALPHA=ALPHA, SEED=SEED)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    roll = np.convolve(losses, np.ones(100) / 100, mode="valid")
    axes[0].plot(np.arange(len(roll)) + 99, roll, lw=1.2)
    axes[0].set_title("train loss (rolling 100)")
    axes[0].set_xlabel("sample")
    rolla = np.convolve(accs, np.ones(100) / 100, mode="valid")
    axes[1].plot(np.arange(len(rolla)) + 99, rolla, lw=1.2, color="tab:green")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("train acc (rolling 100)")
    axes[1].set_xlabel("sample")
    axes[2].plot([p[0] for p in eval_pts], [p[1] for p in eval_pts], "o-", color="tab:red")
    axes[2].axhline(0.1, color="gray", ls="--", lw=0.8)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title(f"test acc (final {a_final:.3f})")
    axes[2].set_xlabel("train sample")
    plt.tight_layout()
    plt.savefig("mnist_result.png", dpi=150)
    print("figure saved to mnist_result.png", flush=True)


if __name__ == "__main__":
    run_training()
