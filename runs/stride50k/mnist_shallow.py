import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mnist_loader import load_mnist

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
SAMPLE_T = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
R_IN = float(sys.argv[4]) if len(sys.argv) > 4 else 400.0
ALPHA = float(sys.argv[5]) if len(sys.argv) > 5 else 2e-6
FDA_IN = float(sys.argv[6]) if len(sys.argv) > 6 else 1000.0
BIAS_IN = float(sys.argv[7]) if len(sys.argv) > 7 else 30.0
MODE = sys.argv[8] if len(sys.argv) > 8 else "stride"
TARGET_IN = float(sys.argv[9]) if len(sys.argv) > 9 else 0.0   # 0 时 TARGET=R_IN
DT = float(sys.argv[11]) if len(sys.argv) > 11 else 0.02       # 每步时长（主频提高时按比例缩小）
TAU_SCALE = float(sys.argv[12]) if len(sys.argv) > 12 else 1.0  # 时间常数缩放：τ_eff = τ / TAU_SCALE（主频 = 1/τ）
LOSS = sys.argv[13] if len(sys.argv) > 13 else "mse"    # mse | ce（交叉熵: δ = G·(softmax(f̂/τ_sm)−y)，零和梯度，消除 9:1 稀释）
TAU_SM = float(sys.argv[14]) if len(sys.argv) > 14 else 100.0   # softmax 温度（Hz）：速率差除以该标度后做 softmax
# MODE: stride = 步长2卷积降采样(无池化, 反向无 1/θ 衰减)
#       avg    = 平均池化(θ=4, 反向 δ/4)
#       max    = 侧抑制最大池化(WTA, 反向只给胜者, 无 1/θ 衰减)

RNG = np.random.default_rng(SEED)

THETA = 1.0
THETA_POOL = 4.0        # 平均池化阈值（avg 模式）
THETA_MAX = 1.0         # 最大池化阈值（max 模式）
TAU_E = 0.2 / TAU_SCALE
TAU_R = 0.1 / TAU_SCALE
TAU_F = 0.1 / TAU_SCALE
BIAS_RATE = BIAS_IN
TARGET = TARGET_IN if TARGET_IN > 0 else R_IN
FDA = FDA_IN
C = FDA
R_GATE = 1.0
P_INIT = 0.3
N_EVAL = 200
EVAL_EVERY = 500
N_FINAL_EVAL = int(sys.argv[10]) if len(sys.argv) > 10 else 1000

C1, K1 = 4, 5
NFC = 32
NOUT = 10

if MODE == "stride":
    STRIDE = 2
    HAS_POOL = False
    N_L1 = 0
    N_F = C1 * 12 * 12          # 12x12 conv 输出直接进 FC
else:
    STRIDE = 1
    HAS_POOL = True
    N_L1 = C1 * 24 * 24
    N_F = C1 * 12 * 12          # 池化输出 12x12

OFF_IN = 0
OFF_B1 = 784
OFF_L1 = 785
OFF_F = OFF_L1 + N_L1           # 特征块（池化输出 或 stride 卷积输出）
OFF_B2 = OFF_F + N_F
OFF_FC = OFF_B2 + 1
OFF_B3 = OFF_FC + NFC
OFF_OUT = OFF_B3 + 1
N_NEURONS = OFF_OUT + NOUT

L1 = np.arange(OFF_L1, OFF_L1 + N_L1)
F = np.arange(OFF_F, OFF_F + N_F)
FC = np.arange(OFF_FC, OFF_FC + NFC)
OUT = np.arange(OFF_OUT, OFF_OUT + NOUT)


def build_conv(pre_grid, grid_h, grid_w, n_maps_in, n_maps_out, kernel, stride, out_offset, bias_idx):
    oh = (grid_h - kernel) // stride + 1
    ow = (grid_w - kernel) // stride + 1
    n_out = n_maps_out * oh * ow
    loc = np.arange(oh * ow)
    i = loc // ow
    j = loc % ow
    pg = pre_grid.reshape(n_maps_in, grid_h, grid_w)
    patch = np.stack([pg[m, stride * i + di, stride * j + dj] for m in range(n_maps_in) for di in range(kernel) for dj in range(kernel)], axis=1)
    patch = np.tile(patch, (n_maps_out, 1))
    pre = np.column_stack([patch, np.full(n_out, bias_idx)])
    base = out_offset + np.repeat(np.arange(n_maps_out) * (oh * ow), oh * ow)
    post = np.repeat(base + np.tile(loc, n_maps_out), n_maps_in * kernel * kernel + 1)
    return pre.ravel().astype(np.int64), post.astype(np.int64), n_out


def build_pool(pre_offset, grid_h, grid_w, n_maps, out_offset):
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
if MODE == "stride":
    PRE1, POST1, N1 = build_conv(in_grid[None], 28, 28, 1, C1, K1, STRIDE, OFF_F, OFF_B1)
    PREP1, POSTP1, NP1 = None, None, None
    PRE2 = np.tile(np.concatenate([[OFF_B2], F]), NFC).astype(np.int64)
    POST2 = np.repeat(FC, N_F + 1)
else:
    PRE1, POST1, N1 = build_conv(in_grid[None], 28, 28, 1, C1, K1, 1, OFF_L1, OFF_B1)
    PREP1, POSTP1, NP1 = build_pool(OFF_L1, 24, 24, C1, OFF_F)
    PRE2 = np.tile(np.concatenate([[OFF_B2], F]), NFC).astype(np.int64)
    POST2 = np.repeat(FC, N_F + 1)
PRE3 = np.tile(np.concatenate([[OFF_B3], FC]), NOUT).astype(np.int64)
POST3 = np.repeat(OUT, NFC + 1)

G1 = len(PRE1)
G2 = G1 + len(PRE2)
G3 = G2 + len(PRE3)
SIGN = RNG.choice([-1.0, 1.0], G3)
P = np.full(G3, P_INIT)
P[G2 + np.arange(NOUT) * (NFC + 1)] = 0.6

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
y_onehot = np.eye(NOUT)
PREP1_4 = PREP1.reshape(-1, 4) if HAS_POOL else None


def spiking_step(x_vec, y_vec, learn):
    global u, r_est, f_est, E1, E2, E3, P
    pre_spikes = np.zeros(N_NEURONS)
    pre_spikes[OFF_IN:OFF_IN + 784] = RNG.poisson(R_IN * x_vec * DT)
    pre_spikes[OFF_B1] = RNG.poisson(BIAS_RATE * DT)
    pre_spikes[OFF_B2] = RNG.poisson(BIAS_RATE * DT)
    pre_spikes[OFF_B3] = RNG.poisson(BIAS_RATE * DT)

    k1 = RNG.binomial(pre_spikes[PRE1].astype(np.int64), P[:G1])
    if MODE == "stride":
        u[F] += np.bincount(POST1 - OFF_F, SIGN[:G1] * k1, minlength=N_F)
    else:
        u[L1] += np.bincount(POST1 - OFF_L1, SIGN[:G1] * k1, minlength=N1)
    E1 += -E1 * DT / TAU_E + pre_spikes[PRE1]
    n1 = np.floor(u[L1 if MODE != "stride" else F] / THETA).clip(0.0, None).astype(np.int64)
    u[L1 if MODE != "stride" else F] -= n1 * THETA
    r_est[L1 if MODE != "stride" else F] += -r_est[L1 if MODE != "stride" else F] * DT / TAU_R + n1
    pre_spikes[L1 if MODE != "stride" else F] = n1

    if HAS_POOL:
        if MODE == "avg":
            u[F] += np.bincount(PREP1 - OFF_L1, n1[PREP1 - OFF_L1].astype(np.float64), minlength=N1)[:NP1]
            nF = np.floor(u[F] / THETA_POOL).clip(0.0, None).astype(np.int64)
            u[F] -= nF * THETA_POOL
        else:  # max: 每块取本步最大发放数（侧抑制 WTA）
            bmax = n1[PREP1_4].max(axis=1)
            u[F] += bmax
            nF = np.floor(u[F] / THETA_MAX).clip(0.0, None).astype(np.int64)
            u[F] -= nF * THETA_MAX
        r_est[F] += -r_est[F] * DT / TAU_R + nF
        pre_spikes[F] = nF

    k2 = RNG.binomial(pre_spikes[PRE2].astype(np.int64), P[G1:G2])
    u[FC] += np.bincount(POST2 - OFF_FC, SIGN[G1:G2] * k2, minlength=NFC)
    E2 += -E2 * DT / TAU_E + pre_spikes[PRE2]
    n2 = np.floor(u[FC] / THETA).clip(0.0, None).astype(np.int64)
    u[FC] -= n2 * THETA
    r_est[FC] += -r_est[FC] * DT / TAU_R + n2
    pre_spikes[FC] = n2

    k3 = RNG.binomial(pre_spikes[PRE3].astype(np.int64), P[G2:G3])
    u[OUT] += np.bincount(POST3 - OFF_OUT, SIGN[G2:G3] * k3, minlength=NOUT)
    E3 += -E3 * DT / TAU_E + pre_spikes[PRE3]
    n_out = np.floor(u[OUT] / THETA).clip(0.0, None).astype(np.int64)
    u[OUT] -= n_out * THETA
    r_est[OUT] += -r_est[OUT] * DT / TAU_R + n_out
    f_est += -f_est * DT / TAU_F + n_out

    if learn:
        w1 = SIGN[:G1] * P[:G1] / THETA
        w2 = SIGN[G1:G2] * P[G1:G2] / THETA
        w3 = SIGN[G2:G3] * P[G2:G3] / THETA

        gateF = (r_est[F] / TAU_R) > R_GATE
        gateFC = (r_est[FC] / TAU_R) > R_GATE
        gateL1 = None if MODE == "stride" else ((r_est[L1] / TAU_R) > R_GATE)

        d_out = f_est / TAU_F - TARGET * y_vec
        if LOSS == "ce":
            fh = f_est / TAU_F / TAU_SM
            mx = fh.max()
            p = np.exp(fh - mx)
            p = p / p.sum()
            d_out = TARGET * (p - y_vec)   # 零和梯度：Σδ=0，消除 9:1 错误输出稀释
        d_fc = gateFC * np.bincount(PRE3 - OFF_FC, w3 * d_out[POST3 - OFF_OUT], minlength=NFC + 1)[:NFC]
        d_f = np.bincount(PRE2 - OFF_F, w2 * d_fc[POST2 - OFF_FC], minlength=N_F + 1)[:N_F]

        if MODE == "stride":
            d1 = gateF[POST1 - OFF_F] * d_f[POST1 - OFF_F]
        elif MODE == "avg":
            d_l1 = np.zeros(N1)
            d_l1[PREP1 - OFF_L1] = d_f[POSTP1 - OFF_F] / THETA_POOL
            d1 = gateL1[POST1 - OFF_L1] * d_l1[POST1 - OFF_L1]
        else:  # max: 只给每块胜者
            d_l1 = np.zeros(N1)
            win = np.argmax(r_est[PREP1_4], axis=1)
            flat = PREP1_4[np.arange(NP1), win]
            d_l1[flat - OFF_L1] = d_f
            d1 = gateL1[POST1 - OFF_L1] * d_l1[POST1 - OFF_L1]

        d2 = d_fc[POST2 - OFF_FC]
        d3 = d_out[POST3 - OFF_OUT]

        kk1 = RNG.poisson(np.clip(FDA - d1, 0.0, None) * DT)
        kk2 = RNG.poisson(np.clip(FDA - d2, 0.0, None) * DT)
        kk3 = RNG.poisson(np.clip(FDA - d3, 0.0, None) * DT)
        P[:G1] += SIGN[:G1] * ALPHA * E1 * (kk1 - C * DT)
        P[G1:G2] += SIGN[G1:G2] * ALPHA * E2 * (kk2 - C * DT)
        P[G2:G3] += SIGN[G2:G3] * ALPHA * E3 * (kk3 - C * DT)
        P = np.clip(P, 1e-6, 1.0 - 1e-6)


def evaluate(n_img):
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
    global u, r_est, f_est, E1, E2, E3, P
    print(f"topology: 784 -> conv5x5x{C1}(stride{STRIDE}{'' if HAS_POOL else ' 无池化'}"
          f"{' -> pool' + MODE if HAS_POOL else ''}) -> FC{NFC} -> {NOUT}"
          f"  ({N_NEURONS} neurons, {G3} plastic synapses)  seed={SEED}", flush=True)
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
        if LOSS == "ce":
            fh = f / TAU_SM
            mx = fh.max()
            p = np.exp(fh - mx)
            p = p / p.sum()
            ce = -np.log(p[y] + 1e-12)
            losses[smp] = ce   # ce 模式下报告交叉熵损失
        else:
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

    np.savez("mnist_checkpoint.npz", P=P, SIGN=SIGN, R_IN=R_IN, ALPHA=ALPHA, SEED=SEED, MODE=MODE,
             PRE1=PRE1, POST1=POST1, PREP1=PREP1, POSTP1=POSTP1,
             PRE2=PRE2, POST2=POST2, PRE3=PRE3, POST3=POST3)

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
