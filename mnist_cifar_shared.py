import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cifar10_loader import load_mnist

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
SAMPLE_T = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
R_IN = float(sys.argv[4]) if len(sys.argv) > 4 else 200.0
ALPHA = float(sys.argv[5]) if len(sys.argv) > 5 else 3e-9
FDA_IN = float(sys.argv[6]) if len(sys.argv) > 6 else 3000.0
BIAS_IN = float(sys.argv[7]) if len(sys.argv) > 7 else 30.0
TARGET_IN = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0
N_FINAL_EVAL = int(sys.argv[9]) if len(sys.argv) > 9 else 1000
DT = float(sys.argv[10]) if len(sys.argv) > 10 else 0.02
KAPPA = float(sys.argv[11]) if len(sys.argv) > 11 else 0.0   # 输出层抑制池（全局归一化）：u -= KAPPA*Σf̂*dt
GAMMA = float(sys.argv[12]) if len(sys.argv) > 12 else 0.0   # 输出层侧抑制：u[k] -= GAMMA*(Σf̂-f̂_k)*dt
RESET_PER_SAMPLE = int(sys.argv[13]) if len(sys.argv) > 13 else 0  # 每样本开始前重置 u/r_est/f_est/E（硬清零，对照用；拟真协议用 TAU_M+ISI）
ISI_STEPS = int(sys.argv[14]) if len(sys.argv) > 14 else 0          # 样本间静默步数：无输入无目标，频率/膜电位自然衰减后再进下一样本
TAU_M = float(sys.argv[15]) if len(sys.argv) > 15 else 0.0          # 膜泄漏时间常数（s，论文 §4.5 LIF；0=IF 无泄漏）。每步 u*=exp(-DT/TAU_M)
TAU_E = float(sys.argv[16]) if len(sys.argv) > 16 else 0.2          # 资格迹时间常数（论文 §8.5：应 ≫ 前向-反向延迟，τ_m 大时需同步增大）

RNG = np.random.default_rng(SEED)

THETA = 1.0
TAU_E = TAU_E
TAU_R = 0.1
TAU_F = 0.1
BIAS_RATE = BIAS_IN
TARGET = TARGET_IN if TARGET_IN > 0 else R_IN
FDA = FDA_IN
C = FDA
R_GATE = 1.0
P_INIT = 0.3
N_EVAL = 200
EVAL_EVERY = 500

C1, K1 = 4, 5            # conv1: 4 filters, 5x5, stride 2 -> 12x12
NFC = 32
NOUT = 10

N_F = C1 * 12 * 12

OFF_IN = 0
OFF_B1 = 784
OFF_F = 785
OFF_B2 = OFF_F + N_F
OFF_FC = OFF_B2 + 1
OFF_B3 = OFF_FC + NFC
OFF_OUT = OFF_B3 + 1
N_NEURONS = OFF_OUT + NOUT

F = np.arange(OFF_F, OFF_F + N_F)
FC = np.arange(OFF_FC, OFF_FC + NFC)
OUT = np.arange(OFF_OUT, OFF_OUT + NOUT)


def build_shared_conv(pre_grid, grid_h, grid_w, n_maps_in, n_maps_out, kernel, stride, out_offset, bias_idx):
    """标准卷积（权重共享）：n_maps_out 个滤波器 × kernel² 个共享权重 + 每滤波器 1 个共享 bias。
    返回: PRE/POST 边表, KIDX 每条边对应的共享参数索引（0..n_maps_out*kernel²-1 为核权重, 之后 n_maps_out 个为 bias）。
    """
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
    # 共享参数索引：核位置按滤波器平移
    kpos = np.tile(np.arange(n_maps_in * kernel * kernel + 1), n_out)
    kidx = np.tile(np.arange(n_maps_in * kernel * kernel + 1), n_out)
    fmap = np.repeat(np.arange(n_maps_out), oh * ow * (n_maps_in * kernel * kernel + 1))
    kidx = fmap * (n_maps_in * kernel * kernel + 1) + kpos
    return pre.ravel().astype(np.int64), post.astype(np.int64), n_out, kidx.astype(np.int64)


in_grid = np.arange(784).reshape(28, 28)
PRE1, POST1, N1, KIDX1 = build_shared_conv(in_grid[None], 28, 28, 1, C1, K1, 2, OFF_F, OFF_B1)
N_S1 = C1 * (K1 * K1 + 1)            # 共享参数数（核 + bias）
PRE2 = np.tile(np.concatenate([[OFF_B2], F]), NFC).astype(np.int64)
POST2 = np.repeat(FC, N_F + 1)
PRE3 = np.tile(np.concatenate([[OFF_B3], FC]), NOUT).astype(np.int64)
POST3 = np.repeat(OUT, NFC + 1)

G1 = N_S1
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
E1 = np.zeros(len(PRE1))          # 资格迹按边（各位置局部信号）
E2 = np.zeros(G2 - G1)
E3 = np.zeros(G3 - G2)
y_onehot = np.eye(NOUT)


def spiking_step(x_vec, y_vec, learn):
    global u, r_est, f_est, E1, E2, E3, P, n_out_last, n1_last, n2_last
    if TAU_M > 0.0:
        u *= np.exp(-DT / TAU_M)   # LIF 膜泄漏（论文 §4.5：du/dt = I - λu）；TAU_M=0 时退化为 IF
    pre_spikes = np.zeros(N_NEURONS)
    pre_spikes[OFF_IN:OFF_IN + 784] = RNG.poisson(R_IN * x_vec * DT)
    pre_spikes[OFF_B1] = RNG.poisson(BIAS_RATE * DT)
    pre_spikes[OFF_B2] = RNG.poisson(BIAS_RATE * DT)
    pre_spikes[OFF_B3] = RNG.poisson(BIAS_RATE * DT)

    # conv1（权重共享）：每条边按共享 p 释放
    k1 = RNG.binomial(pre_spikes[PRE1].astype(np.int64), P[KIDX1])
    u[F] += np.bincount(POST1 - OFF_F, SIGN[KIDX1] * k1, minlength=N_F)
    E1 += -E1 * DT / TAU_E + pre_spikes[PRE1]
    n1 = np.floor(u[F] / THETA).clip(0.0, None).astype(np.int64)
    u[F] -= n1 * THETA
    r_est[F] += -r_est[F] * DT / TAU_R + n1
    pre_spikes[F] = n1
    n1_last = n1

    k2 = RNG.binomial(pre_spikes[PRE2].astype(np.int64), P[G1:G2])
    u[FC] += np.bincount(POST2 - OFF_FC, SIGN[G1:G2] * k2, minlength=NFC)
    E2 += -E2 * DT / TAU_E + pre_spikes[PRE2]
    n2 = np.floor(u[FC] / THETA).clip(0.0, None).astype(np.int64)
    u[FC] -= n2 * THETA
    r_est[FC] += -r_est[FC] * DT / TAU_R + n2
    pre_spikes[FC] = n2
    n2_last = n2

    k3 = RNG.binomial(pre_spikes[PRE3].astype(np.int64), P[G2:G3])
    if KAPPA > 0.0 or GAMMA > 0.0:
        out_rate = f_est / TAU_F
        S_out = np.sum(out_rate)
        u[OUT] -= (KAPPA * S_out + GAMMA * (S_out - out_rate)) * DT
    u[OUT] += np.bincount(POST3 - OFF_OUT, SIGN[G2:G3] * k3, minlength=NOUT)
    E3 += -E3 * DT / TAU_E + pre_spikes[PRE3]
    n_out = np.floor(u[OUT] / THETA).clip(0.0, None).astype(np.int64)
    n_out_last = n_out
    u[OUT] -= n_out * THETA
    r_est[OUT] += -r_est[OUT] * DT / TAU_R + n_out
    f_est += -f_est * DT / TAU_F + n_out

    if learn:
        w2 = SIGN[G1:G2] * P[G1:G2] / THETA
        w3 = SIGN[G2:G3] * P[G2:G3] / THETA

        gateF = (r_est[F] / TAU_R) > R_GATE
        gateFC = (r_est[FC] / TAU_R) > R_GATE

        d_out = f_est / TAU_F - TARGET * y_vec
        d_fc = gateFC * np.bincount(PRE3 - OFF_FC, w3 * d_out[POST3 - OFF_OUT], minlength=NFC + 1)[:NFC]
        d_f = np.bincount(PRE2 - OFF_F, w2 * d_fc[POST2 - OFF_FC], minlength=N_F + 1)[:N_F]

        d1 = gateF[POST1 - OFF_F] * d_f[POST1 - OFF_F]      # 每条边（各位置）的局域 δ
        d2 = d_fc[POST2 - OFF_FC]
        d3 = d_out[POST3 - OFF_OUT]

        kk1 = RNG.poisson(np.clip(FDA - d1, 0.0, None) * DT)
        kk2 = RNG.poisson(np.clip(FDA - d2, 0.0, None) * DT)
        kk3 = RNG.poisson(np.clip(FDA - d3, 0.0, None) * DT)

        # 共享权重更新：组内（各位置）局部信号求和 —— 扫描/眼动复用的物理实现
        sum1 = np.bincount(KIDX1, E1 * (kk1 - C * DT), minlength=N_S1)
        P[:G1] += SIGN[:G1] * ALPHA * sum1
        P[G1:G2] += SIGN[G1:G2] * ALPHA * E2 * (kk2 - C * DT)
        P[G2:G3] += SIGN[G2:G3] * ALPHA * E3 * (kk3 - C * DT)
        P = np.clip(P, 1e-6, 1.0 - 1e-6)


def reset_state():
    global u, r_est, f_est, E1, E2, E3
    u[:] = 0; r_est[:] = 0; f_est[:] = 0; E1[:] = 0; E2[:] = 0; E3[:] = 0


def evaluate(n_img):
    idx = RNG.choice(10000, n_img, replace=False)
    hits = 0
    for ii in idx:
        if RESET_PER_SAMPLE:
            reset_state()
        x = te_flat[ii]
        y = tel[ii]
        for _ in range(steps):
            spiking_step(x, y, learn=False)
        if ISI_STEPS > 0:
            zx = np.zeros(784)
            zy = np.zeros(NOUT)
            for _ in range(ISI_STEPS):
                spiking_step(zx, zy, learn=False)
        hits += int(np.argmax(f_est) == y)
    return hits / n_img


def save_checkpoint(path="mnist_checkpoint.npz"):
    np.savez(path, P=P, SIGN=SIGN, R_IN=R_IN, ALPHA=ALPHA, SEED=SEED,
             PRE1=PRE1, POST1=POST1, KIDX1=KIDX1, PRE2=PRE2, POST2=POST2, PRE3=PRE3, POST3=POST3)


def run_training():
    global u, r_est, f_est, E1, E2, E3, P
    print(f"topology: 784 -> CONV5x5x{C1} stride2 (共享权重, {N_S1} 参数) -> FC{NFC} -> {NOUT}"
          f"  ({N_NEURONS} neurons, {G3} params, {len(PRE1) + len(PRE2) + len(PRE3)} edges)  seed={SEED}", flush=True)
    print(f"R={R_IN}  f_da=C={FDA}  TARGET={TARGET}  alpha={ALPHA}  sample={SAMPLE_T}s  train={N_TRAIN}", flush=True)

    losses = np.zeros(N_TRAIN)
    accs = np.zeros(N_TRAIN)
    eval_pts = []
    t0 = time.time()
    for smp in range(N_TRAIN):
        if RESET_PER_SAMPLE:
            reset_state()
        ii = order[smp]
        x = tr_flat[ii]
        y = trl[ii]
        yv = y_onehot[y]
        for _ in range(steps):
            spiking_step(x, yv, learn=True)
        if ISI_STEPS > 0:
            zx = np.zeros(784)
            zy = np.zeros(NOUT)
            for _ in range(ISI_STEPS):
                spiking_step(zx, zy, learn=False)   # 静默期无输入无目标：无反向误差（误差与样本绑定），仅状态自然衰减
        f = f_est / TAU_F
        losses[smp] = float(np.mean(0.5 * ((f - TARGET * yv) / TARGET) ** 2))
        accs[smp] = float(np.argmax(f) == y)
        if (smp + 1) % 250 == 0:
            print(f"  train {smp + 1}/{N_TRAIN}  loss(roll100) {np.mean(losses[smp - 99:smp + 1]):.4f}"
                  f"  acc(roll100) {np.mean(accs[smp - 99:smp + 1]):.3f}  ({time.time() - t0:.0f} s)", flush=True)
        if (smp + 1) % 500 == 0:
            fr = f_est / TAU_F
            print(f"    out rates: y={fr[y]:.0f} mean_other={np.mean(np.delete(fr, y)):.0f}"
                  f" total={fr.sum():.0f}", flush=True)
        if (smp + 1) % EVAL_EVERY == 0:
            a = evaluate(N_EVAL)
            eval_pts.append((smp + 1, a))
            print(f"  >>> test acc @ {smp + 1}: {a:.3f}", flush=True)
        if (smp + 1) % 2000 == 0:
            save_checkpoint()
            print(f"    [checkpoint saved @ {smp + 1}]", flush=True)

    train_time = time.time() - t0
    a_final = evaluate(N_FINAL_EVAL)
    print(f"training done in {train_time:.0f} s; final test acc (n={N_FINAL_EVAL}) = {a_final:.4f}", flush=True)

    save_checkpoint()

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
