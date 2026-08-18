import sys
import time
import numpy as np

CSV = "mnist_table_results.csv"
if not __import__("os").path.exists(CSV):
    with open(CSV, "w", encoding="utf-8") as fh:
        fh.write("which,seed,N,SAMPLE_T,R,alpha,FDA,BIAS,TARGET,KAPPA,GAMMA,DT,loss_plateau,loss_std,frozen_acc,"
                 "align_all,align_std,corr_W1,corr_W2,corr_W3,bias_W1,bias_W2,bias_W3,"
                 "snr_W1,snr_W2,snr_W3,eff_W1,eff_W2,eff_W3,sign_W1,sign_W2,sign_W3,"
                 "align_W1,align_W2,align_W3,ratio_W1,ratio_W2,ratio_W3\n")

# usage: python mnist_table.py <shared|stride> <N> <SAMPLE_T> <R> <alpha> <FDA> <BIAS> <TARGET> [seed] [KAPPA] [GAMMA] [DT]
which = sys.argv[1]
N_TRAIN = int(sys.argv[2])
SAMPLE_T = float(sys.argv[3])
R_IN = float(sys.argv[4])
ALPHA = float(sys.argv[5])
FDA = float(sys.argv[6])
BIAS = float(sys.argv[7])
TARGET = float(sys.argv[8])
SEED = int(sys.argv[9]) if len(sys.argv) > 9 else 0
KAPPA = float(sys.argv[10]) if len(sys.argv) > 10 else 0.0
GAMMA = float(sys.argv[11]) if len(sys.argv) > 11 else 0.0
DT = float(sys.argv[12]) if len(sys.argv) > 12 else 0.02
FINAL_EVAL_N = 500

if which == "shared":
    sys.argv = ["mnist_shared.py", str(SEED), str(N_TRAIN), str(SAMPLE_T), str(R_IN), str(ALPHA),
                str(FDA), str(BIAS), str(TARGET), str(FINAL_EVAL_N), str(DT), str(KAPPA), str(GAMMA)]
    import mnist_shared as m
else:
    sys.argv = ["mnist_shallow.py", str(SEED), str(N_TRAIN), str(SAMPLE_T), str(R_IN), str(ALPHA),
                str(FDA), str(BIAS), "stride", str(TARGET), str(FINAL_EVAL_N), "0.02", "1.0"]
    import mnist_shallow as m

TH = m.THETA
LAYERS = (("W1", slice(0, m.G1)), ("W2", slice(m.G1, m.G2)), ("W3", slice(m.G2, m.G3)))


def rates_of(P, x):
    a_in = np.zeros(m.N_NEURONS)
    a_in[m.OFF_IN:m.OFF_IN + 784] = m.R_IN * x
    a_in[m.OFF_B1] = m.BIAS_RATE
    a_in[m.OFF_B2] = m.BIAS_RATE
    a_in[m.OFF_B3] = m.BIAS_RATE
    if which == "shared":
        z1 = np.bincount(m.POST1 - m.OFF_F, m.SIGN[m.KIDX1] * P[m.KIDX1] / TH * a_in[m.PRE1], minlength=m.N_F)
        a1 = np.maximum(z1, 0)
        p1v = np.zeros(m.N_NEURONS)
        p1v[m.F] = a1
        p1v[m.OFF_B2] = m.BIAS_RATE
    else:  # stride
        z1 = np.bincount(m.POST1 - m.OFF_F, m.SIGN[:m.G1] * P[:m.G1] / TH * a_in[m.PRE1], minlength=m.N_F)
        a1 = np.maximum(z1, 0)
        p1v = np.zeros(m.N_NEURONS)
        p1v[m.F] = a1
        p1v[m.OFF_B2] = m.BIAS_RATE
    z2 = np.bincount(m.POST2 - m.OFF_FC, m.SIGN[m.G1:m.G2] * P[m.G1:m.G2] / TH * p1v[m.PRE2], minlength=m.NFC)
    a2 = np.maximum(z2, 0)
    a2b = np.concatenate([a2, [m.BIAS_RATE]])
    z3 = np.bincount(m.POST3 - m.OFF_OUT, m.SIGN[m.G2:m.G3] * P[m.G2:m.G3] / TH * a2b[m.PRE3 - m.OFF_FC], minlength=10)
    a3 = np.maximum(z3, 0)
    return a1, a2, a3, a_in, p1v


def ana_grad(P, x, yv):
    a1, a2, a3, a_in, p1v = rates_of(P, x)
    d3 = a3 - m.TARGET * yv
    w3 = (m.SIGN[m.G2:m.G3] * P[m.G2:m.G3] / TH).reshape(10, m.NFC + 1)
    d2 = (a2 > 0) * (w3[:, 1:].T @ d3)
    d_f = np.bincount(m.PRE2 - m.OFF_F, m.SIGN[m.G1:m.G2] * P[m.G1:m.G2] / TH * d2[m.POST2 - m.OFF_FC], minlength=m.N_F + 1)[:m.N_F]
    if which == "shared":
        d1 = (a1 > 0)[m.POST1 - m.OFF_F] * d_f[m.POST1 - m.OFF_F]
        s = m.SIGN
        g = np.zeros(m.G3)
        g[:m.G1] = np.bincount(m.KIDX1, s[m.KIDX1] * a_in[m.PRE1] * d1, minlength=m.N_S1) / TH
    else:
        d1 = (a1 > 0)[m.POST1 - m.OFF_F] * d_f[m.POST1 - m.OFF_F]
        s = m.SIGN
        g = np.zeros(m.G3)
        g[:m.G1] = s[:m.G1] * a_in[m.PRE1] * d1 / TH
    g[m.G1:m.G2] = s[m.G1:m.G2] * p1v[m.PRE2] * d2[m.POST2 - m.OFF_FC] / TH
    a2b = np.concatenate([a2, [m.BIAS_RATE]])
    g[m.G2:m.G3] = s[m.G2:m.G3] * a2b[m.PRE3 - m.OFF_FC] * d3[m.POST3 - m.OFF_OUT] / TH
    return g


m.u[:] = 0
m.r_est[:] = 0
m.f_est[:] = 0
m.E1[:] = 0
m.E2[:] = 0
m.E3[:] = 0
m.P[:] = np.full(m.G3, m.P_INIT)
m.P[m.G2 + np.arange(m.NOUT) * (m.NFC + 1)] = 0.6

n_cos = np.zeros((len(LAYERS), 2))
n_ratio = np.zeros((len(LAYERS), 2))
n_g = [np.zeros(sl.stop - sl.start) for _, sl in LAYERS]
n_snr = [np.zeros(sl.stop - sl.start) for _, sl in LAYERS]
n_snr2 = [np.zeros(sl.stop - sl.start) for _, sl in LAYERS]
losses = np.zeros(N_TRAIN)
accs = np.zeros(N_TRAIN)
S = np.zeros((3, 5))
cov_start = max(0, N_TRAIN - 5)

t0 = time.time()
for smp in range(N_TRAIN):
    ii = m.order[smp]
    x = m.tr_flat[ii]
    y = m.trl[ii]
    yv = m.y_onehot[y]
    P0 = m.P.copy()
    g = ana_grad(m.P, x, yv)
    for st in range(m.steps):
        if smp >= cov_start:
            Eb = (m.E1.copy(), m.E2.copy(), m.E3.copy())
            m.spiking_step(x, yv, learn=True)
            w2 = m.SIGN[m.G1:m.G2] * m.P[m.G1:m.G2] / TH
            w3 = m.SIGN[m.G2:m.G3] * m.P[m.G2:m.G3] / TH
            gateFC = (m.r_est[m.FC] / m.TAU_R) > m.R_GATE
            gateF = (m.r_est[m.F] / m.TAU_R) > m.R_GATE
            d_out = m.f_est / m.TAU_F - m.TARGET * yv
            d_fc = gateFC * np.bincount(m.PRE3 - m.OFF_FC, w3 * d_out[m.POST3 - m.OFF_OUT], minlength=m.NFC + 1)[:m.NFC]
            d_f = np.bincount(m.PRE2 - m.OFF_F, w2 * d_fc[m.POST2 - m.OFF_FC], minlength=m.N_F + 1)[:m.N_F]
            if which == "shared":
                d1 = gateF[m.POST1 - m.OFF_F] * d_f[m.POST1 - m.OFF_F]
                ds = (d1, d_fc[m.POST2 - m.OFF_FC], d_out[m.POST3 - m.OFF_OUT])
            else:
                d1 = gateF[m.POST1 - m.OFF_F] * d_f[m.POST1 - m.OFF_F]
                ds = (d1, d_fc[m.POST2 - m.OFF_FC], d_out[m.POST3 - m.OFF_OUT])
            for gi in range(3):
                e = Eb[gi]
                d = ds[gi]
                S[gi, 0] += e.sum(); S[gi, 1] += d.sum()
                S[gi, 2] += (e * d).sum(); S[gi, 3] += (e * e).sum(); S[gi, 4] += (d * d).sum()
        else:
            m.spiking_step(x, yv, learn=True)
    dP = m.P - P0
    f = m.f_est / m.TAU_F
    losses[smp] = float(np.mean(0.5 * ((f - m.TARGET * yv) / m.TARGET) ** 2))
    accs[smp] = float(np.argmax(f) == y)
    for li, (_, sl) in enumerate(LAYERS):
        gs = g[sl]
        ds = dP[sl]
        n_g[li] += gs
        gn = np.linalg.norm(gs)
        dn = np.linalg.norm(ds)
        if gn > 1e-9 and dn > 1e-9:
            c = float(np.dot(gs, ds) / (gn * dn))   # cos with raw gradient; descent alignment = -cos
            n_cos[li, 0] += c
            n_cos[li, 1] += c * c
        n_ratio[li, 0] += dn / (gn + 1e-9)
        n_ratio[li, 1] += 1
        n_snr[li] += dP[sl]
        n_snr2[li] += dP[sl] ** 2
    if (smp + 1) % 200 == 0:
        print(f"  {smp + 1}/{N_TRAIN}  loss_roll {np.mean(losses[max(0, smp - 99):smp + 1]):.4f}  ({time.time() - t0:.0f}s)", flush=True)

# covariance stats
n_steps = (N_TRAIN - cov_start) * m.steps
print("== covariance (last %d samples) ==" % (N_TRAIN - cov_start), flush=True)
cov_rows = []
for gi, (name, _) in enumerate(LAYERS):
    n_el = n_steps * (len(m.E1) if gi == 0 else (len(m.PRE2) if gi == 1 else len(m.PRE3)))
    me = S[gi, 0] / n_el
    md = S[gi, 1] / n_el
    cov = S[gi, 2] / n_el - me * md
    vare = S[gi, 3] / n_el - me * me
    vard = S[gi, 4] / n_el - md * md
    corr = cov / np.sqrt(vare * vard + 1e-12)
    bias = abs(cov) / (abs(me * md) + 1e-12)
    cov_rows.append((corr, bias))
    print(f"  {name}: corr(e,d)={corr:+.3f}  bias={bias:.3f}  E[e]={me:.1f} E[d]={md:.1f}", flush=True)

# per-sample alignment
print("== per-sample alignment (cos with descent dir -g) ==", flush=True)
g_al = np.zeros(3)
g_pk = np.zeros(3)
snr_rows = []
eff_rows = []
sign_rows = []
etaT = m.ALPHA * m.TAU_E * m.SAMPLE_T   # theoretical per-sample factor: E[ΔP] = -etaT * g
for li, (name, sl) in enumerate(LAYERS):
    cmean = -n_cos[li, 0] / N_TRAIN                      # -cos = alignment with descent direction
    cstd = np.sqrt(max(n_cos[li, 1] / N_TRAIN - (n_cos[li, 0] / N_TRAIN) ** 2, 0))
    ratio = n_ratio[li, 0] / N_TRAIN
    dP_mean = n_snr[li] / N_TRAIN
    g_mean = n_g[li] / N_TRAIN
    dP_std = np.sqrt(np.maximum(n_snr2[li] / N_TRAIN - dP_mean ** 2, 0))
    snr = np.linalg.norm(dP_mean) / (np.linalg.norm(dP_std) + 1e-12)
    eff = np.linalg.norm(dP_mean) / (etaT * np.linalg.norm(g_mean) + 1e-12)
    sg = np.sign(g_mean)
    sm = np.sign(-dP_mean)
    sign_c = float(np.mean(sm[g_mean != 0] == sg[g_mean != 0])) if (g_mean != 0).any() else float("nan")
    snr_rows.append(snr)
    eff_rows.append(eff)
    sign_rows.append(sign_c)
    print(f"  {name}: align={cmean:+.3f}±{cstd:.3f}  |dP|/|g|={ratio:.3f}  updateSNR={snr:.3f}"
          f"  expEff={eff:.3f}  signCons={sign_c:.3f}", flush=True)

g_all = -(n_cos[0, 0] + n_cos[1, 0] + n_cos[2, 0]) / (3 * N_TRAIN)
g_std = np.sqrt((n_cos[0, 1] + n_cos[1, 1] + n_cos[2, 1]) / (3 * N_TRAIN) - (-g_all) ** 2)
print(f"  ALL : align={g_all:+.3f}±{g_std:.3f}")

# frozen eval
m.u[:] = 0; m.r_est[:] = 0; m.f_est[:] = 0
m.E1[:] = 0; m.E2[:] = 0; m.E3[:] = 0
idx = np.random.default_rng(SEED).choice(10000, FINAL_EVAL_N, replace=False)
hits = 0
for ii in idx:
    x = m.te_flat[ii]
    y = m.tel[ii]
    for _ in range(m.steps):
        m.spiking_step(x, m.y_onehot[y], learn=False)
    hits += int(np.argmax(m.f_est / m.TAU_F) == y)
frozen = hits / FINAL_EVAL_N

lp = losses[-100:].mean()
lps = losses[-100:].std()
print(f"loss_plateau(last100) = {lp:.4f} ± {lps:.4f}   train_acc_roll = {accs[-100:].mean():.3f}   frozen_test_acc = {frozen:.4f}")

row = (f"{which},{SEED},{N_TRAIN},{SAMPLE_T},{R_IN},{ALPHA},{FDA},{BIAS},{TARGET},{KAPPA},{GAMMA},{DT},"
       f"{lp:.4f},{lps:.4f},{frozen:.4f},{g_all:+.3f},{g_std:.3f},"
       + ",".join(f"{c:+.3f}" for c, _ in cov_rows) + ","
       + ",".join(f"{b:.3f}" for _, b in cov_rows) + ","
       + ",".join(f"{s:.3f}" for s in snr_rows) + ","
       + ",".join(f"{e:.3f}" for e in eff_rows) + ","
       + ",".join(f"{sc:.3f}" for sc in sign_rows) + ","
       + f"{-(n_cos[0,0]/N_TRAIN):+.3f},{-(n_cos[1,0]/N_TRAIN):+.3f},{-(n_cos[2,0]/N_TRAIN):+.3f},"
       + f"{n_ratio[0,0]/N_TRAIN:.3f},{n_ratio[1,0]/N_TRAIN:.3f},{n_ratio[2,0]/N_TRAIN:.3f}")
print("TSVROW|" + row, flush=True)
with open("mnist_table_results.csv", "a", encoding="utf-8") as fh:
    fh.write(row + "\n")