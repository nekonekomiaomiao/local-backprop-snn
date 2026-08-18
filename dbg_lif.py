"""LIF training-failure diagnostic (rebuilt 2026-08-16).

Usage: python3 dbg_lif.py <ckpt.npz> <TAU_M> <ISI_STEPS> [N_SAMPLES=300] [SAMPLE_T=1.0] [SEED=0]

Measures over N training samples (learn=True, same order as training):
  1. per-layer alignment of actual update dP vs analytic gradient
       a) IF-ReLU mean-field gradient (the reference used by mnist_table.py / old dbg_lif)
       b) LIF-corrected gradient (Phi = lam/ln(1+lam/x), x = S_all - lam, paper old v4.5)
  2. layer firing rates (F/FC/OUT), gate fractions, E-trace means
  3. delta statistics per output class (correct vs wrong)
  4. lagged cross-correlation E3(t) vs delta3(t) for the output layer (phase-lag test)
"""
import sys
import numpy as np

CKPT = sys.argv[1]
TAU_M = float(sys.argv[2])
ISI = int(sys.argv[3])
N = int(sys.argv[4]) if len(sys.argv) > 4 else 300
SAMPLE_T = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 0
RAND_INIT = len(sys.argv) > 7 and sys.argv[7] == "rand"
ISI_LEARN = int(sys.argv[8]) if len(sys.argv) > 8 else 0   # 1 = reproduce original learn=True-ISI bug
TAU_E_ARG = float(sys.argv[9]) if len(sys.argv) > 9 else 0.2   # eligibility-trace time constant (paper §8.5)
TARGET_ARG = float(sys.argv[10]) if len(sys.argv) > 10 else 1000.0   # output target rate (recalibration test)
KAPPA_ARG = float(sys.argv[11]) if len(sys.argv) > 11 else 0.2   # output-layer inhibition pool

sys.argv = ["mnist_shared.py", str(SEED), str(N), str(SAMPLE_T), "200", "1.5e-8", "3000", "30",
            str(TARGET_ARG),
            "1000", "0.02", str(KAPPA_ARG), "0", "0", str(ISI), str(TAU_M), str(TAU_E_ARG)]
import mnist_shared as m

TH = m.THETA
LAM = 1.0 / TAU_M if TAU_M > 0 else 0.0
LAYERS = (("W1", slice(0, m.G1)), ("W2", slice(m.G1, m.G2)), ("W3", slice(m.G2, m.G3)))

z = np.load(CKPT)
m.P[:] = z["P"]
if "SIGN" in z:
    m.SIGN[:] = z["SIGN"]
if RAND_INIT:
    m.P[:] = np.full(m.G3, m.P_INIT)
    m.P[m.G2 + np.arange(m.NOUT) * (m.NFC + 1)] = 0.6
    print("RANDOM INIT applied", flush=True)
print(f"ckpt={CKPT}  TAU_M={TAU_M} (lam={LAM:.3f})  ISI={ISI}  N={N}  SAMPLE_T={SAMPLE_T}  "
      f"P mean={m.P.mean():.4f}  seed={SEED}", flush=True)


def phi(x, lam):
    if lam <= 0:
        return np.maximum(x, 0)
    return np.where(x > 0, lam / np.log(1.0 + lam / np.maximum(x, 1e-9)), 0.0)


def phi_deriv(x, lam):
    if lam <= 0:
        return (x > 0).astype(float)
    z = np.maximum(x, 1e-9)
    lnz = np.log(1.0 + lam / z)
    return np.where(x > 0, lam * lam / (z * (z + lam) * lnz * lnz), 0.0)


def rates_of_lif(P, x, lam):
    """Mean-field forward under LIF (old paper v4.5): f_out = Phi(S_all - lam)."""
    a_in = np.zeros(m.N_NEURONS)
    a_in[m.OFF_IN:m.OFF_IN + 784] = m.R_IN * x
    a_in[m.OFF_B1] = m.BIAS_RATE
    a_in[m.OFF_B2] = m.BIAS_RATE
    a_in[m.OFF_B3] = m.BIAS_RATE
    z1 = np.bincount(m.POST1 - m.OFF_F, m.SIGN[m.KIDX1] * P[m.KIDX1] / TH * a_in[m.PRE1], minlength=m.N_F)
    a1 = phi(z1 - lam, lam)
    p1v = np.zeros(m.N_NEURONS)
    p1v[m.F] = a1
    p1v[m.OFF_B2] = m.BIAS_RATE
    z2 = np.bincount(m.POST2 - m.OFF_FC, m.SIGN[m.G1:m.G2] * P[m.G1:m.G2] / TH * p1v[m.PRE2], minlength=m.NFC)
    a2 = phi(z2 - lam, lam)
    a2b = np.concatenate([a2, [m.BIAS_RATE]])
    z3 = np.bincount(m.POST3 - m.OFF_OUT, m.SIGN[m.G2:m.G3] * P[m.G2:m.G3] / TH * a2b[m.PRE3 - m.OFF_FC], minlength=10)
    a3 = phi(z3 - lam, lam)
    return a1, a2, a3, a_in, p1v, z1, z2, z3


def ana_grad_if(P, x, yv):
    """IF-ReLU reference gradient (same as mnist_table.ana_grad)."""
    a1, a2, a3, a_in, p1v, _, _, _ = rates_of_lif(P, x, 0.0)
    d3 = a3 - m.TARGET * yv
    w3 = (m.SIGN[m.G2:m.G3] * P[m.G2:m.G3] / TH).reshape(10, m.NFC + 1)
    d2 = (a2 > 0) * (w3[:, 1:].T @ d3)
    d_f = np.bincount(m.PRE2 - m.OFF_F, m.SIGN[m.G1:m.G2] * P[m.G1:m.G2] / TH * d2[m.POST2 - m.OFF_FC], minlength=m.N_F + 1)[:m.N_F]
    d1 = (a1 > 0)[m.POST1 - m.OFF_F] * d_f[m.POST1 - m.OFF_F]
    s = m.SIGN
    g = np.zeros(m.G3)
    g[:m.G1] = np.bincount(m.KIDX1, s[m.KIDX1] * a_in[m.PRE1] * d1, minlength=m.N_S1) / TH
    g[m.G1:m.G2] = s[m.G1:m.G2] * p1v[m.PRE2] * d2[m.POST2 - m.OFF_FC] / TH
    a2b = np.concatenate([a2, [m.BIAS_RATE]])
    g[m.G2:m.G3] = s[m.G2:m.G3] * a2b[m.PRE3 - m.OFF_FC] * d3[m.POST3 - m.OFF_OUT] / TH
    return g


def ana_grad_lif(P, x, yv, lam):
    """LIF-corrected gradient: backprop through Phi(S-lam), dPhi/dx = lam^2/(x(x+lam) ln(1+lam/x)^2)."""
    a1, a2, a3, a_in, p1v, z1, z2, z3 = rates_of_lif(P, x, lam)
    d3 = a3 - m.TARGET * yv
    w3 = (m.SIGN[m.G2:m.G3] * P[m.G2:m.G3] / TH).reshape(10, m.NFC + 1)
    ph2 = phi_deriv(z2 - lam, lam)
    ph1 = phi_deriv(z1 - lam, lam)
    d2 = ph2 * (w3[:, 1:].T @ d3)
    d_f = np.bincount(m.PRE2 - m.OFF_F, m.SIGN[m.G1:m.G2] * P[m.G1:m.G2] / TH * d2[m.POST2 - m.OFF_FC], minlength=m.N_F + 1)[:m.N_F]
    d1 = ph1[m.POST1 - m.OFF_F] * d_f[m.POST1 - m.OFF_F]
    s = m.SIGN
    g = np.zeros(m.G3)
    g[:m.G1] = np.bincount(m.KIDX1, s[m.KIDX1] * a_in[m.PRE1] * d1, minlength=m.N_S1) / TH
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

n_cos = {k: [np.zeros(2) for _ in LAYERS] for k in ("IF", "LIF")}
n_ratio = [np.zeros(2) for _ in LAYERS]
sig_dP = np.zeros(m.G3)
sig_g = np.zeros(m.G3)
rate_F = np.zeros(m.N_F)
rate_FC = np.zeros(m.NFC)
rate_OUT = np.zeros(m.NOUT)
spk_F = np.zeros(m.N_F)
spk_FC = np.zeros(m.NFC)
spk_OUT = np.zeros(m.NOUT)
n_steps = 0
gateF_on = 0.0
gateFC_on = 0.0
E_means = [np.zeros(len(m.PRE1)), np.zeros(len(m.PRE2)), np.zeros(len(m.PRE3))]
d3_correct = []
d3_wrong = []
lag_samples = []

t0 = __import__("time").time()
for smp in range(N):
    ii = m.order[smp]
    x = m.tr_flat[ii]
    y = m.trl[ii]
    yv = m.y_onehot[y]
    P0 = m.P.copy()
    g_if = ana_grad_if(m.P, x, yv)
    g_lif = ana_grad_lif(m.P, x, yv, LAM)
    e3s = []
    d3s = []
    for st in range(m.steps):
        m.spiking_step(x, yv, learn=True)
        spk_F += m.n1_last if hasattr(m, "n1_last") else 0
        spk_FC += m.n2_last if hasattr(m, "n2_last") else 0
        spk_OUT += m.n_out_last if hasattr(m, "n_out_last") else 0
        gateF_on += float(np.mean((m.r_est[m.F] / m.TAU_R) > m.R_GATE))
        gateFC_on += float(np.mean((m.r_est[m.FC] / m.TAU_R) > m.R_GATE))
        E_means[0] += m.E1
        E_means[1] += m.E2
        E_means[2] += m.E3
        d_out = m.f_est / m.TAU_F - m.TARGET * yv
        d3_correct.append(d_out[y])
        d3_wrong.append(np.mean(np.delete(d_out, y)))
        e3s.append(m.E3.copy())
        d3s.append(d_out.copy())
    if ISI > 0:
        zx = np.zeros(784)
        zy = np.zeros(m.NOUT)
        for _ in range(ISI):
            m.spiking_step(zx, zy, learn=bool(ISI_LEARN))
    if smp >= N - 5:
        lag_samples.append((np.array(e3s), np.array(d3s)))
    n_steps += m.steps
    dP = m.P - P0
    sig_dP += dP
    sig_g += g_if
    for li, (_, sl) in enumerate(LAYERS):
        gs = g_if[sl]
        ds = dP[sl]
        gn = np.linalg.norm(gs)
        dn = np.linalg.norm(ds)
        if gn > 1e-9:
            n_ratio[li][0] += dn / (gn + 1e-9)
            n_ratio[li][1] += 1
        for key, g in (("IF", g_if), ("LIF", g_lif)):
            gs = g[sl]
            ds = dP[sl]
            gn = np.linalg.norm(gs)
            dn = np.linalg.norm(ds)
            if gn > 1e-9 and dn > 1e-9:
                c = float(np.dot(gs, ds) / (gn * dn))
                n_cos[key][li][0] += c
                n_cos[key][li][1] += c * c
    if (smp + 1) % 100 == 0:
        print(f"  {smp + 1}/{N}  ({__import__('time').time() - t0:.0f}s)", flush=True)

print("== firing rates (mean Hz over sample time) ==", flush=True)
print(f"  F : mean={spk_F.sum() / n_steps / m.N_F / m.DT:.2f} Hz/neuron, "
      f"nonzero={np.mean(spk_F > 0):.3f}", flush=True)
print(f"  FC: mean={spk_FC.sum() / n_steps / m.NFC / m.DT:.2f} Hz/neuron, nonzero={np.mean(spk_FC > 0):.3f}", flush=True)
print(f"  OUT: mean={spk_OUT.sum() / n_steps / m.NOUT / m.DT:.2f} Hz/neuron, "
      f"per-class={spk_OUT.sum(0) / n_steps / m.DT:.1f}", flush=True)
print(f"  gates: gateF on={gateF_on / n_steps:.3f}  gateFC on={gateFC_on / n_steps:.3f}", flush=True)
print(f"  E means: W1 {E_means[0].mean() / n_steps:.3f}  W2 {E_means[1].mean() / n_steps:.3f}  "
      f"W3 {E_means[2].mean() / n_steps:.3f}", flush=True)
print(f"  delta3: correct={np.mean(d3_correct):+.1f}  wrong={np.mean(d3_wrong):+.1f}", flush=True)

print("== per-sample alignment (cos with descent dir -g) ==", flush=True)
for li, (name, _) in enumerate(LAYERS):
    ratio = n_ratio[li][0] / max(n_ratio[li][1], 1)
    for key in ("IF", "LIF"):
        cmean = -n_cos[key][li][0] / N
        cstd = np.sqrt(max(n_cos[key][li][1] / N - (n_cos[key][li][0] / N) ** 2, 0))
        print(f"  {name} vs {key}-grad: align={cmean:+.3f}±{cstd:.3f}", flush=True)
    print(f"  {name}: |dP|/|g|={ratio:.3f}", flush=True)
print("== SIGNAL-level alignment (mean dP vs mean g over all samples; predicts learning) ==", flush=True)
for li, (name, sl) in enumerate(LAYERS):
    sd = sig_dP[sl]
    sg = sig_g[sl]
    sn = np.linalg.norm(sd)
    gn = np.linalg.norm(sg)
    csig = float(np.dot(sd, sg) / (sn * gn + 1e-9)) if sn > 0 and gn > 0 else 0.0
    print(f"  {name}: cos(E[dP], E[g])={csig:+.3f}  |E[dP]|/|E[g]|={sn / (gn + 1e-9):.4f}", flush=True)

print("== E3-delta3 lagged correlation (output layer, last 5 samples) ==", flush=True)
if lag_samples:
    corr = []
    for e3s, d3s in lag_samples:
        T = e3s.shape[0]
        # e3 summed over the 33 FC inputs of each OUT neuron; correlate per OUT neuron then average
        e = e3s.sum(1)
        for k in range(m.NOUT):
            d = d3s[:, k]
            if d.std() < 1e-9:
                continue
            lag = np.arange(-T // 2, T // 2)
            cv = np.correlate(e - e.mean(), d - d.mean(), mode="full")
            cv = cv[len(cv) // 2 - T // 2:len(cv) // 2 + T // 2 + 1]
            denom = e.std() * d.std() * T
            corr.append(cv / (denom + 1e-12))
    if corr:
        corr = np.mean(corr, 0)
        T = len(corr) // 2
        for tau in (-T, -T // 2, 0, T // 4, T // 2, T - 1):
            print(f"  tau={tau * m.DT:+.2f}s: corr={corr[tau + T]:+.4f}", flush=True)
        peak = np.argmax(np.abs(corr))
        print(f"  peak |corr| at tau={ (peak - T) * m.DT:+.2f}s value={corr[peak]:+.4f}", flush=True)
print("done", flush=True)