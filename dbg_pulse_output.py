"""Diagnose pulse readout bottleneck: output spike counts / rates for mean-field weights.
Usage: python3 dbg_pulse_output.py <ckpt.npz> [n_img] [SAMPLE_T]
"""
import sys
import numpy as np

CKPT = sys.argv[1] if len(sys.argv) > 1 else "meanfield_ckpts/k0p2/meanfield_checkpoint.npz"
N_IMG = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SAMPLE_T = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

sys.argv = ["mnist_shared.py", "0", "10", str(SAMPLE_T), "200", "3e-8", "3000", "30", "1000", "10", "0.02", "0.2", "0"]
import mnist_shared as m

z = np.load(CKPT)
m.P[:] = z["P"]
if "SIGN" in z:
    m.SIGN[:] = z["SIGN"]
print(f"ckpt={CKPT}  KAPPA_in_file={z['KAPPA'] if 'KAPPA' in z else '?'}  m.KAPPA={m.KAPPA}  steps={m.steps}", flush=True)

all_spk = []
all_rate = []
all_u_out = []
for ii in range(N_IMG):
    x = m.te_flat[ii]
    y = m.tel[ii]
    spk = np.zeros(m.NOUT)
    for _ in range(m.steps):
        m.spiking_step(x, m.y_onehot[y], learn=False)
        spk += m.n_out_last
    all_spk.append(spk)
    all_rate.append(m.f_est / m.TAU_F)
    all_u_out.append(m.u[m.OUT].copy())
    print(f"img {ii} label={y}  out_spikes={spk.astype(int)}  f_est_rate={np.round(m.f_est / m.TAU_F, 1)}  u_out={np.round(m.u[m.OUT], 1)}", flush=True)

S = np.array(all_spk)
print(f"\nmean out spikes per sample: {S.mean(0).round(1)}   (total {S.sum(1).round(0)})", flush=True)
print(f"f_est rates at end: mean {np.mean(all_rate, 0).round(0)}", flush=True)
print(f"u_out end: mean {np.mean(all_u_out, 0).round(0)}", flush=True)
