import sys
import time
import numpy as np

alpha = float(sys.argv[1])
tau_sm = float(sys.argv[2])
n_train = int(sys.argv[3]) if len(sys.argv) > 3 else 800
final_eval_n = int(sys.argv[4]) if len(sys.argv) > 4 else 200

sys.argv = ["mnist_shallow.py", "0", str(n_train), "1.0", "200", str(alpha), "3000", "30",
            "stride", "200", str(final_eval_n), "0.02", "1.0", "ce", str(tau_sm)]
import mnist_shallow as m

m.N_TRAIN = n_train
m.EVAL_EVERY = 10 ** 9
m.N_FINAL_EVAL = final_eval_n
np.random.seed = None

losses = np.zeros(n_train)
accs = np.zeros(n_train)
t0 = time.time()
order = m.order
for smp in range(n_train):
    ii = order[smp]
    x = m.tr_flat[ii]
    y = m.trl[ii]
    yv = m.y_onehot[y]
    for _ in range(m.steps):
        m.spiking_step(x, yv, learn=True)
    f = m.f_est / m.TAU_F
    fh = f / tau_sm
    mx = fh.max()
    p = np.exp(fh - mx)
    p = p / p.sum()
    losses[smp] = -np.log(p[y] + 1e-12)
    accs[smp] = float(np.argmax(f) == y)
    if (smp + 1) % 200 == 0:
        print(f"  {smp + 1}/{n_train}  loss_roll100 {np.mean(losses[smp - 99:smp + 1]):.3f}"
              f"  acc_roll100 {np.mean(accs[smp - 99:smp + 1]):.3f}  ({time.time() - t0:.0f}s)", flush=True)

rate_mean = np.mean(m.r_est[m.OUT]) / m.TAU_R
print(f"alpha={alpha} tau_sm={tau_sm}: loss_early={np.mean(losses[:100]):.3f} loss_late={np.mean(losses[-100:]):.3f} "
      f"acc_early={np.mean(accs[:100]):.3f} acc_late={np.mean(accs[-100:]):.3f} "
      f"out_rate_mean={rate_mean:.0f}Hz", flush=True)

idx = np.random.default_rng(0).choice(10000, final_eval_n, replace=False)
m.u = np.zeros(m.N_NEURONS)
m.r_est = np.zeros(m.N_NEURONS)
m.f_est = np.zeros(10)
m.E1 = np.zeros(m.G1); m.E2 = np.zeros(m.G2 - m.G1); m.E3 = np.zeros(m.G3 - m.G2)
hits = 0
out_rates = []
for ii in idx:
    x = m.te_flat[ii]
    y = m.tel[ii]
    for _ in range(m.steps):
        m.spiking_step(x, m.y_onehot[y], learn=False)
    f = m.f_est / m.TAU_F
    out_rates.append(f)
    hits += int(np.argmax(f) == y)
out_rates = np.array(out_rates)
print(f"frozen test acc = {hits / final_eval_n:.3f}   out rate min/max/mean = "
      f"{out_rates.min():.0f}/{out_rates.max():.0f}/{out_rates.mean():.0f} Hz", flush=True)