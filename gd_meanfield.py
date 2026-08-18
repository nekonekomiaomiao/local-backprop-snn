import sys
import time
import numpy as np

LR = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-8
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
sys.argv = ["mnist_shallow.py", "0", "2000", "1.0", "200", "5e-8", "4000", "30", "stride"]
import mnist_shallow as m
import verify_mnist_shallow as v

rng = np.random.default_rng(0)
order = rng.permutation(60000)[:N]
P = np.full(m.G3, m.P_INIT)
P[m.G2 + np.arange(m.NOUT) * (m.NFC + 1)] = 0.6

t0 = time.time()
for s in range(N):
    v.x = m.tr_flat[order[s]]
    v.yv = m.y_onehot[m.trl[order[s]]]
    g = v.ana_grad(P)
    P -= LR * g
    P = np.clip(P, 1e-6, 1.0 - 1e-6)
    if (s + 1) % 250 == 0:
        print(f"  {s + 1}/{N}  ({time.time() - t0:.0f} s)", flush=True)

m.P[:] = P
from eval_meanfield import mean_field_out
hits = 0
rng2 = np.random.default_rng(2)
idx = rng2.choice(10000, 1000, replace=False)
for ii in idx:
    a3 = mean_field_out(m.te_flat[ii], P)
    hits += int(np.argmax(a3) == m.tel[ii])
print(f"mean-field GD (lr={LR}, {N} samples): test acc = {hits / 1000:.4f}")
