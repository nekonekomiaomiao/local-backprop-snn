import sys
import numpy as np
sys.argv = ["mnist_shallow.py", "0", "50", "1.0", "200", "1e-6", "3000", "30", "stride", "200", "100", "0.02", "1.0", "ce", "200"]
import mnist_shallow as m

ck = np.load("mnist_checkpoint.npz")
m.P = ck["P"].copy()
SIGN = ck["SIGN"]
w = SIGN * m.P
print("P stats per layer:")
for name, sl in (("W1", slice(0, m.G1)), ("W2", slice(m.G1, m.G2)), ("W3", slice(m.G2, m.G3))):
    print(f"  {name}: mean={m.P[sl].mean():.4f}  std={m.P[sl].std():.4f}  min={m.P[sl].min():.5f}  max={m.P[sl].max():.5f}  sat(<0.01|>0.99): {int(((m.P[sl]<0.01)|(m.P[sl]>0.99)).sum())}/{m.P[sl].size}")

idx = np.random.default_rng(0).choice(10000, 200, replace=False)
m.u = np.zeros(m.N_NEURONS)
m.r_est = np.zeros(m.N_NEURONS)
m.f_est = np.zeros(10)
m.E1 = np.zeros(m.G1); m.E2 = np.zeros(m.G2 - m.G1); m.E3 = np.zeros(m.G3 - m.G2)
outs = []
correct = 0
for ii in idx:
    x = m.te_flat[ii]; y = m.tel[ii]
    yv = m.y_onehot[y]
    for _ in range(m.steps):
        m.spiking_step(x, yv, learn=False)
    f = m.f_est / m.TAU_F
    outs.append(f)
    correct += int(np.argmax(f) == y)
outs = np.array(outs)
print(f"frozen eval acc = {correct/200:.3f}")
print("output rates: mean per class:", np.round(outs.mean(axis=0), 1))
print("  min/max overall:", round(outs.min(), 1), round(outs.max(), 1))
pmat = np.exp(outs - outs.max(axis=1, keepdims=True))
pmat = pmat / pmat.sum(axis=1, keepdims=True)
print("  p_correct(y) mean:", round(float(np.mean(pmat[np.arange(200), m.tel[idx]])), 4))
print("  p on class 0 (all zeros case):", round(float(pmat[:, 0].mean()), 4))