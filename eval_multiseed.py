import sys, numpy as np
CKPTS = list(sys.argv[1:])
sys.argv = ["mnist_shared.py", "0", "500", "1.0", "200", "3e-8", "3000", "30", "1000",
            "1000", "0.02", "1.0", "0", "0", "100", "0.5"]
import mnist_shared as m
for ckpt in CKPTS:
    z = np.load(ckpt)
    P = z["P"].copy()
    if "SIGN" in z:
        m.SIGN[:] = z["SIGN"]
    res = []
    for seed in [123, 1, 2, 3, 4]:
        m.P[:] = P
        m.u[:] = 0; m.r_est[:] = 0; m.f_est[:] = 0; m.E1[:] = 0; m.E2[:] = 0; m.E3[:] = 0
        idx = np.random.default_rng(seed).choice(10000, 500, replace=False)
        hits = 0
        for ii in idx:
            x = m.te_flat[ii]; y = m.tel[ii]
            for _ in range(m.steps):
                m.spiking_step(x, m.y_onehot[y], learn=False)
            hits += int(np.argmax(m.f_est) == y)
            zx = np.zeros(784); zy = np.zeros(m.NOUT)
            for _ in range(100):
                m.spiking_step(zx, zy, learn=False)
        res.append(hits / 500)
    print(f"{ckpt}  seeds[123,1,2,3,4]: {['%.3f' % r for r in res]}  mean={np.mean(res):.3f} std={np.std(res):.3f}")