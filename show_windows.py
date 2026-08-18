import numpy as np, glob, os

for f in sorted(glob.glob("sweep_out/cfg_scaled_*.npz")):
    z = np.load(f)
    R, dT = int(z["R"]), float(z["dT"])
    wc = z["w_corr"]
    w1 = " ".join(f"{v:+.3f}" for v in wc[:, 0])
    w2 = " ".join(f"{v:+.3f}" for v in wc[:, 1])
    w3 = " ".join(f"{v:+.3f}" for v in wc[:, 2])
    print(f"R={R:4d} dT={dT:3.1f} | W1 [{w1}] | W2 [{w2}] | W3 [{w3}]")
