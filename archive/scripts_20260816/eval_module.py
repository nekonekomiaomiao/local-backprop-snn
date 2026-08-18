import sys, numpy as np
ckpt = sys.argv[1]
sys.argv = ["mnist_shared.py", "0", "20000", "1.0", "200", "1.5e-8", "3000", "30", "5000", "1000",
            "0.02", "1.0", "0", "0", "50", "0.5"]
import mnist_shared as m
z = np.load(ckpt)
m.P[:] = z["P"]
if "SIGN" in z:
    m.SIGN[:] = z["SIGN"]
a1 = m.evaluate(200)
a2 = m.evaluate(1000)
print(f"{ckpt}  module_evaluate n=200={a1:.4f}  n=1000={a2:.4f}")