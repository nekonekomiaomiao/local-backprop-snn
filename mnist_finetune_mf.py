"""Spiking fine-tune starting from a mean-field (or any) checkpoint.
Usage: python3 mnist_finetune_mf.py <init_ckpt.npz> <ALPHA> <N> [SEED] [KAPPA] [SAMPLE_T] [RESET] [TARGET] [ISI] [TAU_M]
KAPPA defaults to the value stored in the checkpoint (or 0.2 if absent).
TARGET defaults to the checkpoint value (or 1000 if absent). ISI/TAU_M default 0 (IF protocol).
"""
import sys
import numpy as np

CKPT = sys.argv[1]
ALPHA = float(sys.argv[2])
N = int(sys.argv[3])
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
SAMPLE_T = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
RESET = int(sys.argv[7]) if len(sys.argv) > 7 else 0
TARGET_ARG = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0
ISI = int(sys.argv[9]) if len(sys.argv) > 9 else 0
TAU_M = float(sys.argv[10]) if len(sys.argv) > 10 else 0.0

z = np.load(CKPT)
if len(sys.argv) > 5:
    KAPPA = float(sys.argv[5])
else:
    KAPPA = float(z["KAPPA"]) if "KAPPA" in z else 0.2
if TARGET_ARG <= 0:
    TARGET_ARG = float(z["TARGET"]) if "TARGET" in z else 1000.0

sys.argv = ["mnist_shared.py", str(SEED), str(N), str(SAMPLE_T), "200", str(ALPHA), "3000", "30",
            str(TARGET_ARG), "1000", "0.02", str(KAPPA), "0", str(RESET), str(ISI), str(TAU_M)]
import mnist_shared as m

m.P[:] = z["P"]
if "SIGN" in z:
    m.SIGN[:] = z["SIGN"]
print(f"fine-tune from {CKPT}: P inited (mean={m.P.mean():.4f}), KAPPA={KAPPA}, ALPHA={ALPHA}, N={N}, "
      f"seed={SEED}, TARGET={TARGET_ARG}, ISI={ISI}, TAU_M={TAU_M}", flush=True)
m.run_training()
