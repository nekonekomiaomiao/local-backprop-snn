# exp3 batch (2026-08-15)

| dir | config | result |
|---|---|---|
| k0p2_a15_s2_20k | shared, seed=2, N=20000, dT=1, R=200, a=1.5e-8, FDA=3000, BIAS=30, TARGET=1000, DT=0.02, KAPPA=0.2 | final frozen acc **0.272** (n=500); periodic peak 0.305@16k |
| r400_a3p75_3k | shared, seed=0, N=3000, R=400, a=3.75e-9, FDA=6000, BIAS=60, TARGET=1000, DT=0.01, KAPPA=0.2 | final **0.122**; R hypothesis rejected |
| r800_a0p9375_2k | shared, seed=0, N=2000, R=800, a=9.375e-10, FDA=12000, BIAS=120, TARGET=1000, DT=0.005, KAPPA=0.2 | output deadlock @1000 total=0, terminated |
| r800_k0_a3p75_2k | shared, seed=0, N=2000, R=800, a=3.75e-9, FDA=12000, BIAS=120, TARGET=1000, DT=0.005, KAPPA=0 | final **0.106**; R hypothesis rejected |

Each dir contains: `run_log.txt` (full log), `run_err.txt`, `mnist_checkpoint.npz` (every 2000 samples + final), `mnist_result.png`.
