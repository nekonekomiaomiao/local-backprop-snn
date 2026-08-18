# exp4 batch (2026-08-15 late night — 0.8 target achieved)

Protocol: **per-sample hard-reset spiking online training** (reset u / r_est / f_est / E before each sample; within the paper's SDE framework, physically implementable) + local in-place updates. Pure spiking, random initialization, no weight injection.

| dir | config | final n=1000 | independent eval (reset, k=0.2) single / long-window / counting |
|---|---|---|---|
| reset_a15_20k | k=0.2, a=1.5e-8, N=20k, seed=0 | 0.767 | 0.776 / 0.810 / 0.812 |
| reset_a10_30k | k=0.2, a=1e-8, N=30k, seed=0 | 0.778 | 0.789 / 0.815 / 0.811 |
| **reset_a15_cont** | 20k@1.5e-8 continued 20k@7.5e-9 (40k total) | **0.824** | **0.817 / 0.848 / 0.850** |

- **reset_a15_cont = the 0.8-target model** (all three readouts >= 0.8).
- Each dir: `run_log.txt` (full log), `run_err.txt`, `mnist_checkpoint.npz` (every 2000 samples + final), `mnist_result.png`.

*Related LIF+ISI (fully physical, no hard reset) main-result runs are also under this batch: see docs/MNIST_RESULTS_CN.md section 6.5 and docs/EXPERIMENT_PROGRESS.md.*
