# meanfield_ckpts (2026-08-15 evening, started for the 0.8 target; stopped on user request)

Goal: train noise-free mean-field GD weights, save checkpoints, then run spiking frozen evaluation to test whether "mean-field warm-up + spiking online fine-tuning" can reach 0.8.

| dir | kappa | LR | N planned | samples at stop | test acc at stop |
|---|---|---|---|---|---|
| k0 | 0.0 | 1e-8 | 300000 | ~92.5k | 0.927 |
| k0p1 | 0.1 | 1e-8 | 300000 | ~82.5k | 0.930 |
| k0p2 | 0.2 | 1e-8 | 300000 | ~80k | 0.917 |
| k0p4 | 0.4 | 1e-8 | 300000 | ~77.5k | 0.927 |

**Important**:
- Current `gd_meanfield_shared.py` saves `meanfield_checkpoint.npz` only after training finishes, so these 4 stopped runs have **no checkpoint**, only `run_log.txt`.
- To resume, just rerun; to save mid-run, first modify `gd_meanfield_shared.py` to save periodically (suggest every 25k).
