# LIF+ISI Parameter Sweep: Parameter × Metric Relationships (completed 2026-08-16 late night)

> 📖 **Terminology note**: The custom/project-specific terms used throughout this document (recalibration, eligibility trace, cross-sample state pollution, frozen evaluation, multi-seed acceptance, mean-field, gradient alignment, collapse, etc.) are briefly explained at their first occurrence; the authoritative definitions are consolidated in [docs/GLOSSARY.md](GLOSSARY.md).


> 16 groups × N=1000 (MNIST trained from scratch, sample-by-sample online local learning), uniform measurement: loss plateau (mean±std of last 100), frozen test acc (n=500),
> gradient alignment (per-sample cos of ΔP with analytic gradient -g, per-layer + ALL), corr(e,d) (covariance correlation of eligibility trace E with local error d, last-5 samples),
> updateSNR / expected update efficiency expEff / sign consistency signCons.
> Raw data: `mnist_lif_table_results.csv`; per-run logs: `logs/table_lif_*.log`; evolution trajectories: `traj_*.csv` (per 200-sample window).
> Generators: `mnist_lif_table.py` / `analyze_lif_sweep.py` / `analyze_lif_traj.py`.

## Headline Findings

1. **TARGET is non-monotonic, peaking at the recalibrated value 5000**: frozen acc 0.404 (T3000) → **0.648 (T5000)** → 0.372 (T7000). corr_W3 rises monotonically with TARGET (+0.015→+0.026→+0.037).
2. **Recalibrating the κ=1.0 inhibition pool simultaneously "lowers" gradient alignment yet raises acc (counterintuitive)**: κ=0.2/0.5/1.0 → align_ALL 0.764/0.468/0.238 (monotonic down), frozen acc 0.414/0.598/0.648 (monotonic up). **Per-sample gradient alignment high ≠ good end-to-end performance** — the κ=1.0 global inhibition pool fixes the output layer's push/pull imbalance (the metric-side evidence for the paper's recalibration mechanism).
3. **ISI is double-edged: 50 is optimal, both 0 and 100 are bad**: ISI=0's loss plateau is only 0.0136 (cross-sample state pollution makes the training loss artificially low, but frozen acc is only 0.540 and alignment 0.166); ISI=50 → acc 0.648; ISI=100 → acc collapses to 0.210 (silence too long, readout state becomes decoupled from the training distribution).
4. **Cross-protocol "alignment paradox" quantified**: IF no leakage +0.281 / hard reset +0.794 / LIF uncalibrated +0.798 / LIF recalibrated +0.238 — **the two protocols with the highest alignment have the lowest acc (0.134/0.132), while recalibration with the lowest alignment has the highest acc (0.648)**. All such data are N=1000 early-training measurements (full 14k-20k long runs: recalibrated 0.877±0.007, uncalibrated 0.501, hard reset 0.824).
5. **Cross-metric correlations (16 runs)**: align_ALL vs frozen_acc **ρ=-0.553** (negative correlation — the paradox holds across the full parameter space); align_std vs frozen_acc ρ=+0.482 (more alignment noise instead correlates with higher acc — corresponds to the κ=1.0 inhibition-pool low-rate high-randomness operating point); snr_W3 vs align_ALL ρ=+0.553; sign_W1 vs frozen_acc ρ=-0.285 (Pearson -0.637).
6. **Both α and τ_e are non-monotonic at the 1k measurement**: α=1.5e-8 is optimal (0.648 > 3e-8's 0.564 > 7.5e-9's 0.516); τ_e=0.2 is better than 0.1 (0.648 vs 0.570). τ_m=0.2 is a disaster (0.142 — leak too fast, eligibility trace can't hold), τ_m=1.0 slightly better (0.696).

## Caveats

- All acc/loss are **N=1000 early-training snapshots** (single seed, n=500 frozen evaluation), used to compare parameter × metric relationships; they are not the long-run end values (see the acceptance table in EXPERIMENT_PROGRESS.md).
- In the four-protocol comparison, "protocol × calibration" is naturally confounded: uncalibrated/IF/reset use TARGET=1000/κ=0.2, while recalibrated uses TARGET=5000/κ=1.0 — this is exactly what "recalibration" means.
- The corr column of the trajectory files (traj_*.csv) is valid for the whole run only for the runs launched after 21:47 (isi100, batch-2, and the 3 re-run groups); for the two earlier runs (isi0, tm1p0) the window corr has values only in the final window and its normalization is too small.
- The sweep tables use their own ISI for frozen evaluation (train/eval under the same measurement); the formal acceptance protocol (eval_multiseed) fixes ISI=100 for evaluation.

---

## Same-measurement four-protocol comparison

| Protocol | loss | frozen_acc | align_ALL | corrW1/W2/W3 | biasW1 | snrW3 | effW3 | signW3 |
|---|---|---|---|---|---|---|---|---|
| IF no leakage (0.348 reference model) | 0.0547 | 0.1340 | +0.281 | +0.004/+0.001/+0.005 | 3.686 | 0.156 | 0.208 | 0.261 |
| Hard reset (0.824 reference model) | 0.0505 | 0.1840 | +0.794 | -0.005/-0.001/-0.001 | 1.426 | 0.111 | 0.493 | 0.676 |
| LIF+ISI uncalibrated (0.501 model) | 0.0490 | 0.1320 | +0.798 | +0.002/+0.005/+0.010 | 0.411 | 0.109 | 0.505 | 0.648 |
| LIF+ISI recalibrated baseline (0.877 model) | 0.0494 | 0.6480 | +0.238 | -0.003/+0.010/+0.026 | 0.253 | 0.029 | 0.009 | 0.491 |

## Single-variable response (RESET=0, all other parameters == baseline)

### TARGET  (3 runs)

| TARGET | loss_plateau | loss_std | frozen_acc | align_ALL | corrW1/W2/W3 | biasW1/W2/W3 | snrW1/W2/W3 | effW1/W2/W3 | signW1/W2/W3 |
|---|---|---|---|---|---|---|---|---|---|
| 3000.0 | 0.0490 | 0.0019 | 0.4040 | +0.255 | -0.008/+0.018/+0.015 | 0.623/188.113/0.613 | 0.072/0.065/0.051 | 0.009/0.020/0.018 | 0.337/0.457/0.406 |
| 5000.0 | 0.0494 | 0.0012 | 0.6480 | +0.238 | -0.003/+0.010/+0.026 | 0.253/1.743/0.479 | 0.131/0.040/0.029 | 0.017/0.011/0.009 | 0.163/0.497/0.491 |
| 7000.0 | 0.0489 | 0.0019 | 0.3720 | +0.281 | -0.001/+0.014/+0.037 | 0.309/0.617/0.693 | 0.202/0.033/0.022 | 0.025/0.008/0.006 | 0.173/0.555/0.487 |

### KAPPA  (3 runs)

| KAPPA | loss_plateau | loss_std | frozen_acc | align_ALL | corrW1/W2/W3 | biasW1/W2/W3 | snrW1/W2/W3 | effW1/W2/W3 | signW1/W2/W3 |
|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.0493 | 0.0013 | 0.4140 | +0.764 | -0.005/+0.009/+0.033 | 0.340/1.398/0.366 | 0.115/0.044/0.035 | 0.150/0.104/0.073 | 0.221/0.628/0.594 |
| 0.5 | 0.0493 | 0.0012 | 0.5980 | +0.468 | +0.004/+0.011/+0.037 | 0.290/2.336/0.578 | 0.123/0.042/0.033 | 0.047/0.031/0.023 | 0.250/0.563/0.556 |
| 1.0 | 0.0494 | 0.0012 | 0.6480 | +0.238 | -0.003/+0.010/+0.026 | 0.253/1.743/0.479 | 0.131/0.040/0.029 | 0.017/0.011/0.009 | 0.163/0.497/0.491 |

### TAU_M  (3 runs)

| TAU_M | loss_plateau | loss_std | frozen_acc | align_ALL | corrW1/W2/W3 | biasW1/W2/W3 | snrW1/W2/W3 | effW1/W2/W3 | signW1/W2/W3 |
|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.0493 | 0.0012 | 0.1420 | +0.251 | -0.003/+0.010/+0.022 | 0.213/3.796/0.340 | 0.138/0.039/0.029 | 0.021/0.012/0.009 | 0.135/0.520/0.469 |
| 0.5 | 0.0494 | 0.0012 | 0.6480 | +0.238 | -0.003/+0.010/+0.026 | 0.253/1.743/0.479 | 0.131/0.040/0.029 | 0.017/0.011/0.009 | 0.163/0.497/0.491 |
| 1.0 | 0.0496 | 0.0012 | 0.6960 | +0.206 | -0.007/+0.012/+0.029 | 0.725/29.520/0.608 | 0.132/0.044/0.030 | 0.015/0.011/0.008 | 0.202/0.500/0.497 |

### ISI  (3 runs)

| ISI | loss_plateau | loss_std | frozen_acc | align_ALL | corrW1/W2/W3 | biasW1/W2/W3 | snrW1/W2/W3 | effW1/W2/W3 | signW1/W2/W3 |
|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.0136 | 0.0156 | 0.5400 | +0.166 | -0.002/+0.009/+0.012 | 7.049/0.907/0.553 | 0.117/0.038/0.022 | 0.013/0.009/0.007 | 0.144/0.502/0.381 |
| 50.0 | 0.0494 | 0.0012 | 0.6480 | +0.238 | -0.003/+0.010/+0.026 | 0.253/1.743/0.479 | 0.131/0.040/0.029 | 0.017/0.011/0.009 | 0.163/0.497/0.491 |
| 100.0 | 0.0492 | 0.0014 | 0.2100 | +0.238 | -0.006/+0.010/+0.022 | 0.463/2.678/0.542 | 0.132/0.040/0.030 | 0.018/0.011/0.009 | 0.173/0.507/0.491 |

### TAU_E  (2 runs)

| TAU_E | loss_plateau | loss_std | frozen_acc | align_ALL | corrW1/W2/W3 | biasW1/W2/W3 | snrW1/W2/W3 | effW1/W2/W3 | signW1/W2/W3 |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.0495 | 0.0013 | 0.5700 | +0.240 | -0.006/+0.015/+0.021 | 0.463/1.160/0.512 | 0.153/0.056/0.044 | 0.020/0.017/0.015 | 0.183/0.483/0.447 |
| 0.2 | 0.0494 | 0.0012 | 0.6480 | +0.238 | -0.003/+0.010/+0.026 | 0.253/1.743/0.479 | 0.131/0.040/0.029 | 0.017/0.011/0.009 | 0.163/0.497/0.491 |

### alpha  (3 runs)

| alpha | loss_plateau | loss_std | frozen_acc | align_ALL | corrW1/W2/W3 | biasW1/W2/W3 | snrW1/W2/W3 | effW1/W2/W3 | signW1/W2/W3 |
|---|---|---|---|---|---|---|---|---|---|
| 7.5e-09 | 0.0492 | 0.0015 | 0.5160 | +0.245 | -0.009/+0.014/+0.019 | 0.804/1.609/0.570 | 0.156/0.060/0.047 | 0.018/0.015/0.015 | 0.163/0.493/0.416 |
| 1.5e-08 | 0.0494 | 0.0012 | 0.6480 | +0.238 | -0.003/+0.010/+0.026 | 0.253/1.743/0.479 | 0.131/0.040/0.029 | 0.017/0.011/0.009 | 0.163/0.497/0.491 |
| 3e-08 | 0.0495 | 0.0010 | 0.5640 | +0.278 | -0.001/+0.008/+0.035 | 0.092/1.755/0.777 | 0.114/0.032/0.020 | 0.015/0.008/0.005 | 0.202/0.528/0.500 |


## Cross-metric correlations (all runs)

| X | Y | Spearman ρ | Pearson r | n |
|---|---|---|---|---|
| align_all | loss_plateau | +0.162 | +0.225 | 16 |
| align_all | frozen_acc | -0.553 | -0.427 | 16 |
| align_all | corr_W1 | +0.474 | +0.229 | 16 |
| align_all | corr_W3 | -0.091 | -0.257 | 16 |
| corr_W1 | loss_plateau | +0.100 | -0.013 | 16 |
| corr_W3 | loss_plateau | -0.062 | +0.154 | 16 |
| sign_W3 | loss_plateau | +0.250 | +0.196 | 16 |
| snr_W3 | loss_plateau | +0.309 | +0.292 | 16 |
| eff_W3 | loss_plateau | +0.265 | +0.170 | 16 |
| bias_W1 | loss_plateau | -0.015 | -0.801 | 16 |
| corr_W3 | align_all | -0.091 | -0.257 | 16 |
| snr_W3 | align_all | +0.553 | +0.465 | 16 |
| align_std | loss_plateau | -0.050 | -0.119 | 16 |
| align_std | frozen_acc | +0.482 | +0.560 | 16 |
| snr_W1 | loss_plateau | +0.012 | +0.124 | 16 |
| sign_W1 | frozen_acc | -0.285 | -0.637 | 16 |

## Parameter vs key metric (relative change from baseline)

| Parameter combination | Δloss | Δacc | Δalign | ΔcorrW3 | ΔsnrW3 | ΔeffW3 | ΔsignW3 |
|---|---|---|---|---|---|---|---|
| T3000/K1.0/TM0.5/ISI50.0 | -0.0004 | -0.2440 | +0.017 | -0.011 | +0.022 | +0.009 | -0.085 |
| T5000/K0.2/TM0.5/ISI50.0 | -0.0001 | -0.2340 | +0.526 | +0.007 | +0.006 | +0.064 | +0.103 |
| T7000/K1.0/TM0.5/ISI50.0 | -0.0005 | -0.2760 | +0.043 | +0.011 | -0.007 | -0.003 | -0.004 |
| T5000/K1.0/TM0.5/ISI0.0 | -0.0358 | -0.1080 | -0.072 | -0.014 | -0.007 | -0.002 | -0.110 |
| T5000/K1.0/TM1.0/ISI50.0 | +0.0002 | +0.0480 | -0.032 | +0.003 | +0.001 | -0.001 | +0.006 |
| T5000/K1.0/TM0.5/ISI100.0 | -0.0002 | -0.4380 | +0.000 | -0.004 | +0.001 | +0.000 | +0.000 |
| T1000/K0.2/TM0.0/ISI0.0/RESET | +0.0011 | -0.4640 | +0.556 | -0.027 | +0.082 | +0.484 | +0.185 |
| T1000/K0.2/TM0.0/ISI0.0 | +0.0053 | -0.5140 | +0.043 | -0.021 | +0.127 | +0.199 | -0.230 |
| T5000/K1.0/TM1.0/ISI50.0 | -0.0002 | -0.0920 | -0.018 | +0.006 | -0.011 | -0.006 | -0.041 |
| T5000/K1.0/TM0.5/ISI50.0 | +0.0001 | -0.0780 | +0.002 | -0.005 | +0.015 | +0.006 | -0.044 |
| T5000/K1.0/TM0.5/ISI50.0 | -0.0002 | -0.1320 | +0.007 | -0.007 | +0.018 | +0.006 | -0.075 |
| T5000/K1.0/TM0.5/ISI50.0 | +0.0001 | -0.0840 | +0.040 | +0.009 | -0.009 | -0.004 | +0.009 |
| T1000/K0.2/TM0.5/ISI50.0 | -0.0004 | -0.5160 | +0.560 | -0.016 | +0.080 | +0.496 | +0.157 |
| T5000/K0.5/TM0.5/ISI50.0 | -0.0001 | -0.0500 | +0.230 | +0.011 | +0.004 | +0.014 | +0.065 |
| T5000/K1.0/TM0.2/ISI50.0 | -0.0001 | -0.5060 | +0.013 | -0.004 | +0.000 | +0.000 | -0.022 |

Raw full data: `mnist_lif_table_results.csv`; per-run logs `logs/table_lif_*.log`
# LIF Sweep Trajectory Evolution Summary (traj_*.csv)

| run | steps | loss@last | acc@last | alAll@first→@last (Δ) | alAll_std@last | corrW1/W2/W3@last | biasW1@last |
|---|---|---|---|---|---|---|---|
| T1000/K0.2/TM0.0/ISI0/TE0.2/RS0 | 1000 | 0.0547 | 0.070 | +0.404 → +0.263 → +0.225 (Δ-0.179) | 0.241 | +0.004/+0.001/+0.008 | 1.086 |
| T1000/K0.2/TM0.0/ISI0/TE0.2/RS1 | 1000 | 0.0505 | 0.160 | +0.830 → +0.783 → +0.778 (Δ-0.052) | 0.150 | +0.001/-0.000/+0.003 | 1.602 |
| T1000/K0.2/TM0.5/ISI50/TE0.2/RS0 | 1000 | 0.0490 | 0.110 | +0.826 → +0.791 → +0.785 (Δ-0.041) | 0.142 | +0.002/+0.002/+0.009 | 19.702 |
| T5000/K0.5/TM0.5/ISI50/TE0.2/RS0 | 1000 | 0.0493 | 0.750 | +0.559 → +0.401 → +0.504 (Δ-0.055) | 0.343 | -0.004/+0.005/+0.025 | 127.854 |
| T5000/K1.0/TM0.2/ISI50/TE0.2/RS0 | 1000 | 0.0493 | 0.220 | +0.326 → +0.201 → +0.299 (Δ-0.028) | 0.403 | -0.003/+0.004/+0.015 | 19.465 |
| T5000/K1.0/TM0.5/ISI0/TE0.2/RS0 | 1000 | 0.0136 | 0.880 | +0.274 → +0.130 → +0.121 (Δ-0.154) | 0.367 | -0.001/+0.017/-0.008 | 242.973 |
| T5000/K1.0/TM0.5/ISI100/TE0.2/RS0 | 1000 | 0.0493 | 0.250 | +0.337 → +0.177 → +0.260 (Δ-0.077) | 0.408 | -0.003/+0.005/+0.016 | 39.259 |
| T5000/K1.0/TM0.5/ISI50/TE0.1/RS0 | 1000 | 0.0495 | 0.630 | +0.504 → +0.144 → +0.197 (Δ-0.307) | 0.338 | -0.002/+0.010/+0.021 | 2.161 |
| T5000/K1.0/TM0.5/ISI50/TE0.2/RS0 | 1000 | 0.0495 | 0.930 | +0.270 → +0.258 → +0.332 (Δ+0.062) | 0.453 | -0.004/+0.006/+0.023 | 20.801 |
| T5000/K1.0/TM1.0/ISI50/TE0.2/RS0 | 1000 | 0.0496 | 0.930 | +0.328 → +0.150 → +0.203 (Δ-0.124) | 0.400 | -0.015/+0.011/-0.015 | 68.009 |
| T5000/K1.0/TM1.0/ISI50/TE0.5/RS0 | 1000 | 0.0493 | 0.980 | +0.251 → +0.195 → +0.238 (Δ-0.012) | 0.431 | -0.003/+0.008/+0.022 | 5.826 |

Note: traj has one row per 200 samples; alAll=200-window mean alignment (cos with descent direction), corr=within-window (E,d) correlation.
Alignment decreases with training → in local learning alignment is high early and noise-dominated late; the relationship between corr@last and frozen acc is in the analysis tables.
