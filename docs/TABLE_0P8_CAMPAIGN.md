# 0.8 Target Sprint: Parameter-Combination Comparison Table (starting 2026-08-15 evening)

> 📖 **Terminology**: The project-specific/self-coined terms used here (recalibration, eligibility trace, cross-sample state pollution, frozen evaluation, multi-seed acceptance, mean-field, gradient alignment, collapse, etc.) are briefly explained at their first occurrence; authoritative definitions are consolidated in [docs/GLOSSARY.md](GLOSSARY.md).


> Data source: `mnist_0p8_campaign.csv` (generation script `make_campaign_table.py`); mean-field acc parsed from run_log.

## 1. Mean-field GD training (LR=1e-8, seed=0, N=300k, checkpoint saved every 25k)

| κ | 25k | 50k | 75k | 100k | 150k | 200k | 250k | 300k(final n=1000) |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.883 | 0.917 | 0.920 | 0.940 | 0.957 | 0.957 | 0.967 | **0.965** |
| 0.1 | 0.907 | 0.923 | 0.920 | 0.933 | 0.933 | 0.947 | 0.930 | **0.965** |
| 0.2 | 0.897 | 0.907 | 0.920 | 0.920 | 0.940 | 0.933 | 0.943 | **0.965** |
| 0.4 | 0.890 | 0.907 | 0.923 | 0.923 | 0.940 | 0.933 | 0.940 | **0.961** |

## 2. Mean-field weights → spiking readout (capability proof: weight quality is not the bottleneck)

> Mean-field 300k weights (noise-free acc 0.96) are fed into the spiking network. Without reset (cross-sample state pollution) acc is only 0.12-0.20; **with per-sample reset it is 0.93-0.95** — proving that the architecture + readout protocol has sufficient capacity; the bottleneck lies in the training protocol and state pollution.

| κ ckpt | spiking single (no reset) | spiking single (reset) | long-window (reset) | counting (reset) | mean-field control |
|---|---|---|---|---|---|
| meanfield_ckpts/k0 | no-reset 0.115 | 0.942 | 0.960 | 0.961 | 0.961 |
| meanfield_ckpts/k0p1 | no-reset 0.164 | 0.947 | 0.955 | 0.959 | 0.962 |
| meanfield_ckpts/k0p2 | no-reset 0.203 | 0.947 | 0.956 | 0.954 | 0.958 |
| meanfield_ckpts/k0p4 | no-reset 0.175 | 0.953 | 0.955 | 0.954 | 0.954 |

## 3. Reset-protocol spiking training (head-on: pure spiking, random init, online local learning)

| Directory | κ | α | N(samples) | final n=1000 | independent eval single/long-window/counting |
|---|---|---|---|---|---|
| exp4/reset_a15_20k | 0.2 | 1.5e-8 | 20000 | **0.767** | 0.776/0.810/0.812 |
| exp4/reset_a10_30k | 0.2 | 1e-8 | 30000 | **0.778** | 0.789/0.815/0.811 |
| exp4/reset_a15_cont | 0.2 | 7.5e-9 | 20000 | **0.824** | 0.817/0.848/0.850 |

## 4. Historical baseline (before the 0.8 target, no-reset protocol)

| Configuration | frozen acc |
|---|---|
| historical best spiking κ=0.2 α=1.5e-8 20k; reset eval 0.265/0.281/0.283 | 0.265 (reset) |
| seed2 reproduction; reset eval 0.195/0.200/0.199 | 0.195 (reset) |
| seed1 reproduction (n=1000) | 0.334 |
| α=1e-8 30k (n=1000) | 0.330 |
