# MNIST Experiment Record — Probabilistic-Synapse SNN (English translation)

> 📖 **Terminology note**: Self-coined/project-specific terms used here (recalibration, eligibility trace, cross-sample state pollution, frozen evaluation, multi-seed acceptance, mean-field, gradient alignment, overtraining collapse, etc.) are briefly explained at first occurrence; authoritative definitions are consolidated in [GLOSSARY.md](GLOSSARY.md).


> Date: starting 2026-08-13, updated through 2026-08-15; main scripts: `mnist_shallow.py` (stride/avg/max three downsampling modes), `mnist_conv_snn.py` (dual-convolution version)
> Data: downloaded via PyTorch official mirror ossci-datasets (yann.lecun.com is not directly reachable), `mnist_data/` + `mnist_loader.py`

---

## 1. Topology decision (why not "true SOTA") and the correction on weight sharing

The conflict points between a true-SOTA SNN topology (Spiking ResNet/VGG + surrogate gradient, MNIST ≈99.5%) and this constructive framework need to be revisited:
- **Weight sharing (convolution) does not violate per-synapse locality (2026-08-13 user correction)**: biology implements **temporal reuse** of the same set of receptive-field weights by "changing the field of view" (eye movements/saccades) — the same physical synapse processes different spatial positions at different moments, and each scanned position provides a local signal (that position's eligibility trace and local error). The weight update = **sum** of the local signals across positions (= standard convolution's gradient). Shared weights are therefore still "each synapse updated using only its local signal";
- The surrogate gradient solves the numerical problem of spike non-differentiability, which is not on the same level as this construction (a physical local weight-update mechanism), so it is still not adopted.

**Final topology (two generations)**:
```
v1 (locally-connected version, mnist_shallow.py, stride/avg/max three modes):
  784 → local conv5x5x4 → downsampling → FC32 → 10   （74k synapses / 37k parameters)

v2 (shared-convolution version, mnist_shared.py, adopting the user correction):
  784 → standard CONV5x5x4 stride2 (shared kernel, only 104 parameters) → FC32 → 10
  (18,898 parameters, 33,770 edges; shared-parameter update = sum of local signals per position in the group)
```

Three downsampling modes (the "pooling" discussed with the user):
| Mode | Implementation | Backward factor | Note |
|---|---|---|---|
| **avg** | pooling IF neuron θ=4: output rate = Σf/θ = average | δ/4 | i.e. the user's "average-pooling tuned threshold" (4 inputs → threshold 4) |
| **max** | lateral-inhibition WTA: each block takes this step's max firing count, θ=1 | δ given only to the winner (no 1/θ) | i.e. the user's "lateral inhibition = max pooling"; no backward attenuation and feature-sparse |
| **stride** | stride-2 convolution (θ=1, no pooling layer) | no attenuation | same downsampling, backward δ 4× stronger |

**Advantage of shared convolution (computed)**: each shared parameter gets evidence from 144 positions → per-parameter SNR ∝ √144 = 12×, learns fast and stably.

## 2. Verified correctness (all passed)

1. **Gradient direction (finite difference 12/12)**: `verify_mnist_shallow.py` picks 12 high-gradient synapses for each of the three modes; the analytical gradient (including the s/θ factor, i.e. ∂L/∂P) and the numerical difference sign match exactly;
2. **Forward equivalence**: spike-layer rates vs mean-field rates (F layer 20.0 vs 24.6, FC 183 vs 162, output 5932 vs 4770 Hz, same scale);
3. **Per-sample alignment**: actual update vs analytical gradient cos = 0.68~0.91, magnitude ratio 0.42~1.04 — **the mechanism itself works**;
4. **Mean-field GD control** (no noise): lr=1e-6, 4000 samples → test 0.42; lr=1e-8, 2000 samples → 0.23 — the architecture is learnable; the cost of the spiking system is noise loss.

## 3. Debugging history (fixed bugs)

1. **Backward-graph bincount absolute-index bug**: the bincount for δ backprop used absolute neuron indices, so `[:NP2]` sliced into bins 0..191 rather than the P2 neurons (5874..6065) — W1/W2's δ were all 0 (manifesting as all-zero W1/W2 gradients). Fix: subtract the layer base offset (`PRE3−OFF_P2` etc.);
2. **gate broadcast bug**: `gateF * d_f[POST1−OFF_F]` has dimensions (576,)·(14976,) — illegal/misaligned; must look up per-synapse `gateF[POST1−OFF_F]`;
3. **Training-script import side effect**: the main loop lacked `__main__` protection, so importing triggered a full training run (profiling dragged for 30+ minutes); added run_training() protection;
4. **Validation-script object misplacement**: ∂L/∂w vs ∂L/∂P differ by the s/θ factor, once misjudged as "gradient sign mismatch".

## 4. Key findings on training behavior (usable for the paper discussion)

### 4.1 Training acc inflated (in-sample transient adaptation)
The training rolling acc can reach 0.4-0.5, but **frozen evaluation is only 0.10-0.16**: learning acts on the current sample in real time during its presentation (zero latency), and the argmax is read at the sample's end — the weights briefly adapt to the current sample's target within 50 steps. **Evaluation must freeze learning** (in this project frozen evaluation is the real metric).

### 4.2 Per-sample update magnitude and noise
- The theoretical per-sample update is lr·g (lr = α·τ_e·θ·ΔT); the measured accumulation is only 12-19% (over 120 samples) — noise random walk cancels it out;
- Per-sample SNR = |δ|√ΔT/√f_da: output layer ~2-4 (at f_da=3000-8000); deeper layers have larger δ from fan-in summation (~1300 Hz), giving higher SNR;
- **Carrier noise dominates**: f_da must be ≥ max δ (deep-layer fan-in summation can reach ~3000 Hz), so the carrier noise √f_da of the output layer cannot be lowered — the output layer is the learning bottleneck;
- **Magnitude of α**: MNIST firing rates are 5-20× higher than XOR → gradient magnitude 25-400× larger. α=2e-6 (XOR's 2.5e-6) saturates the boundary outright and weights jump wildly between 0/1 (mechanism of loss exploding to 19-63: weight saturation → output burst → δ > FDA → f_err truncated to 0 → negative-feedback deadlock). **α must be lowered to ~1e-7 magnitude**;
- FDA-truncation deadlock: when δ exceeds FDA, f_err=0, updates freeze, and weights get stuck at a saturated position — FDA must keep enough margin.

### 4.3 Cross-sample transient pollution
At sample boundaries E (eligibility trace) and f_est carry residue from the previous input (τ_e/τ_f magnitude); ΔT=1s is only 5τ_e, so the first 40% of each sample is polluted. Measured: continuous mode per-sample cos=0.08 vs reset mode 0.18 (halved). Paper §8.4 criterion ΔT ≫ transient; at ΔT=4 the magnitude ratio does not improve (0.19 vs 0.19) — the transient is not the main cause.

### 4.4 Covariance bias (direct evidence for paper §8.5) + control experiment on the user hypothesis

**Baseline (R=200, ΔT=1, FDA=3000, 40 samples)**:
| Layer | corr(e,d) | |cov|/|E[e]E[d]| |
|---|---|---|---|
| W1 | +0.016 | 1.20 |
| W2 | +0.004 | 0.07 |
| W3 | +0.036 | 0.28 |

corr is small but the **bias share is large**: the error-signal mean E[d]→0 (especially deeper), amplifying |cov|/|E[e]E[d]| — XOR is 2-3%, MNIST reaches 28-120%.

**User hypothesis (2026-08-13)**: the deeper-layer high covariance arises because the network is too deep, making the effective forward-backward delay too large; one should raise the carrier frequency or sample duration 10×, expecting the covariance to drop to ~3%. Control experiment:

| Config | corr W1 | corr W2 | corr W3 |
|---|---|---|---|
| R=200, ΔT=1 (baseline) | +0.016 | +0.004 | +0.036 |
| **R=2000, ΔT=1** (carrier ×10) | −0.033 | −0.039 | +0.061 |
| **R=200, ΔT=10** (sample ×10) | **+0.004** | **+0.000** | **+0.025** |

**Conclusion: hypothesis partially confirmed** — ΔT×10 brings corr(e,d) down to 0.000~0.025 (≈3% magnitude ✓); carrier R×10 has no effect (the τ_e/τ_f time window is unchanged, so the window-overlap ratio is unchanged). Mechanism: the covariance comes from overlap of the in-sample e window and the f_est window (zero-latency implementation); longer samples dilute the boundary transient and window overlap; equivalent to shrinking the "relative forward-backward delay". **The training config therefore adopts ΔT=10.**

### 4.5 Effect of temporal discretization granularity (dt) on covariance — user hypothesis #2 (tested 2026-08-13)

**User framework**: carrier frequency = 1/dt; the simulation should ensure "max spike frequency = at most one spike per step" (λ·dt ≤ 1); the minimum forward-backward delay = dt×number of layers; what actually matters is the number of time steps a sample experiences = ΔT/dt. Raising the carrier frequency accordingly should shrink the effective delay → covariance drops.

**Executed precisely per the user's prescription** (same shared-conv architecture, ΔT=1s to isolate dt):

| dt | steps/sample | λ·dt(input peak) | corr W1 | corr W2 | corr W3 |
|---|---|---|---|---|---|
| 20 ms | 50 | 4.0 (multi-spike per step ✗) | +0.015 | +0.003 | +0.013 |
| 2 ms | 500 | 0.4 (≤1 spike per step ✓) | +0.016 | +0.004 | +0.016 |
| 1 ms | 1000 | 0.2 (✓) | +0.015 | +0.003 | +0.015 |
| 1 ms | 10000 (ΔT=10s) | 0.2 (✓) | +0.013 | +0.012 | +0.037 |

**Result: dt has no effect on corr(e,d)** (20ms→1ms, steps ×20, completely unchanged even after λ·dt≤1 is satisfied). Reason: the covariance is determined by the shared rate fluctuation within the **physical time window** (τ_e=0.2s and the f_est window ≈3τ_f=0.3s); discretization granularity does not change the window content; in the zero-latency implementation there is no explicit τ_delay, so the "dt×layers minimum delay" argument does not take effect here. **That argument can be tested directly once an explicit forward-backward delay τ_delay is added** (paper §8.4/8.5 direction; already on the next-step list).

### 4.6 Deep gradient dilution (9:1 wrong outputs dominating fan-in) — root cause of the accuracy bottleneck

Measured at initialization (shared conv, 50-sample mean, TARGET=500):
```
avg δ_out[correct output] = −447 Hz
avg Σ δ_out[9 wrong outputs] = +581 Hz
|correct| / |Σ wrong| = 0.77
```
The FC layer δ fan-in summation (δ_fc = Σ_k w3·δ_out[k]) is **dominated by the stochastic contributions of the 9 wrong outputs** (their directions are random across samples; mean-field GD averages them away with enough samples; in the spiking system the per-sample noise makes the correct signal only ~0.8× the wrong sum) — so deep features (W1/W2) can hardly become discriminative, and the output layer cannot read them out → frozen acc stuck at 0.1-0.2. **This is the root cause of the accuracy bottleneck**: it needs ~50-100k sample magnitude (this machine ~10+ hours) or a stronger output-layer correction mechanism.

## 5. Results summary (all frozen evaluation)

| Config | Samples | ΔT | α | FDA | TARGET | Test acc |
|---|---|---|---|---|---|---|
| avg pool, R=200 | 1500 | 0.5 | 1e-6 | 2000 | 200 | 0.102 |
| avg pool, R=400 | 2000 | 1.0 | 2e-6 | 1000 | 400 | 0.159 |
| avg pool, R=400 | 4000 | 1.0 | 5e-6 | 1000 | 400 | 0.102 (exploded, loss→19) |
| stride, R=200 | 2500 | 1.0 | 2e-6 | 1000 | 200 | 0.147 |
| stride, R=200 | 2500 | 1.0 | 5e-6 | 8000 | 200 | 0.088 |
| stride, R=200 | 800 | 4.0 | 1.25e-6 | 8000 | 400 | 0.101 (exploded) |
| stride, R=200 | 2500 | 1.0 | 5e-8 | 4000 | 200 | 0.111 |
| **stride, R=200** | **10000** | 1.0 | **1e-7** | 3000 | 1000 | **0.202** (10k long run completed, peak 0.265) |
| **stride, R=200** | **8000/30000** | 1.0 | 1e-7 | 3000 | 1000 | **~0.20-0.25 plateau** (30k plan stopped at 8k per user request; acc oscillates at 0.16-0.25, no upward trend) |
| **shared conv, R=200** | **2000** | **10.0** | **1e-8** | 3000 | 500 | **0.110** (final eval n=1000; periodic-eval peak 0.155; ΔT=10 covariance ~3% ✓) |
| stride CE, α=1e-6, τ_sm=200 | 3000 | 1.0 | — | 3000 | 200(G) | 0.114 (CE negative result, rate explosion, see §5.1) |
| stride CE, α=1e-6, τ_sm=1000 | 800 | 1.0 | — | 3000 | 200(G) | 0.095 (same as above) |
| stride CE, α=3e-7, τ_sm=500 | 800 | 1.0 | — | 3000 | 200(G) | 0.140 (same as above) |
| **stride, R=200 (overnight long run)** | **50000** | 1.0 | 1e-7 | 3000 | 1000 | **0.124 (overtraining collapse! see §5.2)** |
| **shared conv (overnight long run)** | **30000** | 1.0 | 3e-8 | 3000 | 1000 | **0.202 (n=1000; @30k periodic eval 0.285; see §5.2)** |

> Full list of runs is in `mnist_runs_summary.csv` (ready to use).

### 5.1 Cross-entropy mode (2026-08-13 formally completed — negative result, mechanism located)

Implemented the **CE loss** per the user's suggestion (the paper's default MSE-type δ=f̂−TARGET·y can be swapped out; the constructive equivalence holds for any differentiable loss):
- `mnist_shallow.py` adds a `LOSS` parameter (argv[13]="ce") and `TAU_SM` (argv[14], softmax temperature, default 100 Hz);
- δ_out = G·(softmax(f̂/τ_sm) − y), G=TARGET (gain); **zero-sum gradient Σδ=0**;
- Gradient FD 12/12 passed (stride × CE, re-checked 2026-08-13).

**Official results (all failed)**:

| Config | Samples | Frozen acc | Phenomenon |
|---|---|---|---|
| α=1e-6, τ_sm=200, T=200 | 3000 | 0.114 | CE loss 2.4→3.6 (rising, not falling); output rate explosion: class 7 reaches 3351 Hz, global peak 11938 Hz; softmax fully saturated |
| α=1e-6, τ_sm=1000, T=200 | 800 | 0.095 | same as above (rate peak 16115 Hz) |
| α=3e-7, τ_sm=500, T=200 | 800 | 0.140 | rates moderate (peak 2528 Hz) but still no discriminative learning |

**Mechanism (located, material for the paper discussion)**:
1. **Unbounded rates → softmax saturation**: the CE loss constrains only the relative ordering of output rates, with no MSE-type TARGET·y "anchor" → all output rates grow unboundedly (up to tens of kHz) → f̂/τ_sm far exceeds 1 → p becomes one-hot, and the gradient is nonzero only at argmax and y;
2. **FC-layer fan-in cancellation → deep-layer starvation (particular to this construction)**: the zero-sum δ makes the FC sum d_fc = Σ_k w3·δ_k ≈ G·(w3[:,argmax] − w3[:,y]); when the two weight rows are close, d_fc ≈ 0. And early in each sample (softmax near-uniform, δ_y≈−0.9G, δ_k≈+0.1G) Σw3·δ_k ≈ 0.1G·Σw3 − G·w3[:,y] ≈ 0 (structural cancellation when w3 rows are uniform) → **W1/W2 receive almost no signal** (measured: W1/W2's P mean 0.303 barely moves); CE trains only W3 and the output bias;
3. Per-sample transient adaptation inflates the rolling acc (0.21-0.25); frozen evaluation reveals the true level ≈ random.

**Conclusion**: the CE mode needs extra mechanisms in this construction (output-rate normalization/adaptive temperature/explicit τ_delay to suppress in-sample adaptation) to work; the MSE-type δ=f̂−TARGET·y's TARGET anchoring is a necessary self-balancing mechanism for the noisy spiking system. **The CE code is retained as a negative control**, and the MNIST main results still use the MSE type.

**Running/planned** (2026-08-13 overnight, user asleep, all automated):
- Long run A: stride MSE 50k samples (α=1e-7, TARGET=1000, ΔT=1, FDA=3000) → tests the §4.6 "deep features need 50-100k samples" hypothesis (runs/stride50k/);
- Long run B: shared-conv MSE 30k samples (α=3e-8, TARGET=1000, ΔT=1, FDA=3000) → the shared version's first long training (runs/shared30k/);
- Table measurement: mnist_table.py 8 configs × gradient alignment/variance/SNR/covariance/loss plateau/frozen acc (results → mnist_table_results.csv).

### 5.2 Overnight long-run results (2026-08-14 morning — important findings)

**Long run A: stride locally-connected version 50k samples (α=1e-7, TARGET=1000, ΔT=1) → final 0.124, overtraining collapse**

```
Periodic-eval acc trajectory:
@500-5k:   0.22 → 0.205 → 0.16 → 0.185 → 0.235 → 0.285(3k) → 0.255 → 0.285(4k) → 0.265 → 0.260(5k)
@5.5k-13k: 0.245 → 0.165 → 0.185 → 0.190 → 0.205 → 0.240 → 0.265 → 0.185 → 0.230 → 0.265(10k) → ... 0.205
@13k on:   0.105-0.155 band oscillation, peak never exceeds 0.18; @30k-50k essentially locked at 0.09-0.16
Final (n=1000): 0.124
```

**Conclusion: the locally-connected version's "50-100k sample breakthrough" hypothesis is rejected** — it is not undertraining but **overtraining collapse**: the peak is at ~3-4k samples (0.285), after which the noise random walk (per-parameter noise 12× higher than the shared version, see t1/t7 table comparison) keeps destroying the learned structure, and acc permanently falls back to ~0.1-0.15. The training loss stays low (~0.043), a fake signal from in-sample transient adaptation (§4.1), decoupled from test acc. **The local version needs early stopping (~3-5k samples) or stronger regularization/inhibition.**

**Long run B: shared conv 30k samples (α=3e-8, TARGET=1000, ΔT=1) → final 0.202 (n=1000), @30k periodic eval 0.285 (n=200)**

```
Periodic-eval acc trajectory:
@500-8k:  0.08 → 0.13 → 0.115 → 0.115 → 0.125 → 0.095 → 0.175 → 0.135 → 0.14 → 0.145 → 0.15 → 0.18(6k) → ...
@10k-20k: 0.155 → ... 0.175-0.19 band climbing, 0.215(15k) first breaks 0.2
@21k-30k: 0.18-0.235 band stably climbing, @26k 0.235 → @30k 0.285 (final eval jumped up)
Final (n=1000): 0.202
```

**Conclusion: the shared-conv version shows genuine, monotonically accumulating learning with sample count** (position evidence pooling → per-parameter SNR×12 → signal growth outruns noise drift); 30k samples (2.7h) doubled frozen acc from 0.110 (2k@ΔT=10) to 0.202, with no sign of collapse — continuing longer training is the clear next step (projected 50-100k samples → 0.3+).

**Paper significance of the two-version contrast**: same construction, same learning rule, differing only in "whether weights are shared" (= temporal reuse of the receptive field), decides whether long training accumulates or collapses — the per-parameter evidence count (144 positions vs 1) directly determines SNR, and hence whether the learning dynamics are stable. This provides scaling-up empirical support for the noise-dominated limit of §8.5.

**Mean-field (noise-free) control evaluation (2026-08-14 morning, eval_meanfield_shared.py / eval_meanfield.py)**:

| Checkpoint | Mean-field test | Mean-field train | Spike frozen test (final n=1000) | Ratio (spike/mean-field) |
|---|---|---|---|---|
| shared30k (30k samples) | 0.230 | 0.240 | 0.202 | 0.88 (noise loss ~12%) |
| stride50k (50k samples) | **0.326** | 0.315 | 0.124 | **0.38 (noise loss 62%!)** |

**Upgraded conclusion**: the stride version's final weights are *actually better* under noise-free mean-field (0.326 > 0.230), but the spike evaluation is beaten down to 0.124 by Poisson noise — the learned decision boundary is fragile to noise (thin margin); the shared version's weights have a lower mean-field ceiling but a robust boundary (noise loss only 12%). Mechanism: stride accumulates evidence from only 1 position per parameter (large training-signal noise), so the learned weights "learn along the noise" — the discriminative structure exists but the margin is thin; the shared version has 144-position evidence per parameter (training signal cleaner by 12×), learning a more robust boundary. **"Weight sharing" improves both training dynamics (accumulate vs collapse) and evaluation noise robustness.**

### 5.3 Parameter measurement table (mnist_table.py, 2026-08-14 morning, seed=0, N=1000 samples per config)

Definitions: **align** = per-sample cos(ΔP, −g), alignment with the descent direction (ideal=1, noise-dominated→0); **updateSNR** = ‖mean(ΔP)‖/‖std(ΔP)‖ (cross-sample consistency; 1/SNR ≈ per-sample noise multiplier); **expEff** = ‖mean(ΔP)‖/‖α·τ_e·ΔT·mean(g)‖ (fraction of the theoretically expected update actually realized); **corr(e,d)/bias** = per-layer covariance over the last 5 samples; **loss plateau** = mean±std over the last 100 samples (MSE); frozen = frozen evaluation n=500.

**Table 1: overall (loss plateau / frozen acc / global alignment)**

| # | Config | α | TARGET | ΔT | loss plateau | train acc(roll) | frozen acc | align(global) |
|---|---|---|---|---|---|---|---|---|
| t1 | stride | 1e-7 | 1000 | 1.0 | 0.0450±0.010 | 0.24 | **0.188** | +0.261 |
| t2 | stride | 2e-7 | 1000 | 1.0 | 0.0415±0.012 | 0.29 | **0.204** | +0.216 |
| t9 | stride | 5e-7 | 1000 | 1.0 | 0.0388±0.012 | 0.34 | 0.098 | +0.189 |
| t5 | stride | 1e-6 | 1000 | 1.0 | 0.0323±0.019 | 0.52 | 0.142 | +0.252 |
| t3 | stride | 1e-7 | 500 | 1.0 | 0.0521±0.014 | 0.15 | 0.130 | +0.149 |
| t10 | stride | 2e-7 | 2000 | 1.0 | 0.0348±0.015 | 0.43 | 0.170 | +0.338 |
| t4 | stride | 1e-7 | 1000 | 4.0 | 0.0444±0.013 | 0.25 | 0.156* | +0.272 |
| t6 | stride,R=400 | 1e-7 | 1000 | 1.0 | 0.0499±0.002 | 0.08 | 0.104 | +0.072 |
| t7 | **shared** | 3e-8 | 1000 | 1.0 | 0.0491±0.010 | 0.15 | 0.086 | +0.232 |
| t8 | **shared** | 1e-8 | 500 | 4.0 | 0.0650±0.031 | 0.09 | 0.096 | +0.218 |

\* t4 used only 400 samples (ΔT=4 time cost), so the frozen value is low; t7/t8's low frozen values are because the shared version is far from finished at 1000 samples (see long run B).

**Table 2: gradient angle (per-sample cos(ΔP, −g) per layer, ± is cross-sample std)**

| # | Config | W1(conv) | W2(FC) | W3(out) | Global |
|---|---|---|---|---|---|
| t1 | stride 1e-7 | +0.152±0.158 | +0.181±0.175 | +0.449±0.194 | +0.261±0.221 |
| t2 | stride 2e-7 | +0.099±0.143 | +0.119±0.156 | +0.428±0.208 | +0.216±0.228 |
| t5 | stride 1e-6 | +0.089 | +0.106 | +0.561 | +0.252±0.290 |
| t10 | stride 2e-7, T=2000 | +0.167 | +0.191 | **+0.656** | +0.338±0.288 |
| t4 | stride ΔT=4 | +0.164 | +0.198 | +0.454 | +0.272±0.251 |
| t6 | stride R=400 | +0.044 | +0.053 | +0.120 | **+0.072** |
| t7 | shared | +0.146 | +0.191 | +0.359 | +0.232±0.231 |
| t8 | shared ΔT=4 | +0.188 | +0.175 | +0.291 | +0.218±0.266 |
| early(60 samples) | stride 1e-7 | +0.376 | +0.433 | +0.622 | +0.477 |

Structural regularity: **W3 (output layer) alignment is always highest (0.29-0.66), deep layers (W1/W2) only ~0.1-0.2** — error decays layer by layer + deep noise accumulation (consistent with XOR); per-sample alignment decays with training (early at 60 samples +0.48 → 1000 samples +0.26, as weights enter the noise-dominated region the signal share drops); alignment std ~0.15-0.29 (huge per-sample directional fluctuation).

**Table 3: update variance and SNR (per-layer updateSNR / expEff / sign-consistency rate)**

| # | Config | SNR W1/W2/W3 | expEff W1/W2/W3 | signCons W1/W2/W3 |
|---|---|---|---|---|
| t1 | stride 1e-7 | 0.058/0.059/0.062 | 0.081/0.082/0.100 | 0.564/0.467/0.476 |
| t2 | stride 2e-7 | 0.047/0.047/0.040 | 0.062/0.060/0.066 | 0.543/0.446/0.474 |
| t5 | stride 1e-6 | 0.032/0.030/**0.007** | 0.056/0.049/0.016 | 0.521/0.422/0.463 |
| t10 | stride 2e-7,T=2000 | 0.046/0.045/0.014 | 0.083/0.080/0.049 | 0.538/0.436/0.480 |
| t6 | stride R=400 | 0.041/0.042/0.047 | 0.027/0.027/0.032 | 0.559/0.474/0.451 |
| t7 | shared | **0.108**/0.084/0.123 | 0.075/0.149/0.203 | **0.788**/0.436/0.306 |
| t8 | shared ΔT=4 | **0.261**/0.110/0.132 | 0.159/0.139/0.156 | **0.808**/0.469/0.340 |

Regularity: **updateSNR decreases monotonically with α** (1e-7→1e-6: 0.058→0.007, noise ∝ α·√f_da grows); **expEff only 3-20%** — the noise random walk eats 80-97% of the theoretical update (per-sample noise ~16-70× the signal); **the shared version's W1 SNR and sign-consistency rate are doubled/higher** (direct evidence of position evidence pooling: 0.108 vs 0.058, 0.788 vs 0.564); ΔT=4 further raises the shared version's SNR (0.261). Sign-consistency rate ~0.45-0.8, deep layers ~0.45 (near random) → the gradient manifests only at the expectation/accumulation level (consistent with the XOR construction).

**Table 4: covariance (last 5 samples, per-layer corr(e,d) and |cov|/|E[e]E[d]|)**

| # | Config | corr W1/W2/W3 | bias W1/W2/W3 |
|---|---|---|---|
| t1 | stride 1e-7, ΔT=1 | −0.003/+0.002/+0.018 | 0.171/0.063/2.232 |
| t3 | stride T=500 | +0.001/−0.000/+0.052 | 0.087/0.001/3.644 |
| t4 | stride ΔT=4 | +0.002/−0.002/+0.004 | 0.160/0.168/0.243 |
| t5 | stride 1e-6 | −0.009/+0.005/+0.002 | 0.855/0.467/0.176 |
| t7 | shared ΔT=1 | +0.007/+0.003/+0.001 | 0.695/0.151/0.027 |
| t8 | shared ΔT=4 | +0.015/+0.004/−0.000 | 1.490/0.315/0.002 |

Regularity: **corr(e,d) all ≤0.05, mostly ~0.001-0.02 — the negligible-covariance conclusion holds across all configs**; W3 has the highest corr (+0.018~+0.052, zero-latency window overlap); **ΔT=4 pushes W3 corr down to ~0.004** (window dilution, reproducing the §4.4 conclusion); deep-layer bias (|cov|/|E[e]E[d]|) can be as high as 0.7-3.6 (the share is amplified as E[d]→0, §4.4 mechanism) — but the absolute update-direction bias is still ~2-3% (corr magnitude).

### 5.4 Output-layer inhibition pool (κ normalization) — added 2026-08-14

**Background**: §4.6 deep gradient dilution — 9 anchor-free free-firing outputs send inhibiting instructions each sample (+581 vs correct −447, 0.77×). Solution: a shared inhibition pool on the output layer (biological counterpart: Carandini–Heeger normalization circuit / wide-area PV inhibitory interneuron): each output neuron receives inhibition proportional to the total rate, `u[OUT] -= κ·Σf̂·dt` (subtractive version, mnist_shared.py argv[11]=KAPPA). δ remains an independent MSE-vector injection (does not touch the gradient structure, avoiding the CE zero-sum pitfall).

**κ scan (3k samples, seed=0, n=500 frozen, α=3e-8, others default)**:

| κ | Frozen acc | Note |
|---|---|---|
| 0 (control) | 0.134 | prior baseline |
| 0.05 | 0.150 | |
| 0.1 | 0.158 | |
| 0.2 | 0.192 | beats old 30k best (0.202) at only 3k |
| 0.4 | 0.280 | 3k peak, but 10k collapse (0.170) |

**Controls rejected**: lateral inhibition γ=0.1 → 0.114 (mutual suppression hurts the correct output); scaled R=400 synchronous rescale → 0.106 (no benefit within 3k); κ+ΔT=2 → 0.172@5k (halving the sample count at same physical time is ineffective); TARGET=500 → 0.164@20k (lowering the anchor weakens too early).

**Mechanism evidence (table measurement, 3k samples κ=0 vs κ=0.2)**: W3 layer E[e] 2.1→10.9 (output 5× more active), E[d] −71.5→−29.7 (**inhibiting instructions cut by 58%**), signal product E[e]·E[d] ×2.2; corr(e,d) still ~0 (inhibition does not break the independence assumption).

**Collapse phenomenon (structural deadlock of the subtractive inhibition pool)**: κ=0.2 full α peaks at 10-13k (0.29-0.31) then runs out of control (correct output rate 4212 Hz >> TARGET, all wrong outputs extinguished) → total extinction ~19.5k (total=2, final 0.126). Mechanism: no anchor (correct output can drift to 4×TARGET) + ReLU truncation + mutual dependence → an all-extinction equilibrium; the larger κ, the earlier the collapse.

**α-halving solution**: κ=0.2 + α=1.5e-8 @20k → **0.348** (current best, n=500); output rates moderate (y~200, no sign of runaway); seed=1 reproduces **0.334**; κ=0.2 full α @10k seed=0/1 both 0.292 (fully reproduced).

**30k spike long run final (n=500)**: k0p2_a10_30k (α=1e-8) → 0.330, k0p15_a15_30k (κ=0.15) → 0.300, k0p2_a15_30k → ~0.30 (@28k periodic 0.300). **Spiking saturates at ~0.33-0.35 by 20-30k** (0.348@20k is still the peak) — more samples no longer gives linear gains; entering the noise-bottleneck region.

**Mean-field GD theoretical ceiling (2026-08-14 night, gd_meanfield_shared.py, noise-free expected gradient)**:
- κ scan @30k samples (LR=1e-8, n=1000): κ=0→0.900, κ=0.1→**0.916**, κ=0.2→0.915, κ=0.3→0.905, κ=0.4→0.913;
- **300k samples κ=0.1 → 0.965** (8.5 min; N>60000 auto-repeats epochs);
- **Conclusion**: the small architecture (18,898 parameters) has a theoretical ceiling of ~0.96; the user's 70-80% target (proving the framework works) is fully within the architecture's capacity; **spike vs mean-field = 0.35 vs 0.9 → ~2.5× noise loss is the only bottleneck**; the path = make spiking approach mean-field.

**R-hypothesis test (completed 2026-08-15, conclusion: not supported on MNIST shared conv for now, 3/3 negative results)**:
- Math (following the user's derivation): scaling R and δ/f_da/e up together K× → signal∝K², variance∝K³ → **SNR∝√K**; scale α by 1/R² to keep the expected step constant; shrink dt to keep λ·dt≤1;
- `exp3/r400_a3p75_3k` (R=400/FDA=6000/BIAS=60/dt=0.01/κ=0.2/TARGET=1000/α=3.75e-9/3k): periodic eval 0.060→0.080→0.075→0.110→0.085→0.090, **final 0.122**; output layer extinguished several times (total=0/1) → **R hypothesis rejected**;
- `exp3/r800_a0p9375_2k` (R=800/FDA=12000/BIAS=120/dt=0.005/κ=0.2/α=9.375e-10/2k): @500 total=0, @1000 total=0 and acc=0.100, output deadlocked, terminated proactively;
- `exp3/r800_k0_a3p75_2k` (complementary test, κ=0 to rule out inhibition-pool interference, R=800/dt=0.005/α=3.75e-9/2k): periodic 0.100→0.075→0.095→0.130, **final 0.106** → still no benefit;
- Difference from XOR: the MNIST output layer has 10 independent targets + deep fan-in structure; under R scaling, the κ inhibition pool / weight-sharing dynamics do not stay invariant under rate scaling; **XOR's √R gain does not automatically transfer to MNIST**. If continuing, one should use a "R scaling + output-layer competitive anchor" joint design rather than directly applying the XOR scaled config.

## 5.5 2026-08-15 parameter-table measurements and checkpoint batch evaluation

### 5.5.1 Parameter-table measurements (merged into `mnist_table_results.csv`, with new KAPPA/GAMMA/DT columns; full four tables in `docs/TABLE_RESULTS_SHARED_CN.md`)

**N=1000 quick table (2026-08-15)**

| α | κ | loss plateau | frozen acc | align_all | corr W1/W2/W3 | updateSNR W1/W2/W3 |
|---|---|---|---|---|---|---|
| 1.5e-8 | 0.2 | 0.0547±0.0138 | 0.134 | +0.281 | +0.004/+0.001/+0.005 | 0.122/0.122/0.156 |
| 1e-8 | 0.2 | 0.0556±0.0117 | 0.098 | +0.313 | −0.003/+0.002/+0.004 | 0.162/0.122/0.161 |
| 1.5e-8 | 0 | 0.0515±0.0074 | 0.100 | +0.287 | −0.002/−0.002/+0.002 | 0.169/0.100/0.137 |
| 2e-8 | 0.2 | 0.0578±0.0193 | 0.158 | +0.253 | +0.009/+0.005/+0.004 | 0.089/0.114/0.153 |
| 1.5e-8 | 0.4 | 0.0600±0.0185 | 0.126 | +0.241 | −0.007/+0.000/+0.003 | 0.078/0.127/0.162 |

**N=3000 official table (2026-08-15, 4-core parallel)**

Overall:

| Config (shared, ΔT=1, R=200, FDA=3000, TARGET=1000) | seed | loss plateau | frozen acc | align_all |
|---|---|---|---|---|
| α=1.5e-8, κ=0.2 | 0 | 0.0524±0.0145 | **0.164** | +0.209±0.271 |
| α=1e-8, κ=0.2 | 0 | 0.0536±0.0158 | 0.156 | +0.227±0.275 |
| α=1.5e-8, κ=0 | 0 | 0.0479±0.0085 | 0.148 | +0.212±0.222 |
| α=1.5e-8, κ=0.2 | 1 | 0.0495±0.0087 | 0.104 | +0.229±0.227 |
| α=3e-8, κ=0 (08-14) | 0 | 0.0483±0.0050 | 0.132 | +0.166±0.204 |
| α=3e-8, κ=0.2 (08-14) | 0 | 0.0500±0.0120 | 0.138 | +0.176±0.274 |

Gradient angle (per-sample cos(ΔP,−g)):

| Config | W1 | W2 | W3 |
|---|---|---|---|
| α=1.5e-8, κ=0.2, seed0 | +0.094 | +0.176 | +0.356 |
| α=1e-8, κ=0.2, seed0 | +0.126 | +0.198 | +0.357 |
| α=1.5e-8, κ=0, seed0 | +0.115 | +0.171 | +0.350 |
| α=1.5e-8, κ=0.2, seed1 | +0.130 | +0.175 | +0.383 |

Update variance/SNR/expected-update efficiency:

| Config | updateSNR W1/W2/W3 | expEff W1/W2/W3 | signCons W1/W2/W3 |
|---|---|---|---|
| α=1.5e-8, κ=0.2, seed0 | 0.064/0.083/0.090 | 0.024/0.075/0.087 | 0.760/0.421/0.239 |
| α=1e-8, κ=0.2, seed0 | 0.080/0.094/0.116 | 0.033/0.095/0.120 | 0.798/0.422/0.248 |
| α=1.5e-8, κ=0, seed0 | 0.090/0.066/0.108 | 0.059/0.117/0.174 | 0.750/0.428/0.300 |
| α=1.5e-8, κ=0.2, seed1 | 0.069/0.074/0.145 | 0.058/0.148/0.248 | 0.808/0.447/0.394 |

Covariance (last 5 samples):

| Config | corr W1/W2/W3 | bias W1/W2/W3 |
|---|---|---|
| α=1.5e-8, κ=0.2, seed0 | +0.014/+0.005/+0.007 | 0.917/0.481/0.232 |
| α=1e-8, κ=0.2, seed0 | +0.008/+0.008/+0.006 | 0.495/6.173/0.174 |
| α=1.5e-8, κ=0, seed0 | +0.016/+0.002/+0.005 | 1.128/0.149/0.062 |
| α=1.5e-8, κ=0.2, seed1 | −0.005/+0.003/+0.006 | 0.700/0.798/0.101 |

Key points: **corr(e,d) still ≤0.016 across all KAPPA configs** (the independence assumption continues to hold); at N=3000 per-sample alignment falls back to 0.21-0.23 (lower than N=1000's 0.25-0.31; training has entered the noise-dominated region); κ=0.2 does not lower per-sample alignment, so the gain comes from suppressing the inhibiting instructions (§5.4), not a covariance change; α=1e-8 has slightly higher SNR at N=3000 but a slightly lower frozen acc than α=1.5e-8 (slower learning).

### 5.5.2 Best-config seed reproduction (κ=0.2, α=1.5e-8, 20k)

| seed | final frozen acc (n=500) | Note |
|---|---|---|
| 0 | 0.348 | existing best |
| 1 | 0.334 | reproduced |
| 2 | **0.272** (added 2026-08-15) | periodic peak 0.305@16k, @20k=0.265 |

Mean 0.318, std 0.033. The train-acc/test decoupling still exists (train roll 0.26 @20k); output rates basically healthy (y mostly 0-366, no 4212-scale runaway).

### 5.5.3 Checkpoint batch evaluation (n=1000 frozen spike + n=1000 mean-field, KAPPA self-consistent solution 8 iterations)

Data file: `mnist_checkpoint_eval_batch.csv`. Key points:
- Best weights `k0p2_a15`: spike 0.308 / mean-field 0.299; seed1 weights mean-field 0.387 (better boundary structure but slightly lower spike);
- Collapsed weights `k0p2_20k`: spike 0.098 / mean-field 0.312 — **the mean-field still has structure; the spike death is the dynamical death of the inhibition-pool all-extinction, not the complete loss of weight structure**;
- κ=0.4 weights: spike 0.175 / mean-field 0.337, same phenomenon;
- base (κ=0): 0.126/0.128, mean-field and spike agree (robust boundary).

## 5.6 0.8-goal sprint (2026-08-15 late night → 2026-08-16 early morning, achieved)

### 5.6.1 Capability proof: mean-field weights → spike readout

- Mean-field GD 300k (LR=1e-8, seed=0) all completed: κ=0/0.1/0.2/0.4 → **0.965/0.965/0.965/0.961** (n=1000, `meanfield_ckpts/`, checkpoint every 25k).
- **No-reset evaluation** (old protocol, continuous cross-sample state): pouring these 0.96 weights into the spike network gives frozen acc only **0.115-0.203**; long-window readout (SAMPLE_T=2) gives zero improvement (0.115→0.116 etc.); R-scaling scan (R=200/400/800/1600) is even worse (0.203→0.115-0.158).
- **Diagnosis (`dbg_pulse_output.py`)**: the output-layer membrane potential is pushed to a deep negative by κ inhibition (−300~−3500), a "winner-lock" dynamic — the first-firing class suppresses the others and, with no leakage, the lock persists across samples. The lock-error rate determines acc. Inflated training acc = in-sample adaptation locking the correct class vs frozen evaluation with no adaptation locking the wrong class — the same underlying mechanism.
- **Per-sample reset evaluation (`eval_mf_batch.py --reset_each`)**: mean-field weights spike readout **0.942-0.953** (single) / **0.955-0.960** (long-window) / **0.954-0.961** (counting) ≈ mean-field level → **the architecture + readout protocol capacity is fully sufficient; the bottleneck = training protocol and state pollution**.

### 5.6.2 Head-on: per-sample-boundary-reset spike online training (within the paper's SDE framework)

- `mnist_shared.py` adds `RESET_PER_SAMPLE` (argv[13]): reset u/r_est/f_est/E before each sample (physical implementation = state reset/silent interval before sample presentation); training and evaluation protocols are consistent.
- Three from-scratch trainings (pure spike, random init, in-place local update per step, no weight injection):

| Dir | κ | α | N | Final n=1000 | Independent eval single/long-window/counting |
|---|---|---|---|---|---|
| exp4/reset_a15_20k | 0.2 | 1.5e-8 | 20k | 0.767 | 0.776/0.810/0.812 |
| exp4/reset_a10_30k | 0.2 | 1e-8 | 30k | 0.778 | 0.789/0.815/0.811 |
| **exp4/reset_a15_cont** | 0.2 | 1.5e-8→7.5e-9 | 20k+20k | **0.824** | **0.817/0.848/0.850** |

- **The 0.8-reaching model = exp4/reset_a15_cont**: first run 20k@1.5e-8 (0.767) + continuation 20k@7.5e-9 (0.824), all three readouts ≥0.8.
- Comparison: no-reset same config (κ=0.2, α=1.5e-8, 20k) is only 0.348 → **per-sample-boundary reset is the single biggest improvement** (0.35 → 0.77, same sample budget).
- Full trajectory (n=200 periodic eval): a15: @7k 0.56 → @10k 0.72 → @15.5k 0.765 → @20k 0.815; no overtraining collapse.
- Capability-proof control: mean-field weight reset readout 0.94-0.95 (ceiling); spike training 0.82 → margin 0.13, theoretical room remains (longer training / multi-seed).

### 5.6.3 Rejected and shelved routes (recorded for reference)

- **K-sample local-update accumulation: user rejected** — a neuron has no capacity to store K samples of update magnitude, contradicting the paper's online SDE framework (in-place update per step; drift dominates the long-time average).
- **R scaling: shelved** — scanning R=400/800/1600 directly with mean-field weights gives no spike-readout improvement (0.203→0.115-0.158); needs joint recalibration with TARGET/KAPPA/output anchor before revisiting.
- **Mean-field weight-init fine-tuning: research control only** — a tape-out chip has no weight injection; not a delivery route.

## 5.7 Plausible-protocol: LIF membrane leakage + inter-sample silent interval (2026-08-16, user ruled hard reset non-plausible)

### 5.7.1 Background and ruling

- Hard reset (resetting state per sample) reaches 0.824 but is **not plausible**: chips/biological systems have no "clear" signal. User ruling: the acceptable upper bound = **present the samples separated by a gap** (the ISI silent interval).
- The user points out that the key is frequency residue, not membrane-potential residue; and provides the old paper `一生有爱何惧风飞沙.txt` (§4.5 complete LIF derivation).

### 5.7.2 Pollution-source diagnosis (mean-field κ=0.2 weights, n=1000)

| Protocol | single readout | Conclusion |
|---|---|---|
| no handling (baseline) | 0.203 | — |
| clear frequency only (u kept) | 0.205 | **frequency residue is not important** (τ_F=0.1s << sample 1s, already decayed clean within the sample) |
| clear u only (frequency kept) | **0.948** | **u residue is the main cause**: IF has no leakage, deep-negative lock persists 100% across samples |
| ISI=25/50 (pure silence, u doesn't decay) | 0.246/0.257 | silence ineffective — u has no leakage so it doesn't decay |
| hard reset (clear all) | 0.947 | control upper bound |

→ The user's "membrane-potential residue is not important" does not hold under IF (IF's u doesn't decay); but real neurons are **LIF** (membrane potential decays as e^{-λt}) — the silent period naturally returns to zero, requiring no hard reset.

### 5.7.3 Paper §4.5 LIF theory (old paper, direct citation)

- LIF definition: du/dt = I − λu; firing condition I > λθ; **frequency formula f_out = −λ/ln(1 − λθ/I)**;
- IF is the LIF λ→0 special case; LIF asymptotically approaches ReLU, deviation = **frequency offset λ/2 + dead zone λθ**;
- Numerical verification (θ=1): τ_m=0.5 (λ=2) → offset 1 Hz (0.1%); τ_m=0.2 (λ=5) → 2.5 Hz (0.25%) — small-leakage LIF ≈ IF; expected-frequency/SDE deviation is negligible.

### 5.7.4 Plausible-protocol verification (mean-field weights, n=1000)

| τ_m | ISI steps | single/long-window/counting |
|---|---|---|
| 0.1 | 25 | 0.937 / 0.940 / 0.944 |
| 0.2 | 50 | 0.944 / 0.948 / 0.950 |
| 0.5 | 0 | 0.936 / 0.952 / 0.950 (leakage alone, no silence already suffices) |
| **0.5** | **100** | **0.955 / 0.956 / 0.953 (≥ hard reset 0.947)** |

**LIF leakage itself solves the cross-sample u pollution** (sample 1s = 2~10×τ_m; u memory naturally disappears); ISI guarantees both frequency and membrane potential return to baseline.

### 5.7.5 Plausible-protocol training (in progress)

- `exp4/lif_tm0p5_isi50_a15_20k`, `exp4/lif_tm0p2_isi50_a15_20k`: κ=0.2, α=1.5e-8, N=20k, ISI=50, τ_m=0.5/0.2 (mnist_shared.py argv[14]=ISI_STEPS, argv[15]=TAU_M).
- Goal: reproduce 0.8+ with a purely plausible protocol (no hard reset); see EXPERIMENT_PROGRESS.md result summary.

## 6.5 Plausible version (LIF+ISI) achieved, seed reproduction, and parameter relationships (supplement 2026-08-16 late night)

1. **Plausible-protocol (no hard reset) goal achieved**: LIF membrane leakage (old paper §4.5) + inter-sample silent ISI. Failure mechanism = **output-layer push/pull imbalance** (at TARGET=1000 the output target rate is far below the κ-inhibited achievable rate, so push cannot regain traction) → **TARGET=5000 + κ=1.0 recalibration** restores training.
2. **Multi-seed acceptance table** (protocol: 5 seeds × n=500 frozen, τ_m=0.5/ISI=100 readout):

| Model | Config | train eval | acceptance (mean±std) |
|---|---|---|---|
| main run `lif5_t5000_k1_isi50_a15_20k @14k` | T5000/K1.0/ISI50/τ_m0.5 | 0.507* | **0.877±0.007** (all seeds ≥0.868) |
| `@12k` | same | — | 0.873±0.017 (single 0.894/mean-field 0.900) |
| `@20k` | same (overtrained) | 0.507 | 0.750±0.021 (**TARGET-saturation collapse**) |
| s1 `lif5_s1_..._14k` | seed 1 | 0.701 | **0.874±0.006** (all seeds ≥0.864) |
| s2 `lif5_s2_..._14k` | seed 2 | 0.748 | **0.806±0.015** (all seeds ≥0.784) |

   *The main run's 20k built-in training evaluation (single implementation, no per-sample reset) is pathologically low in the peak region; acceptance relies on multi-seed frozen.
   **Conclusion: all cross-seed means ≥0.8; s2 is significantly lower by ~7 points (worst single seed 0.784) — cross-seed variance ~7 points; deliver with the 12-14k checkpoints.**

3. **Parameter × metric relationship scan** (16 groups × N=1000, same measurement protocol; full analysis in `TABLE_LIF_PARAM_SWEEP.md`):
   - **TARGET non-monotonic**: frozen acc 0.404 (3000) → 0.648 (5000) → 0.372 (7000); corr_W3 increases monotonically with TARGET (0.015→0.026→0.037);
   - **κ alignment paradox**: κ=0.2/0.5/1.0 → alignment 0.764/0.468/0.238 (monotonic decrease) but acc 0.414/0.598/0.648 (monotonic increase) — high per-sample gradient alignment ≠ good end-to-end performance; the global inhibition pool (κ=1.0) is metric-side evidence of fixing the push/pull imbalance;
   - **ISI double-edged**: 0/50/100 → acc 0.540/0.648/0.210; ISI=0's loss plateau is misleadingly low (0.0136, cross-sample pollution) but its alignment is also the lowest (0.166);
   - **Four-protocol same-measure control**: IF no-leakage 0.134 / hard reset 0.184 / LIF uncalibrated 0.132 / LIF recalibrated 0.648; alignment reversed 0.281/0.794/0.798/0.238 — **the "alignment paradox" holds across protocols as well**;
   - **Cross-metric**: align_ALL vs frozen_acc ρ=-0.553 (n=16); align_std vs acc ρ=+0.482; snr_W3 vs align ρ=+0.553;
   - α=1.5e-8 optimal (0.648 > 3e-8's 0.564 > 7.5e-9's 0.516); τ_e=0.2 better than 0.1 (0.648 vs 0.570); τ_m=0.2 catastrophic (0.142), τ_m=1.0 slightly better (0.696) — all at the 1k early-stage measure.

## 6. Conclusions and next steps

1. **Mechanism fully verified**: per-sample alignment cos 0.68-0.91, FD 12/12 (locally-connected + shared conv), forward equivalence — local online BP works as theory predicts at MNIST scale;
2. **✅ User goal (frozen test acc ≥ 0.8) achieved (2026-08-16 early morning)**: `exp4/reset_a15_cont` pure-spike online training (reset protocol) final 0.824; independent evaluation single 0.817 / long-window 0.848 / counting 0.850;
3. Key mechanism conclusion: cross-sample state pollution is the biggest bottleneck for spike readout/training (0.96 weights → 0.2 readout); per-sample-boundary reset brings mean-field weight spike readout back to 0.94-0.95 and lifts same-budget training from 0.348 to 0.767;
4. R hypothesis (SNR∝√R): holds on XOR, fails on MNIST (including the pre-recalibration 4-tier scan with no improvement); K accumulation rejected (physically unrealizable);
5. Covariance still ≤0.016 across the full KAPPA parameter table; the independence assumption has empirical support;
6. **✅ Seed reproduction completed (2026-08-16 late night, see §6.5)**; optional follow-ups: 60-100k long training toward 0.85+, low-α fine-tuning of s2, write up the reset-mechanism analysis (paper material); mean-field 0.965 remains the architecture ceiling.

## 7. Reproduction

```bash
python mnist_loader.py                                  # data check
python verify_mnist_shallow.py <stride|avg|max>         # gradient FD verification (locally-connected version)
python verify_mnist_shared.py                           # gradient FD verification (shared-conv version)
python mnist_shallow.py <seed> <N> <ΔT> <R> <α> <FDA> <BIAS> <mode> <TARGET> <final_eval_n>
python mnist_shared.py <seed> <N> <ΔT> <R> <α> <FDA> <BIAS> <TARGET> <final_eval_n> [DT] [KAPPA] [GAMMA]
# e.g.: python mnist_shared.py 0 2000 10.0 200 1e-8 3000 30 500 1000
# e.g.: python mnist_shared.py 0 20000 1.0 200 1.5e-8 3000 30 1000 500 0.02 0.2 0   (inhibition-pool best config)
# e.g.: python mnist_shared.py 0 20000 1.0 200 1.5e-8 3000 30 1000 1000 0.02 0.2 0 1   (reset-protocol 0.8-achieving config)
python eval_checkpoints_batch.py <ckpt1> [ckpt2 ...]          # batch frozen spike + mean-field evaluation
python eval_mf_batch.py --reset_each --kappa=0.2 <ckpt...>    # reset-protocol three-readout evaluation (0.8-sprint standard protocol)
python gd_meanfield_shared.py <LR> <N> <KAPPA> <SEED>   # mean-field GD noise-free ceiling
python eval_meanfield.py                                # frozen-P mean-field evaluation
python measure_cov2.py <R> <ΔT> <FDA>                  # covariance control measurement (carrier/sample-duration scan)
python compare_rates.py / measure_terms.py / measure_dt_accum.py / per_sample_audit.py   # diagnostic tools
```
