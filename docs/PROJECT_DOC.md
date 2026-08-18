# Local Online Backpropagation (Probabilistic-Synapse SNN) — Simulation Project Documentation

> 📖 **Terminology note**: This document's coined/project-specific terms (recalibration, eligibility trace, cross-sample state pollution, frozen evaluation, multi-seed acceptance, mean-field, gradient alignment, collapse, etc.) carry a brief explanation at their first occurrence; the authoritative definitions all live in [docs/GLOSSARY.md](GLOSSARY.md).


> Project purpose: provide a constructive simulation verification for the paper *Local Online Backpropagation Based on Probabilistic Synapses: A Constructive Proof*:
> a spiking network of probabilistic synapses + IF neurons that learns online using only local signals (eligibility traces + local error spikes).
> Tasks: XOR residual topology (primary verification) + MNIST CNN (scale extension, from 2026-08-13).
> **This document is the complete handover document (updated 2026-08-15)**; detailed data is separately in docs/SWEEP_RESULTS_CN.md and docs/MNIST_RESULTS_CN.md.
> **2026-08-15 documentation reorganized**: all descriptive documentation is in `docs/` (the root keeps same-name symlinks), old logs in `logs/`, deprecated/superseded artifacts in `archive/`; project entry `README.md`.
> **Timeline log also in PROJECT_LOG.md (from 2026-08-14, for handovers at any time).**

---

## 1. Deliverables and Quick Use

### Main file (for the paper): `xor_residual_local_bp.py` (single self-contained file)

```bash
python xor_residual_local_bp.py            # default seed 0, 5000 samples (~3.5 min)
python xor_residual_local_bp.py 1 8000     # optional args: [seed] [num samples]
```

Run output:
- Header prints all parameters and the topology;
- Every 500 samples prints: loss (rolling), acc (rolling), cos/sample (rolling), cos/500-sample-window cumulative, cos/full-run cumulative, p sign consistency rate;
- At the end prints: training time, p range, frozen evaluation of the 4 XOR patterns (OK/FAIL), final metrics;
- Saves a four-panel figure `xor_local_bp_result.png` (loss, gradient alignment, p sign consistency rate).

### Helper files (all troubleshooting/analysis artifacts, see §7 status notes)

| File | Purpose |
|---|---|
| verify_mechanism.py | Constructed expected update vs analytic gradient (algebraic identity verification) |
| verify_trajectory.py | Actual noisy update vs gradient window-by-window cosine measurement |
| measure_covariance.py | Eligibility-trace–error covariance measurement vs sample duration |
| test_forward.py | Equivalence test of spiking forward vs rate-domain mean-field |
| clean_gd_xor.py | Exact gradient-descent baseline (**deprecated, see §7**) |
| noise_isolation.py | Noise-component isolation experiment (**deprecated, see §7**) |
| sweep_rd_dt.py / plot_sweep.py | R×ΔT grid sweep: time evolution of covariance × final loss (**see §4.5 and SWEEP_RESULTS_CN.md**) |
| mnist_loader.py / mnist_shallow.py | MNIST experiments: data loading + locally connected CNN with three downsampling modes (stride/avg/max) (**see §9 and MNIST_RESULTS_CN.md**) |

---

## 2. Paper Construction → Code Mapping

| Paper mechanism | Code implementation |
|---|---|
| Rate encoding (Poisson firing) | `RNG.poisson(R·x·dt)`, R=200 Hz as the "1" rate |
| Probabilistic synapse p_e, sign s_e | `P` (release probability), `SIGN` (±1, fixed) |
| Effective weight w = s·p/θ | `SIGN*P/THETA`, θ=1.0 (**uniform across all IF neurons**) |
| IF neuron ≡ ReLU (rate domain) | membrane `u += SIGN·k`, `n = floor(u/θ)` firing reset |
| Identity residual (physical implementation of h2 = ReLU(W2h1+b) + h1) | `u[h2] += n1` (identity injection into same membrane, effective weight 1/θ); mean-field `a2 = ReLU(z2 + a1/θ)` |
| Eligibility trace de/dt = −e/τ_e + S(t) | `E_trace += -E_trace·dt/TAU_E + pre_spikes[pre]`, E[e]=τ_e·f |
| Dopamine baseline encoding f_err = f_da − δ | error spike `k ~ Poisson(clip(FDA − δ_post,0)·dt)` |
| Local fixed drift rate C | `C = FDA = 500` |
| SDE: dz = s·α·e/(2p(1−p))·(dN−C·dt) | **p-domain path-level equivalent form**: `P += SIGN·ALPHA·E_trace·(k − C·DT)` (the 2p(1−p) cancels exactly, avoiding pathological numerics at the boundary) |
| Backward graph (isomorphic, transposed, ReLU-gated) | δ computed analytically via rate-domain chain rule: `δ_out = f_est/TAU_F − TARGET·y` (spike estimate), `δ_h2 = gate·w3·δ_out`, `δ_h1 = gate·(W2ᵀ·δ_h2 + δ_h2/θ)`, gate = r_est rate estimate > 1 Hz |

Network: 2 inputs + 3 bias neurons (30 Hz background input) + 16 h1 (dense) + 16 h2 (identity residual block) + 1 output,
38 neurons total, 337 plastic synapses. The residual topology demonstrates that the construction does not require a standard fully-connected layer-wise topology.

### Key modeling choices (adjudications where the paper is silent; new sessions must know)

1. **The error δ uses the spiked estimate of the output rate** (f_est/τ_f recovered to a rate), not the mean-field analytic value (the user explicitly required "use my SDE");
2. The backward-graph δ propagation is computed analytically by chain rule (the backward graph's functional result in the rate domain), and the error channel stays Poisson-stochastic;
3. **The current implementation is "zero forward-backward delay"** (δ uses the current-moment f_est) — the covariance measurement and the de-correlation discussion are both based on this, see §5;
4. The output-layer error is not gated (δ_out = f_est/τ_f − TARGET·y);
5. **The p-domain update** is path-level exactly equivalent to the paper's z-domain SDE;
6. **Rate-trace scaling**: r_est/f_est are leaky integrators whose steady-state value = τ×rate (e.g. f_est steady state = 0.1×true rate); every rate reading must be divided by τ to recover it.

---

## 3. Parameter Defaults (top of xor_residual_local_bp.py)

```
R=200.0       input rate scale (Hz)        BIAS_RATE=30.0  background input (Hz)
THETA=1.0     IF threshold (uniform)        TARGET=200.0    output target rate (Hz)
TAU_E=0.2     eligibility time constant (s) TAU_R=0.1       activity-estimate time constant (s)
TAU_F=0.1     output-rate-estimate time constant (s) FDA=500.0=C  dopamine baseline = local drift rate
DT=0.02       simulation step (s)           SAMPLE_T=2.0    sample presentation duration (s)
ALPHA=2.5e-6  learning-rate constant        P_INIT=0.3       initial release probability
R_GATE=1.0    ReLU gate rate threshold (Hz) N_H=16           hidden layer width
P[G2]=0.6     output bias synapse initial p (deliberately strengthened to guarantee initial output firing)
```

---

## 4. Key Results

### 4.1 Current version (SDE error, 5000 samples, seed 0)

```
loss:  0.123 → 0.070 (decreasing)
acc:   0.61 → 0.81 (increasing)
cos/sample(rolling):  +0.03 ~ +0.11 (weakly positive per-sample alignment)
cos/window(500):   +0.34, +0.04, +0.20, +0.13, −0.40... (oscillating, mean ~+0.05)
p-sign:           ~0.44-0.48 (near random, slightly below 0.5)
eval: (0,0)=72 ✓  (0,1)=275 ✓  (1,0)=152 ✓  (1,1)=105 ✗  → 3/4
```

At 10000 samples: loss plateaus at **0.066-0.11 (mean ~0.08)**, acc oscillates 0.61-0.87, eval snapshot 2/4 (oscillatory state, not converged).

### 4.2 Noise-dominated loss plateau (citable for the paper)

- Plateau value ~0.08, entering noise-dominated oscillation after ~2000-3000 samples;
- Criterion: gradient signal η·a·δ·ΔT ≈ noise floor α·e·√(f_da·ΔT) (~1.5-3e-3), corresponding to δ ≈ 30-80 Hz, i.e. L ≈ 0.03-0.08;
- The pure rate-estimate measurement-noise floor ≈ ½·(1/√(2τ_f·f))² ≈ 0.013; weight jitter raises the plateau to ~0.08.

### 4.3 Per-sample SNR structure (theoretical basis for improvement directions)

```
SNR = |δ|·√ΔT / √f_err   （f_err ≈ f_da）
```

- Raising only f_da: worse (the carrier noise √f_da grows);
- Raising the overall rate scale R (δ and f_da scaled up together): SNR ∝ √R ✓;
- Bursting/regularized firing (Fano factor <1): equivalently lowers √f_err ✓ ("equivalent higher carrier frequency");
- Longer samples ΔT: √ΔT ✓ (verified: loss 0.164→0.064, ΔT 0.5s→4s);
- The hidden-layer SNR is worse (~1.3) and additionally sits on error-propagation attenuation: δ_hidden ~ w·δ_out ~ 0.3×.

### 4.4 Covariance measurement (verifying the "covariance is negligible" conclusion)

`measure_covariance.py`, 1500 samples, corr(e, δ) by layer / by sample phase:

| ΔT | W3 top 25% | W3 bottom 75% | W1 | W2 | loss(last 300) | p-sign |
|---|---|---|---|---|---|---|
| 0.5 s | +0.138 | +0.169 | +0.03 | −0.01 | 0.164 | 0.483 |
| 1.0 s | +0.137 | +0.177 | +0.03 | −0.00 | 0.137 | 0.474 |
| 2.0 s | +0.136 | +0.153 | +0.03 | −0.01 | 0.113 | 0.447 |
| 4.0 s | +0.127 | +0.112 | +0.03 | −0.01 | 0.064 | 0.425 |

Conclusions:
1. corr(e,δ) is concentrated in W3 (+0.13~0.17), W1 ~ +0.03, W2 ≈ 0; the bias on the update direction is ~2-3% — **negligible**;
2. **The covariance does not shrink with sample duration**; the top 25% and bottom 75% are the same — it is **intra-sample window overlap** (the e window (t−τ_e,t) and the f_est window (t−3τ_f,t) both end at the current moment and share the h2→output spike process), not a boundary effect;
3. Longer samples lower loss through the error channel SNR ∝ √ΔT, unrelated to the covariance;
4. **Note**: the current implementation has zero delay. If a real forward-backward delay τ_delay is added, the windows separate and the covariance is left with only a boundary residue ∝ τ_delay/ΔT — user's view: in real runs ΔT ≫ τ_delay, so the covariance is negligible. This claim holds.

### 4.5 R×ΔT grid sweep (added 2026-08-13, full data in SWEEP_RESULTS_CN.md)

`R ∈ {100,200,400,800} × ΔT ∈ {0.5,1,2,4}s`, 2000 samples per configuration (scaled mode: R and f_da/TARGET/BIAS/R_GATE are co-scaled, mean-field dynamics unchanged, only the Poisson noise level changes; plus a fixed control: only R changed). Key points:

1. **The loss plateau decreases monotonically with R** (ΔT=2: 0.136→0.116→0.063→0.018, acc 0.58→1.00); **at R=800, ΔT=1~2 it reaches loss≈0.017, acc≈1.00 (4/4 converged) within 2000 samples** — the SDE-error version converges stably for the first time;
2. ΔT's effect is masked by weight jitter at low R and only shows at high R (R=400: ΔT=0.5→4 drops loss 0.108→0.017); R=800/ΔT=4 shows late-training p saturation oscillation (the last-500 mean overtakes ΔT=2, but best value 0.0078 is still the grid-wide minimum);
3. **corr(e,δ) decays monotonically with R from +0.12 (R=100) to ≈0 (R=800)** (mechanism: once training is complete, δ for y=1 flips around 0, diluting the mixture correlation); W1 ≤ +0.04, W2 ≈ 0 unchanged; it barely varies with ΔT and training time — the "covariance is negligible" conclusion holds across the whole parameter space and is stronger at high R;
4. The p sign consistency rate ≈ 0.31~0.41 does not improve with R: per-sample signs remain noise-dominated, and the gradient shows only at the expectation/cumulative level (consistent with the construction);
5. fixed control: changing only R is much worse (R=800: 0.034 vs scaled 0.018), verifying "R and f_da must be scaled up together, SNR ∝ √R". The normalization test loss·√(kΔT) is not constant → the plateau falls faster than the pure-noise 1/√(kΔT) prediction.

---

## 5. Troubleshooting History (three root bugs + one scaling bug — all fixed)

1. **Topology disorder (W2 layer)**: `pre2` previously used `np.repeat` (16-long block) misaligned against `post2`'s `np.repeat(...,17)` (17-long block) — the bias was amplified 16× and the whole W2 path was scrambled. Fix: `pre2 = np.tile(...)`. All earlier results were built on the wrong network.
2. **Incorrect residual mean-field model**: the identity path physically injects into the same IF membrane (sum first, then threshold), so the rate-domain function is `a2 = ReLU(z2 + a1/θ)`, not the MLP-style `a1 + ReLU(z2)`; the identity edge's effective weight = **1/θ**; the chain-rule identity error term = `δ_h2/θ` (which is `+δ_h2` when θ=1).
3. **τ scaling of f_est/r_est**: a leaky integrator's steady state is E[f] = τ·rate; the target unit was wrong by 10× (the "learning stalls" illusion). Fix: rate = f_est/TAU_F.
4. **Deep error attenuation at θ=20** (exploratory): the propagated error δ ∝ θ^(−depth), deep learning is killed by θ — reverted to θ=1 (after the topology fix, θ=1's forward equivalence was verified unit-by-unit by test_forward).

### Nature of the mechanism verification (important)

`verify_mechanism.py` previously gave cos(g, −E[Δw]) = 1.0000 and 100% sign consistency — this is an **algebraic identity** (both sides share the same δ and gating code); it only verifies the algebra of "local rule + correct δ ⟹ gradient descent", **not** an empirical measurement of the actual noisy update. The empirical measurement (verify_trajectory.py) shows: over a 300-sample window the actual-update vs gradient cosine is only ~0.05-0.1 (per-sample noise ~15-30× the signal).

---

## 6. Confirmed Correctness

- The spiking forward matches the rate-domain mean-field unit-by-unit (test_forward.py, θ=1, including the residual 1/θ correction);
- The constructed expected update = −η·∂L/∂w (algebraic identity);
- The mean-field δ version (before being replaced by the current SDE-error version) converged 4/4 on XOR at 10000 samples across three seeds:
  seed0: (0,0)=0 ✓ (0,1)=159 ✓ (1,0)=176 ✓ (1,1)=0 ✓; seed1: 228/257/0; seed2: 248/212/32 ✓
- The current SDE-error version converges slowly at default parameters (R=200) (3/4 in 5000 samples); **raising R to 800 (f_da=2000 co-scaled) reaches acc=0.99~1.00 within 2000 samples** (see §4.5), consistent with the "error-channel noise dominated" conclusion.

---

## 7. Helper File Status (important)

| File | Status |
|---|---|
| verify_mechanism.py | ✅ usable (algebraic verification; consistent with the current main-file model) |
| measure_covariance.py | ✅ usable (recent artifact) |
| test_forward.py | ⚠️ usable but contains debug prints (DEBUG step 1234, k2acc, etc.); needs cleaning if used |
| verify_trajectory.py | ✅ usable (empirical cosine measurement; includes a two-segment run) |
| sweep_rd_dt.py | ✅ usable (R×ΔT sweep main script; `docseed` mode matches measure_covariance.py item-for-item) |
| plot_sweep.py | ✅ usable (sweep plotting and summary table, reads sweep_out/) |
| show_windows.py | ✅ usable (prints per-configuration window covariances, small utility) |
| SWEEP_RESULTS_CN.md | ✅ full Chinese-annotated sweep results (topology/covariance/loss plateau, citable for the paper) |
| sweep_out/ | ✅ sweep data: sweep_summary.csv (19 configs), two figures, cfg_*.npz curves |
| mnist_loader.py | ✅ usable (MNIST idx-format loading, >u4 big-endian) |
| mnist_shallow.py | ✅ usable (locally connected version, three downsampling modes: stride/avg/max; **LOSS parameter supports mse/ce cross-entropy**, argv[13-14]=LOSS/TAU_SM) |
| mnist_shared.py | ✅ usable (**shared-convolution version**: standard conv kernel sharing, update = within-group local signal sum; argv[11]=KAPPA output-layer subtractive inhibition pool u[OUT]-=κ·Σf̂·dt, argv[12]=GAMMA lateral inhibition, prints out-rate diagnostics every 500 samples; **from 2026-08-15 saves a checkpoint every 2000 samples**, no longer only at the end of training; full argv: `<seed> <N> <ΔT> <R> <α> <FDA> <BIAS> <TARGET> <final_eval> <DT=0.02> <KAPPA=0> <GAMMA=0>`) |
| mnist_conv_snn.py | ✅ usable but slow (two-conv version, ~40 ms/step, control only) |
| verify_mnist_shallow.py / verify_mnist_shared.py | ✅ usable (gradient FD verification, four topologies) |
| mnist_table.py | ✅ usable (parameter-table measurement script, outputs per-sample gradient alignment/variance/SNR/expected-update efficiency/covariance/loss plateau/frozen acc → mnist_table_results.csv; **extended 2026-08-15**: CSV gains KAPPA/GAMMA/DT columns, argv[12]=DT supports R-amplified configs) |
| mnist_ce_probe.py | ✅ usable (CE-loss quick probe, added 2026-08-13) |
| inspect_ce_ckpt.py | ✅ usable (CE post-training weight/rate inspection, added 2026-08-13) |
| runs/stride50k, runs/shared30k | ✅ overnight long-run directories (copied scripts + mnist_data junction, each with its own log) |
| eval_meanfield.py / eval_meanfield_shared.py / compare_rates.py / measure_terms.py / measure_dt_accum.py / per_sample_audit.py / measure_accumulation.py / measure_cov2.py / dbg_*.py | ✅ diagnostic tools (mean-field eval/rate comparison/term audit/accumulation audit/covariance sweep/debug) |
| MNIST_RESULTS_CN.md | ✅ full Chinese-annotated MNIST experiment record (topology decisions/verification/debug history/findings/result tables) |
| mnist_runs_summary.csv | ✅ summary of all MNIST run configs and frozen-test acc (ready to use; **some rows in the inhibition-pool series pending, see last update**) |
| PROJECT_LOG.md | ✅ project log (created 2026-08-14: timeline, decisions, run-command quick reference, data-file index, for handovers at any time) |
| gd_meanfield_shared.py | ✅ usable (**added 2026-08-14 night**: mean-field GD training script, measures the noise-free theoretical ceiling; argv: `<LR> <N> <KAPPA> <SEED>`, rates_of contains the κ self-consistent solution over 8 iterations, ana_grad analytic gradient, periodic + final n=1000 evaluation; N>60000 auto-repeats epochs; 300k samples ~8.5 min) |
| exp/ | ✅ 2026-08-14 inhibition-pool series experiment directory (base…k0p2_a10_100k; each dir has run_log.txt + checkpoint + figure; the R-hypothesis 4 dirs were interrupted by shutdown) |
| exp3/ | ✅ 2026-08-15 batch: k0p2_a15_s2_20k (seed2 final 0.272), r400_a3p75_3k (0.122, R hypothesis negated), r800_a0p9375_2k (output deadlock termination), r800_k0_a3p75_2k (0.106) |
| tables_20260815/ | ✅ 2026-08-15 second batch of parameter tables (N=3000 × 4 configs, completed and merged into the main CSV) |
| eval_checkpoints_batch.py | ✅ usable (batch checkpoint evaluation: frozen spike n=1000 + mean-field n=1000, outputs `mnist_checkpoint_eval_batch.csv`) |
| TABLE_RESULTS_SHARED_CN.md | ✅ markdown four tables of all shared-convolution parameter tables (overall/gradient-angle/variance SNR/covariance, generated by `make_tables.py`) |
| EXPERIMENT_PROGRESS.md | ✅ current progress at a glance (for handovers at any time: completed/running/next steps) |
| meanfield_ckpts/ | ✅ mean-field GD 300k × κ=0/0.1/0.2/0.4 completed (0.965/0.965/0.965/0.961, checkpoint every 25k); used for capability proof (reset reads 0.94-0.95), not a delivery route |
| exp4/ | ✅ **0.8 achievement batch** (2026-08-16 early morning): reset-protocol spike training reset_a15_20k (0.767)/ reset_a10_30k (0.778)/ **reset_a15_cont (0.824)** |
| mnist_train_log.txt / mnist_long_log.txt / mnist_shared_log.txt | ✅ full logs of each run |
| mnist_checkpoint.npz / mnist_result.png | weights and figure of the most recent run |
| clean_gd_xor.py | ❌ **deprecated** (moved to archive/old_scripts/): old model (np.repeat disorder + a1+ReLU(z2)), results untrustworthy |
| noise_isolation.py | ❌ **deprecated** (moved to archive/old_scripts/): old model, same as above |
| xor_residual_local_bp.png | figure of the old version (mean-field δ version), removable |
| xor_local_bp_result.png | figure of the current version |

---

## 8. Next Directions (usable for the paper's discussion/improvement)

1. ~~Raise the rate scale R (SNR ∝ √R)~~ ✅ **completed (2026-08-13)**: empirically confirmed at R=400/800, loss plateau dropped as expected (acc=1.00 within 2000 samples at R=800); **conclusion: R must be scaled up together with f_da**; raising f_da alone is ineffective or even harmful (§4.5 and SWEEP_RESULTS_CN.md);
2. Bursting/regularized firing (Fano<1) to equivalently raise the error-channel SNR;
3. A better eligibility-trace function (e.g. double-exponential, calcium dynamics) to lower the e noise;
4. Add a forward-backward delay τ_delay (paper §8.4/8.5) to verify that the W3 covariance drops with τ_delay (window separation), and observe the boundary residue ∝ τ_delay/ΔT;
5. Sensitivity of the output-layer bias initial p and the gate threshold R_GATE;
6. Longer training + seed sweep to give convergence statistics: **partially complete** — 4/4 converges within 2000 samples at R=800, ΔT=1~2; low R still needs longer training; note the p saturation oscillation at R=800/ΔT=4 (to assess whether it reflects real behavior near the boundary);
7. ~~MNIST ultra-long training to test the deep-gradient-dilution hypothesis~~ ✅ **completed (2026-08-14 morning)**: **hypothesis negated** — the stride version over-trains and degrades at 50k samples (peak 0.285@3-4k, final 0.124, mean-field 0.326 but poor noise robustness); **the shared-convolution 30k samples genuinely accumulate (0.202, 0.285 evaluated at @30k, mean-field 0.230)** — the shared version is the clear scale-extension direction;
8. **MNIST next steps**: shared-convolution long training 50-100k samples (expected 0.3+); seed sweep on the shared version; study the stride version's "good mean-field but bad spike" boundary-thinning mechanism (marginality/noise robustness, paper-discussion material);
9. **Output-layer independent loss (user's plan, evaluated 2026-08-14: feasible, essentially the current MSE mode)**: each output computes its own loss independently, injected simultaneously as a δ vector (one δ per output), without softmax shared normalization. Evaluation conclusion:
   - The current MSE mode **is** this plan: δ_out[k] = f̂_k − TARGET·[k=y], 10 δs computed independently, injected vector-wise each step, each responsible for its own target;
   - **CE's root cause of failure is precisely abandoning the independent loss**: softmax normalization ties the 10 outputs into a shared distribution → zero-sum δ → FC fan-in sums cancel each other (d_fc ≈ w3[argmax]−w3[y]) → deep starvation. The independent loss has no such problem (the correct output −447 and the 9 wrong outputs +581 inject independently, no cancellation) — this explains why MSE (0.202) is structurally far better than CE (0.11);
   - **Note**: the independent loss **cannot by itself solve** §4.6's deep gradient dilution (the 9 wrong outputs fire freely, and their random contributions in the fan-in sum still ≈ 0.77× the correct signal) — it needs to be combined with scaled R (SNR∝√R, to be run) or output-layer competitive normalization (lateral inhibition, optional enhancement);
   - Enhancement candidates (if done): wrong-output independent targets >0 (control free firing rate), margin-style (wrong outputs are penalized only when they exceed the correct output);
   - Recommend adding to the paper's discussion the "effect of the loss function's competitive structure (zero-sum shared vs independent vector) on deep signals" — the empirical CE-starved vs MSE-usable contrast;
10. **Output-layer lateral inhibition (user's direction, decided in discussion 2026-08-14, implementation pending)**: the deep-starvation mechanism clarified — the firing/fighting problem **exists only at the output layer**:
    - Only the output layer has "multiple independent targets" (10 outputs each computing its own loss); the deep (FC/conv) δ is a fan-in forwarding of the output instructions — it does not itself generate weakening instructions, it only "receives the fight's outcome";
    - Only the output layer has "anchor-free freely-firing units": the 9 non-target outputs have no target constraint (bias synapse P=0.6 feeds continuously), firing freely → every sample continuously sends weakening instructions (measured +581 vs correct −447, 0.77×), fighting the correct instruction at the FC fan-in → deep starvation;
    - Lateral inhibition is applied at the **output layer**: structurally suppress the wrong outputs (not learning-dependent, effective from initialization) → the weakening instruction immediately weakens → the deep-starvation window shrinks dramatically; the correct output becomes competitively strengthened;
    - Two physical implementation options (both within the paper's locality framework): ① output-layer lateral inhibition connections (same-layer local hard-wired); ② shared inhibition pool/group rate normalization (global broadcast channel, of the same kind as f_da, allowed by the paper §2);
    - Direct verification metric: the deep alignment align(W1/W2) (currently 0.1-0.2) should rise notably with lateral inhibition on/off;
11. **Output competition mechanism selection: global normalization (recommended, evaluated 2026-08-14, implementation pending user confirmation)**:
    - Candidate A **global normalization** (divisive normalization, the biological counterpart of the Carandini–Heeger normalization circuit): a wide-field inhibitory neuron collects population activity, and each output receives inhibition proportional to the total: f̂'_i = f̂_i/(1+κ·Σf̂) (using the drive as the denominator also works). Properties: total rate auto-clamped to ~1/κ (fixing "correct output drifting to the rate ceiling"), one parameter κ, a global broadcast channel of the same kind as the paper §2 and f_da/C, effective from initialization without learning;
    - Candidate B **lateral inhibition** (the cortical local GABA circuit counterpart): pair-wise lateral inhibition within the output layer, whoever is strong suppresses the other (WTA tendency). Properties: relative competition directly forms discrimination, but mutual inhibition reaches the correct output (interaction with the TARGET anchor needs tuning), the γ parameter is sensitive to tune, absolute rate remains unbounded, and there is oscillation risk;
    - **The distinction from CE (critical warning)**: global normalization is the linear relative of softmax (exponential divisive normalization); CE failed because δ used the softmax zero-sum gradient → FC fan-in cancellation. The correct approach: **normalization acts only on the forward rate level, δ still injects an independent MSE vector** (each output its own target), not touching the gradient structure → no zero-sum cancellation;
    - Implementation points: ① f̂' enters δ (δ_i = f̂'_i − TARGET·[i=y]), and TARGET must be recalibrated against κ (target share ≈ TARGET·κ); ② add κ as a command-line argument; ③ first do FD verification + forward equivalence (reuse the verify_mnist_shared.py pattern); ④ control experiments: normalization on/off × deep alignment align(W1/W2), frozen acc, weakening-instruction magnitude (Σ_{k≠y} f̂'_k should trend to zero with training); ⑤ the paper discussion can present both (biologically they coexist), the experiments do A first;
12. **Output-layer inhibition pool implemented and verified (2026-08-14 afternoon, notable results)**: mnist_shared.py gains KAPPA (argv[11], u[OUT] -= κ·Σf̂·dt)/ GAMMA (argv[12], lateral inhibition). 3k-sample frozen acc (n=500):
    - κ monotonic: 0→0.134, 0.05→0.150, 0.1→0.158, 0.2→0.192, 0.4→**0.280** (prior best 30k samples 0.202 → sample efficiency ×10);
    - **Lateral inhibition negated** (γ=0.1→0.114, mutual suppression hurts the correct output); **scaled R negated** (R=400 co-scaled→0.106);
    - Mechanism evidence: the κ version's output total rate is higher (~1000 vs ~400), the correct-output rate climbs (y=299 vs y=0) → the weakening instruction is suppressed, the correct output wins the competition, and the deep layers start learning;
    - 10k long run under verification (κ=0.2/0.4 × 10k, κ=0.2+ΔT=2, κ=0.2 seed=1); detailed data to be written into MNIST_RESULTS_CN.md;
    - **Later long-run results (2026-08-14 evening)**: κ=0.2@10k → **0.292** (new record, sample efficiency ×3 vs the old 30k best 0.202); κ=0.3@10k → 0.224; κ=0.4@10k → 0.170; κ=0.2+ΔT=2@5k → 0.172 (ineffective under the same physical time); κ=0.2 seed=1@3k → 0.200 (robust);
    - **Crash phenomenon (key limitation)**: κ=0.2@20k peaks at ~11-13k (0.295-0.31) then runs away (y=4212 Hz >> TARGET, other=0) → all-dead deadlock at ~19.5k (total=2, final 0.126). The subtractive inhibition pool structurally deadlocks: anchor-free + ReLU truncation + mutual dependence → an all-dead fixed point; the larger κ, the earlier the crash;
    - Mechanism verification (table measurement): W3 E[d] −71.5 → −29.7 (weakening instruction suppressed by 58%), E[e] 2.1 → 10.9 (5× more active), signal product ×2.2; corr(e,d) still ~0;
    - **Anti-crash experiments running**: κ=0.2×TARGET=500, κ=0.2×α=1.5e-8, κ=0.15@20k, κ=0.2 seed=1@10k; candidate schemes: inhibition-current reference anchor S_ref (max(Σf̂−S_ref,0)), or accept early stopping (peak window 3-13k);
13. **Mean-field GD theoretical ceiling (2026-08-14 night, gd_meanfield_shared.py) — key positioning**:
    - Noise-free (expected-gradient) training: LR=1e-8 @30k → κ=0: 0.900 / κ=0.1: **0.916** / κ=0.2: 0.915 / κ=0.3: 0.905 / κ=0.4: 0.913; **300k samples κ=0.1 → 0.965** (8.5 min); N>60000 needs repeated epochs (fixed);
    - **Conclusion**: the current small architecture (18,898 parameters) has a theoretical ceiling of ~0.96, so the user's target of **70-80% (proving the theoretical framework works, not SOTA) is entirely within the architecture's capability**; current spike 0.348 vs mean-field ~0.9 → **the ~2.5× noise loss is the only bottleneck; the route = make the spike approach the mean-field**;
    - User's theoretical confirmation (no engineering limits on the simulation, free to proceed): **SNR ∝ √R** (scaling R together with δ/f_da/e up by a factor K: signal∝K², variance∝K³ → SNR∝√K); **shrinking dt does not directly raise SNR** (in the continuous limit the update is a Riemann sum, independent of dt); it is a prerequisite for scaling R (keeping the λ·dt≤1 Poisson approximation valid) — this explains the earlier scaledR400 failure (TARGET co-scaled to 2000 slows output differentiation + λ·dt untouched);
    - **R-hypothesis test completed (2026-08-15, negated on MNIST)**: R=400 (κ=0.2, α=3.75e-9, dt=0.01) final 0.122; R=800+κ=0.2 output-deadlock termination; R=800+κ=0 retest final 0.106 — XOR's SNR∝√R benefit does not auto-transfer to MNIST (10 independent output targets + deep fan-in break the scaling). **R amplification must be recalibrated together with TARGET/KAPPA/output anchors**; the two 100k long runs were still interrupted by shutdown; mean-field 0.965 shows the small architecture needs no enlargement to reach 70-80%;
    - **✅ User's new target achieved (2026-08-16 early morning)**: frozen-test acc ≥ 0.8. Key mechanism = **cross-sample state pollution** (u deep-negative locking carries across samples; 0.96-mean-field weights spike-read only 0.2); with **sample-boundary reset** (training + evaluation protocol, physically realizable): mean-field weights spike-read 0.94-0.95 (capability proof), same-budget spike training 0.348 → 0.767, continued to 40k → **0.824** (single 0.817 / long-window 0.848 / counting 0.850, n=1000). The K-accumulation route was rejected by the user (neurons have no storage capacity); the 4-level R-amplification sweep showed no improvement (paused).

---

## 9. MNIST Experiments (2026-08-13, full record in MNIST_RESULTS_CN.md)

**Topology decision**: the real SOTA (Spiking ResNet + surrogate gradient)'s surrogate gradient is incompatible with this construction; **convolutional weight sharing does not violate locality (user correction 2026-08-13)** — biology reuses the same receptive-field set in time via eye movement/saccades, and the shared-weight update = sum of local signals at each position (= standard convolutional gradient). Two generations of topology:
- **v1 locally connected version** (`mnist_shallow.py`, stride/avg/max three downsampling modes): 784 → conv5x5x4 → FC32 → 10 (37k parameters)
- **v2 shared-convolution version** (`mnist_shared.py`): 784 → CONV5x5x4 stride2 → FC32 → 10 (**18,898 parameters**, only 104 shared kernels; each parameter aggregates 144 positional evidences → SNR ×12)

**Verified** (all passed, see MNIST_RESULTS_CN.md §2):
- Gradient finite difference 12/12 (three local modes + shared-convolution mode), forward rate equivalence, per-sample alignment cos=0.68~0.91
- Mean-field GD control: noise-free can reach test 0.42 (lr=1e-6, 4000 samples) — the architecture is learnable

**Key findings (paper-discussion material)**:
1. **Training acc is inflated**: in-sample transient adaptation (zero delay) pushes rolling acc to 0.4-0.5, but the frozen evaluation is only 0.10-0.16 — evaluation must be frozen;
2. **α must be scaled down with the rate scale**: MNIST gradients are 25-400× larger than XOR, so α=2e-6 is immediately at the saturation boundary → weights jump 0/1 → output burst → δ>FDA → f_err truncation → deadlock (the mechanism by which loss explodes to 19+); only the α~1e-7 order stabilizes;
3. **Deep covariance bias (direct evidence for the paper §8.5) + user-hypothesis verification**: corr(e,d) is small (≤0.04) but |cov|/|E[e]E[d]| reaches 0.28-1.20 (deep E[d]→0 amplifies the ratio). **User hypothesis: caused by an effective forward-backward delay that is too large; raising the carrier frequency or sample duration 10× → covariance ~3%. Control experiment: ΔT×10 drops corr to 0.000~0.025 (✓ confirmed); R×10 ineffective** (the time window is unchanged). Training configuration adopts ΔT=10;
4. **Per-sample SNR ~ 2-4 is the bottleneck**: f_da must be ≥ the deep fan-in δ (~3000 Hz), and the carrier noise √f_da suppresses the output-layer SNR; accumulation efficiency ~19% (120-sample cumulative/theoretical); shared convolution raises per-parameter SNR ×√144 through positional-evidence aggregation;
5. Cross-sample transient pollution (ΔT=1 is only 5τ_e) halves the alignment, but ΔT=4 does not improve the amplitude ratio — not the main cause;
6. **dt granularity does not affect the covariance (the user's second hypothesis, measured; see MNIST_RESULTS_CN.md §4.5)**: executed per the user's prescription (dt 20ms→1ms, λ·dt≤1 ≤1 spike per step, steps/sample ×20); corr(e,d) is completely unchanged (0.015/0.003/0.013) — the covariance is set by the physical time window; the "minimum delay" argument of dt×layers can only be tested after an explicit τ_delay implementation;
7. **Deep gradient dilution is the accuracy bottleneck (§4.6)**: initial δ correct output −447 vs sum of 9 wrong outputs +581 (0.77×) — the FC fan-in is dominated by wrong outputs, deep features are hard to form, ~50k~2013100k samples needed. (**2026-08-14 update**: the 50k long run proves the local version over-trains and degrades rather than continuing to learn, see below)

**Status**:
- XOR: 4/4 converges within 2000 samples at R=800/ΔT=1~2 (acc≈1.00); default parameters 3/4 in 5000 samples;
- MNIST (MSE loss): stride local version 10k samples → frozen test **0.202**; extended to 8k of the 30k it stalls at the ~0.20-0.25 plateau; shared-convolution version 2000 samples (ΔT=10) → **0.110**; the ~3% covariance is achieved (ΔT=10);
- **Cross-entropy mode finished (2026-08-13 night, formal 3000 samples + 2 probes) — negative result**: frozen acc 0.095~0.140; mechanism = unbounded rate → softmax saturation + zero-sum δ making FC fan-in cancel → deep hunger (W1/W2's P barely moves); see MNIST_RESULTS_CN.md §5.1. CE code is kept as a negative control; the MNIST main result still uses the MSE form;
- **Overnight long runs complete (2026-08-14 morning, see MNIST_RESULTS_CN.md §5.2/5.3)**:
  - `runs/stride50k/`: stride MSE 50k samples → **over-training degradation**: acc peaks 0.285@3-4k, permanently drops to 0.09-0.16 after 13k, final 0.124; mean-field evaluation 0.326 (the weights themselves are learnable but poor in noise robustness, spike/mean-field ratio 0.38); the "50k~2013100k samples breakthrough" hypothesis **negated**;
  - `runs/shared30k/`: shared-convolution MSE 30k samples → **genuine cumulative learning**: 0.08→0.285@30k (n=200), final 0.202 (n=1000); mean-field 0.230, noise loss only 12%;
  - **Parameter-measurement table of 10 configs complete** (`mnist_table_results.csv` + four tables in MNIST_RESULTS_CN.md §5.3): per-sample gradient alignment, update SNR/variance, expected-update efficiency (3-20%), per-layer covariance (corr≤0.05 holds across all configs), loss plateau, frozen acc;
- **Output-layer inhibition pool (2026-08-14, see MNIST_RESULTS_CN.md §5.4) — current best configuration**:
  - `mnist_shared.py` argv[11]=KAPPA (u[OUT] -= κ·Σf̂·dt, subtractive inhibition pool), argv[12]=GAMMA (lateral inhibition, **negated** γ=0.1→0.114);
  - **κ=0.2 + α=1.5e-8 @20k → 0.348** (n=500, current spike best); seed=1 reproduces 0.334; κ=0.2+α=1e-8@30k → 0.330;
  - Crash phenomenon: κ=0.2 with full α runs away at 10-13k (y=4212 Hz >> TARGET) → all-dead deadlock; **halving α postpones the crash point to ~20k+** (the correct anti-crash fix); κ=0.3/0.4 degrade faster;
  - Mechanism (table measurement): W3 E[d] −71.5→−29.7 (weakening instruction suppressed 58%), E[e] 2.1→10.9, signal product ×2.2; corr(e,d) still ~0;
- **Mean-field GD (noise-free theoretical ceiling, gd_meanfield_shared.py)**: κ=0.1 @300k samples → **0.965** (8.5 min; κ sweep at 30k: 0→0.900, 0.1→0.916, 0.2→0.915) — **architecture capacity confirmed 0.95+, the user's 70-80% target fully reachable, all the gap is spike noise** (current 0.35 vs mean-field 0.9, noise loss ~2.5×);
- **R-hypothesis test complete (2026-08-15, negative result)**: R=400 κ=0.2 → 0.122; R=800 κ=0.2 output-deadlock termination; R=800 κ=0 retest → 0.106 — XOR's √R benefit cannot transfer directly to MNIST; the two 100k long runs (k0p2_a15_100k / k0p2_a10_100k, ~10h) were interrupted by shutdown mid-run and need reruns;
- **Best-config seed sweep (2026-08-15)**: κ=0.2, α=1.5e-8, 20k → seed0/1/2 = **0.348/0.334/0.272** (mean 0.318); the current spike plateau is ~0.32-0.35, the mean-field ceiling 0.965;
- **Parameter-table extension (2026-08-15)**: mnist_table_results.csv gains KAPPA/GAMMA/DT columns; N=1000×5 configs + 08-14 N=3000×2 configs merged; the second batch N=3000×4 configs is in tables_20260815/;
- **✅ 0.8 target achieved (2026-08-16 early morning)**: sample-boundary reset protocol (`mnist_shared.py` argv[13]=1) + per-step in-place local update (the paper's SDE framework) — `exp4/reset_a15_cont` (20k@1.5e-8 + 20k@7.5e-9) final **0.824** (n=1000), independent evaluation single 0.817 / long-window 0.848 / counting 0.850; reset_a15_20k=0.767, reset_a10_30k=0.778. **Key mechanism: cross-sample state pollution (u deep-negative locking carried across samples) is the biggest bottleneck on spike readout/training**; diagnostic script `dbg_pulse_output.py`; capability proof: the mean-field 300k weights reset-read 0.94-0.95. K accumulation rejected by the user; the 4-level R-amplification sweep showed no improvement (paused).
