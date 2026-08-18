# Project History & Milestone Timeline

> A concise, chronological record of the project's daily work, key findings, decisions,
> incidents, and final numbers, condensed from the Chinese work log (`docs/PROJECT_LOG.md`)
> for an international readership.
> **Terminology**: every project-specific term is written with its authoritative English term
> defined in [GLOSSARY.md](GLOSSARY.md) (e.g. recalibration, eligibility trace, cross-sample
> state pollution, frozen evaluation, multi-seed acceptance, mean-field, push/pull imbalance,
> κ inhibition pool, LIF membrane leakage, ISI/silent interval). Terms are self-explanatory at
> first occurrence in reading order.
>
> Scope: 2026-08-11 → 2026-08-18. Goal posts: the paper's online local-learning / SDE
> framework, first validated on XOR, then taken to MNIST.

---

## 2026-08-11 — XOR residual simulation completes

- The main XOR residual-topology simulation (`xor_residual_local_bp.py`, SDE error version)
  converged on the default 5000-sample 3/4-pattern schedule.
- Debugging history resolved: topology ordering bug (`np.repeat`), the residual mean-field
  model, τ calibration, and deep decay at θ=20 — all fixed (see `PROJECT_DOC.md` §5).
- Supporting verification tools built: `verify_mechanism.py` (algebraic identity) and
  `verify_trajectory.py` (empirical cosine).

## 2026-08-12

- No major recorded item; continued carrying forward the measurement tooling from the prior day.

## 2026-08-13 — Sweep completes; MNIST experiment launches

- **R×ΔT grid sweep** (`sweep_rd_dt.py`) done: loss plateau falls monotonically with R;
  R=800 / ΔT=1–2 reached 4/4 convergence within 2000 samples (acc≈1.00); corr(e,d) decays to
  ≈0 with R; conclusion established that "R must be scaled up together with f_da".
- **MNIST launch**: data loading (`mnist_loader.py`, ossci mirror); topology decision —
  local-connection v1 + shared-convolution v2 (the user corrected that "weight sharing does
  not violate locality", which became the **shared convolution** route).
- Gradient FD 12/12, forward equivalence, and the mean-field GD control (0.42 @ 4000 samples)
  all passed.
- Key training-behavior findings: **in-sample transient adaptation** makes training accuracy
  illusory high; α needs ~1e-7 scale; covariance hypothesis verified (ΔT×10 → corr ~3%);
  dt granularity does not affect covariance; deep gradient dilution by 0.77×.
- Early results: stride 10k → 0.202; shared conv 2k (ΔT=10) → 0.110.
- Evening: CE (cross-entropy) mode implemented (LOSS/TAU_SM), FD 12/12 passed; first test
  stopped by the user.
- Night (user resting): formal CE run of 3000 samples + 2 probes → **negative result**
  (frozen 0.095–0.140); mechanism = unbounded rate → softmax saturation + zero-sum δ
  cancels FC fan-in → deep-layer starvation.
- Started overnight long runs: stride 50k (`runs/stride50k`), shared 30k (`runs/shared30k`).

## 2026-08-14 — Long runs, κ inhibition pool, first crash, mean-field upper bound

- Morning, two long runs done: **stride 50k → 0.124 (overtraining collapse)**: peak 0.285
  @3-4k, then permanently stuck at 0.09–0.16 after 13k; mean-field (noise-free) 0.326 —
  weights learnable but poor noise robustness (spikes/mean-field = 0.38). **shared 30k →
  0.202** (n=1000), @30k cycle eval 0.285 (real cumulative learning); mean-field 0.230
  (noise loss only 12%). Weight sharing decides "cumulative vs degenerate" behavior.
- Parameter measurement table (`mnist_table.py`, 10 configs): gradient angle output 0.29–0.66,
  deep 0.1–0.2, decaying with training; updateSNR 0.04–0.26 (noise 4–70× signal), expected
  update efficiency 3–20%; corr≤0.05 across configs, ΔT=4 pushes to ~0.004 — supports the
  independence hypothesis; loss plateau + frozen acc (α peak ~2e-7, TARGET=1000 optimal;
  R=400 without scaled f_da fails at 0.104).
- Diagnosed the "where does the bottleneck lie" question: per-sample SNR too low + deep
  starvation (9 wrong-output fan-in dominant) + in-sample transient adaptation illusion — not
  an implementation bug (FD 12/12, mean-field control, XOR convergence all normal).
- **Afternoon (user out, autonomous experiments)**: implemented output-layer suppression in
  `mnist_shared.py` — **κ inhibition pool** (Carandini–Heeger-style shared subtractive pool,
  argv[11]) and GAMMA lateral inhibition (argv[12]).
  - κ sweep @3k frozen (n=500): κ=0→0.134, 0.05→0.150, 0.1→0.158, 0.2→0.192, 0.4→0.280 —
    inhibition pool monotonically helpful; **lateral inhibition rejected** (γ=0.1→0.114 < base);
    **scaled-R rejected** (R=400 sync → 0.106).
  - 10k runs revealed a **κ-dependent differentiation–generalization balance**: κ=0.2@10k
    → **0.292** (new record, ×3 sample efficiency over the old 30k best of 0.202); κ=0.4
    degenerates (0.280@3k → 0.170@10k, winner-take-all, deep layer never catches up); ΔT=2 no
    benefit; seed 1 robust (0.200@3k).
  - Table mechanism evidence (κ=0 vs κ=0.2): W3 E[e] 2.1→10.9 (5× activity), E[d] −71.5→−29.7
    (weakening instructions cut 58%); corr(e,d) stays ~0 (independence unaffected).
  - **Crash (important) — structural deadlock of the subtractive inhibition pool**: κ=0.2@20k
    peaked at 11–13k (0.295–0.31), rate ran away at ~16.5k (y=4212 Hz >> TARGET), full
    extinction deadlock by ~19.5k (total=2, test 0.105). Mechanism: winning output runs away →
    S rises → inhibition strengthens → all other outputs die → δ≈0 → no learning recovery →
    layer collapses. Larger κ collapses earlier. Best window: κ=0.2 × 3–13k samples (0.29–0.31).
  - **Crash-prevention result (night) — halving α is the fix**: κ=0.2 + α=1.5e-8@20k → **0.348**
    (new record, n=500); TARGET=500 ineffective (0.164); seed replication: κ=0.2+α=1.5e-8 seed=1
    → 0.334 (vs seed=0 0.348); full-α crash ~10–13k, halved-α pushes to ~20k+.
- **Mean-field (noise-free theoretical upper bound)** via new `gd_meanfield_shared.py`: κ scan
  @30k → κ=0.1 gives **0.916**; **300k samples κ=0.1 → 0.965** (8.5 min). User target 0.7–0.8
  is well within architecture capability; the entire gap is spiking noise (current 0.35,
  ~2.5× noise loss).
- User theoretical confirmation (night): **SNR ∝ √R** (signal ∝ K², variance ∝ K³); shrinking
  dt alone does not raise SNR in the continuous limit but is the precondition for R scaling
  (keeping λ·dt≤1 for the Poisson approximation); longer ΔT gives SNR∝√ΔT but no net gain at
  fixed wall-clock time.
- R-hypothesis runs interrupted at shutdown (r400/r800 trending inconclusive); 100k runs lost
  (no checkpoints). Overnight takeaway: pulses saturate ~0.33–0.35 at 20–30k; mean-field 0.965
  → the route is to make pulses approach mean-field; the R-hypothesis needs a full test.

## 2026-08-15 — Environment rebuild, 0.8 target set, reset-protocol breakthrough → 0.824

- Environment rebuilt (Python3.14 with no numpy/matplotlib; `python3-numpy`, `python3-matplotlib`
  apt-installed; all scripts run via `python3`, not `python`).
- **Hardened checkpoints**: `mnist_shared.py` gained `save_checkpoint()` (every 2000 samples →
  `mnist_checkpoint.npz`; previously weights were lost if killed mid-run); `mnist_table.py`
  gained `[DT]` (argv[12]) for R-scaling/dt-shrinking measurements.
- Morning experiment batch (in `exp3/`): best-config seed=2 reproduction → **final 0.272**
  (n=500); seeds 0/1/2 = 0.348/0.334/0.272, mean 0.318, std 0.033. R-hypothesis runs rejected:
  r400 (α scaled 1/R²) → 0.122, output layer repeatedly went fully silent (total=0/1);
  r800+κ=0.2 deadlocked (total=0) and was terminated; r800+κ=0 → 0.106. **R hypothesis rejected
  on shared-convolution MNIST (3 negative results)** without synchronized output-anchor
  recalibration.
- Batch checkpoint evaluation (`eval_checkpoints_batch.py`, new; n=1000 frozen pulse + n=1000
  mean-field): e.g. k0p2_a15 pulse 0.308 / mf 0.299; crashed k0p2_20k 0.098 / 0.312.
- **The user set a new goal (evening): frozen test accuracy MUST reach 0.8; 0.3 is unacceptable.**
  The gap between the current pulse best 0.348 and the mean-field upper bound 0.965 is spiking
  noise; all subsequent work targets "make pulses approach mean-field".
- Mean-field preheating route started then stopped per user (no checkpoints saved until the
  easy fix); `gd_meanfield_shared.py` hardened to checkpoint every 25000 samples.
- **Second round (late night)**: four 300k mean-field GD runs completed (LR=1e-8, seed=0):
  κ=0/0.1/0.2 → **0.965**, κ=0.4 → 0.961 (n=1000), 21–22 min, ~1.1 GB peak memory.
- User rejected the K-sample local-update accumulation route (real neurons have no capacity to
  store K sample updates; the paper's core is the **online SDE framework** — in-place local
  updates each step, noise dominated by the long-time-averaged drift term).
- **Key capability finding**: mean-field 300k weights (noise-free acc 0.96) → frozen pulse
  readout only **0.115/0.164/0.203/0.175** (κ=0/0.1/0.2/0.4). The bottleneck is **not weight
  quality, it is the pulse forward readout** (noise loss ~5×, far above the earlier ~2.5×
  estimate). Long-window readout (SAMPLE_T=2) gave **zero improvement** — not a time-averaging
  problem. Diagnosis: κ normalization pins output rates to ~100–300 Hz, so within a 50-step
  (1 s) window each class emits only 2–6 spikes and argmax(f_est) is near random.
- **🚀 Breakthrough — cross-sample state pollution is the biggest killer of the pulse readout**:
  `eval_mf_batch.py` gained `--reset_each`; mean-field κ=0.2 weights jumped **0.203 → 0.947**
  (long-window 0.956, counting 0.954, near the noiseless 0.958). Mechanism: without reset, u is
  deep-negative-locked and this carries across samples; after reset each sample is independent.
  But reset **hurt** trained weights (exp/k0p2_a15 0.308→0.265) — training weights had
  over-fitted to the cross-sample continuous stream. Conclusion: the **training protocol must
  also reset at sample boundaries** (physically = state reset / silent interval before a sample,
  standard SNN protocol, within the paper's SDE framework).
- Added `RESET_PER_SAMPLE` (argv[13]) to `mnist_shared.py`; started reset-protocol pulse
  training (`exp4/`): **reset_a15_20k and reset_a10_30k**.
- **🚀 Historical acceleration with the reset protocol**: reset_a15_20k @2000 test 0.270 →
  @7000 0.560 (vs. history-best same-params no-reset: 0.225 @5000, 0.348 @20k); reset_a10_30k
  @7000 0.480; no degeneration observed.
- reset_a15_20k finished (n=1000): **0.767** (cycle peak 0.815@20k); continuation `reset_a15_cont`
  (from 20k ckpt, α halved to 7.5e-9, +20k) — **🎉 0.8 target met: final n=1000 = 0.824**
  (n=200 cycle 0.830). **Purely spiking, purely online local learning (paper's SDE framework),
  from random initialization, no weight injection** — protocol = in-place update each step +
  sample-boundary reset (physically implementable).
- Independent confirmation (n=1000, reset, κ=0.2): reset_a15_cont **0.817 / 0.848 / 0.850**;
  mean-field 4 ckpts reset readouts 0.942–0.953 / 0.955–0.960 / 0.954–0.961. Campaign table:
  `TABLE_0P8_CAMPAIGN.md`; documents updated (EXPERIMENT_PROGRESS, MNIST_RESULTS_CN).

## 2026-08-16 — Physically realistic protocol (LIF + ISI) replaces hard reset; recalibration → 0.877

- **The user rejected hard reset** as un-physical; the acceptable upper bound is "leave a
  silence between samples." The user flagged firing-rate residue (f_est/r_est) rather than
  membrane potential as the key, and located the old-draft paper `一生有爱何惧风飞沙.txt`
  with the full §4.5 LIF derivation.
- Diagnostic experiment (mean-field κ=0.2 weights, n=1000) **reversed the "firing-rate residue"
  hypothesis**: no-op 0.203; clear-rates only 0.205 (no improvement, τ_F=0.1 s << sample 1 s so
  rates already decay within-sample); clear-u only **0.948** — u (IF, no leakage, deep-negative
  lock persisted 100% across samples) is the main polluter; ISI=25/50 pure-silence ineffective
  (0.246/0.257). This also named **deep negative locking**.
- Old-draft §4.5: LIF firing formula `f_out = −λ/ln(1−λθ/I)` (I>λθ, else 0); IF is the λ→0
  special case; small-λ LIF≈IF (deviation λ/2 Hz + dead zone λθ, negligible).
- **Realistic protocol = LIF membrane leakage (paper §4.5) + ISI silent interval**
  (`mnist_shared.py` argv[14]=ISI_STEPS, argv[15]=TAU_M; u *= exp(−DT/TAU_M), no hard reset).
  Verification (mean-field weights, n=1000): τ_m=0.5+ISI=100 → **0.955/0.956/0.953**;
  τ_m=0.2+ISI=50 → 0.944/0.948/0.950; τ_m=0.5+ISI=0 → 0.936 (leakage alone suffices, ISI helps) —
  **all ≥ the hard-reset 0.947, via pure physical decay**.
- **LIF training initially failed** (reset protocol was at 0.615 same-epoch): (①) an
  implementation bug — the training loop's ISI silent steps used learn=True, so the quiet
  period produced negative δ updates that dragged output weights down; fixed to learn=False.
  (②) a physical mismatch — LIF transient ~3τ_m; with τ_m=0.5 a 1 s sample is only 2τ_m, so the
  in-sample average firing rate is ~57% of steady-state (learning signal mismatched to TARGET).
- Diagnosis (`dbg_lif.py`) found per-sample cosine (gradient alignment) is noise-dominated
  (σ~0.4) and not predictive; the real difference is the **eligibility–error covariance**
  corr(E₃,δ₃): IF (the protocol that trained to 0.824) +0.46 (favorable) vs LIF −0.16 to −0.25
  (harmful) — LIF's membrane delay puts E₃ anti-phase with δ₃. Also fixed a `dbg_lif.py` rand-branch
  bug (earlier "random-initialization" runs actually used trained weights).
- **Signal-level diagnosis confirmed the push/pull imbalance mechanism**: net update ≈
  push·(TARGET−f_correct)·N_correct − pull·f_wrong·N_wrong. Under random init, IF net +6514
  (2:1 positive, trains); LIF net −686 (~0, stalls). LIF output rate 1132 Hz/class ~2.2× IF
  (508/class), so wrong-class δ_wrong=+105 vs IF +52 — the wrong-class pull is ~4× stronger.
- **Recalibration breakthrough (probes P1–P7)**: **κ↑ (inhibition pool 1.0) presses all output
  rates down (1132→626 Hz/class) + TARGET↑ (2000→5000) amplifies the correct-class push**
  (net +5544→+8524). Probe P3 (TARGET=2000, κ=1.0) @2k tests 0.280, matching the reset baseline;
  **P6 (TARGET=5000, κ=1.0) @2k final 0.7160**; P7 (TARGET=10000) overfits (train 1.0/test 0.505)
  → TARGET=5000 is the sweet spot. **P6's 2k checkpoint under the acceptance protocol reached
  0.814/0.812/0.817/0.821 (n=1000) and 0.801/0.809/0.806/0.810 (n=2000) — all ≥ 0.8**.
- **Main run `lif5` (TARGET=5000, κ=1.0, τ_m=0.5, ISI=50, α=1.5e-8, N=20k, seed=0)** completed
  (12,977 s): 12k-cpeaks at **0.894/0.894/0.895** (mean-field 0.900), then overtraining collapse
  after 12–14k (TARGET saturates, push stops converging). The in-training single-implementation
  eval at 20k reported 0.49 vs acceptance 0.76 — a 28-point gap traced to **implementation
  sensitivity** (κ=1.0 pool → very low output rates near criticality, argmax fragile to Poisson
  implementation noise).
- **Acceptance protocol re-established as multi-seed frozen evaluation** (5 seeds × n=500,
  mean ≥ 0.8 = pass): lif5 12k=**0.873±0.017**, **14k = 0.877±0.007** (worst single seed 0.852),
  10k=0.824±0.015 (worst 0.800), 20k=0.750±0.021 (collapse confirmed). **Goal met and robust**;
  the uncalibrated κ=0.2/TARGET=1000 control (`lif2`) ended at **0.501** — recalibration
  (TARGET=5000/κ=1.0) is the decisive factor.
- **Seed replication**: s1 (seed 1, N=14000) → seeds[123,1,2,3,4] = 0.882/0.874/0.874/0.876/0.864,
  **mean 0.874±0.006**, all ≥ 0.864; s2 (seed 2) → **0.806±0.015** (all ≥ 0.784). Cross-seed:
  main 0.877±0.007 / s1 0.874±0.006 / s2 0.806±0.015 — pass, but s2 is ~7 points lower, noted as
  cross-seed variance.
- **Incidents & lessons** (2026-08-16 evening): the seed-launch script forgot to `cd` into the
  run directory, so s1/s2 checkpoints/plots collided in the repo root (s2 final lost → rerun);
  lesson — launcher scripts must `cd` into the run dir; background stdout must be redirected to a
  file (terminal pipes drop output); WSL/container rebooted overnight interrupting tasks
  (mitigated by `oom_score_adj=-800` + `supervise_overnight.sh` auto-restart).
- **Parameter sweep** (`mnist_lif_table.py`, per-sample training-replica with true gradient +
  online local update, sampling significant) — 16 configurations (9 main + reruns), 6 dimensions:
  - **TARGET non-monotonic**: frozen acc 0.404 (T3000) → 0.648 (T5000) → 0.372 (T7000); peak at
    the recalibrated 5000; corr_W3 rises monotonically with TARGET.
  - **κ alignment paradox**: κ=0.2 gradient alignment ALL **+0.764** (far above κ=1.0's +0.238)
    with higher expEff/signCons, yet lower frozen acc (0.414 vs 0.648) — good local gradient
    alignment does not imply good end-to-end results.
  - **ISI dual edge**: ISI=0 gives artificially low loss plateau (0.0136 vs 0.0494) but worse
    frozen acc (0.540 < 0.648) and alignment (0.166 < 0.238) — the silent interval improves both
    alignment and generalization.
  - Sweep analysis includes Spearman/Pearson cross-metric correlations and trajectory evolution;
    full results in `TABLE_LIF_PARAM_SWEEP.md`.
- **Project packaging & open-source phase (late 2026-08-16 → 08-18)**:
  - **Single-file paper demo** `mnist_demo_train.py` — double-click to train with live terminal
    visualization, four protocol configs (recal flagship T5000/κ1.0/τ_m0.5/ISI50 + uncal/reset/if
    controls), multi-seed frozen acc + checkpoint + 3-panel PNG (English captions). Packaged as a
    53.7 MB single ELF (`dist/mnist_train_demo`, PyInstaller 6.22.1 with bundled numpy/matplotlib/
    12 MB MNIST data) and a **Windows click-to-run folder** (`dist/windows_demo/`, 11.6 MB,
    fully offline, zero download).
  - **Git-ified the project** (`git init -b main`; first commit 4060954, 506 files, 8.5 MB);
    fixed root symlinks to relative paths; the user decided to make **everything public and
    traceable** ("every result and every process verifiable") — full-inclusion commit of 607
    files / 147 MB (90 .npz checkpoints, dist artifacts, MNIST data), within GitHub's 100 MB
    single-file limit.
  - Fixed a `mnist_demo_train.py` plotting crash (`--eval-every` vs 200-block mismatch; now eval
    strictly on `eval_every`, verified on 260/120 and 100/500 cases).
  - **Terminology standardization** (user request: every coined term explained at first
    occurrence): created `GLOSSARY.md` with 26 authoritative plain-language entries; added a
    12-term quick-view table to the reading-first doc; inserted the terminology notice block in
    all docs.
  - **Cognitive scaffolding**: new `BACKGROUND.md` (context & reading map: SNN training challenge
    → local learning vs STDP/surrogate-gradient/BPTT; the 5 questions; 7-node decision chain
    paper→XOR→MNIST→0.877; terms & docs maps; key-conclusions overview); reading order updated to
    GLOSSARY(0) → BACKGROUND(1) → EXPERIMENT_PROGRESS(2).
  - **Repo-upload change**: since GitHub is inconvenient from mainland China, moved to **Gitee**
    (local history of 6 commits preserved; push once the link is sent).
  - **Windows read-only fix**: project files had 600/700 permissions (fine in WSL but blocked
    \\wsl$ copy from Windows) → normalized to .py=644, .sh=755; the "folder read-only" is a classic
    cosmetic attribute myth (refresh restores).
  - **License**: MIT for source code, CC BY 4.0 for documentation, **paper text all rights
    reserved** (under submission); MNIST data attributed to original authors.

---

## 8 Key Technical Takeaways — One-Page Quick Reference

| # | Takeaway | Key numbers / evidence |
|---|---|---|
| 1 | **Cross-sample state pollution is the pulse readout's biggest killer** — not weights, not time-averaging. Physically solved via LIF leakage + a silent interval (ISI). | mean-field κ=0.2 weights: 0.203 (no reset) → **0.947** (reset), long-window 0.956; u-leakage (deep negative locking) is the dominant carrier, not firing-rate residue (clear-rates only 0.205). |
| 2 | **Hard reset vs realistic protocol**: hard reset is the effective "control," but un-physical; the paper's route is LIF membrane leakage + ISI. | realistic LIF+ISI reaches 0.955/0.956/0.953 (n=1000) on mean-field weights — ≥ the hard-reset 0.947, via pure physical decay. |
| 3 | **Push/pull imbalance breaks training; recalibration restores it.** Raising κ (inhibition pool) suppresses all output rates and raising TARGET amplifies the correct-class push. | LIF ↓ output 1132→626 Hz/class, net update −686→+5544→+8524; uncalibrated control ends **0.501**, recalibrated (TARGET=5000/κ=1.0) reaches **0.877**. |
| 4 | **TARGET is non-monotonic (sweet spot at 5000)** — higher is not better (overfits). | frozen 0.404 (T3000) → 0.648 (T5000) → 0.372 (T7000); T10000 overfits (train 1.0/test 0.505). |
| 5 | **The gradient-alignment paradox**: strong local gradient alignment does not imply good end results. Signal-level (aggregated) metrics are more informative than per-sample cosine. | κ=0.2 alignment ALL +0.764 (vs κ=1.0 +0.238) yet lower frozen acc (0.414 vs 0.648); per-sample cos is noise-dominated (σ~0.4). |
| 6 | **Acceptance must be multi-seed frozen evaluation** (stochastic spiking network); single-implementation in-training eval is misleading. | protocol 5 seed × n=500, mean ≥ 0.8: 14k **0.877±0.007**, s1 **0.874±0.006**, s2 **0.806±0.015**; single-run in-training 0.49 vs acceptance 0.76. |
| 7 | **Overtraining collapse** — performance degrades after the peak as TARGET saturates and the push stops converging; early stopping matters. | lif5 peaks 0.894@12–14k → 0.750@20k; stride 50k peak 0.285@3–4k → 0.124; halving α (1.5e-8→7.5e-9) pushes the collapse further out (0.348 record). |
| 8 | **R scaling is not a silver bullet on MNIST**: SNR∝√R holds on XOR, but on MNIST it fails (3/3 negative) without synchronized output-anchor recalibration; simple R×. | R=400→0.122, R=800→0.106 (κ=0); scaled-R mean-field κ=0.2: R=200→0.203 … R=1600→0.129 (locked dynamics worsen). |
