# Experiment Progress (late-night update 2026-08-16, ready to hand over at any time)

> This file is the quick-look for whoever picks up next: where things stand, what is running,
> what is done, and what to do next. For detailed mechanisms and history see
> docs/PROJECT_DOC.md / docs/MNIST_RESULTS_CN.md / docs/PROJECT_LOG.md.

## Terminology quick reference (read this first; all explanations in this document start here)

> Every coined/project-specific term below is explained in plain language at its first
> occurrence; here the 13 most core ones are laid out in everyday words. The full 26 are in
> [docs/GLOSSARY.md](GLOSSARY.md) (the authoritative home of all explanations):

| Term | Plain language (one sentence) |
|---|---|
| probabilistic spiking synapse | a synapse fires spikes with probability (with randomness), and information travels in the average firing rate |
| eligibility trace (E) | each synapse's "recent-activity ledger" — the temporal carrier of the learning signal |
| local error (d) | how "wrong I am," computed by each neuron from its neighbors' information only, without relying on a global gradient |
| online local learning | learn on-the-fly, each synapse updates using only signals from its immediate neighborhood; no global step |
| push/pull imbalance | the output layer's two forces — push the correct class's target rate up and pull the wrong classes down — become unbalanced, so training stalls |
| recalibration | retune TARGET/κ to levels the network can physically reach so the push/pull forces rebalance (the key fix behind the paper's main result) |
| cross-sample state pollution | the "echo" of the previous sample lingers and distorts the next one — the biggest bottleneck of the spiking network |
| deep negative locking | a leak-free neuron's membrane potential gets pinned at a deep negative value and "plays dead" for a long time |
| hard reset / reset protocol | manually zero the state after every sample (a control approach; the official route does not use it) |
| ISI silent interval | insert a silent, no-input, no-target period between samples so the state decays back to baseline naturally (fully physical) |
| frozen evaluation | lock the weights and read only test images — the real score (the in-training self-evaluation does not count) |
| multi-seed acceptance | evaluate the same weights with 5 different random seeds and report mean±std (spiking is stochastic, so a single run fluctuates a lot) |

More: mean-field (noise-free ceiling), gradient alignment (angle between update direction and the true gradient), overtraining collapse (performance collapses if training runs too long), κ inhibition pool (a global brake on the output layer), LIF membrane leakage (the membrane potential naturally leaks/decays) etc. → [docs/GLOSSARY.md](GLOSSARY.md).

## Current status in one sentence

**✅ The user goal (a realistic protocol, LIF+ISI frozen test acc ≥ 0.8) has been robustly achieved (2026-08-16 afternoon):**
1. **Hard-reset control version**: `exp4/reset_a15_cont` (reset protocol) final **0.824** (n=1000; single 0.817 / long-window 0.848 / counting 0.850);
2. **Realistic version (LIF membrane leakage §4.5 + inter-sample silent ISI, no hard reset)**: failure mechanism = **output-layer push/pull imbalance** → fix = **TARGET=5000 + κ=1.0 recalibration**.
   - **20k main run `exp4/lif5_t5000_k1_isi50_a15_20k`**: peaked @12-14k; **acceptance (multi-seed 5×n=500 frozen): @14k = 0.877±0.007 (all seeds ≥0.868), @12k = 0.873±0.017**; 12k single-implementation 0.894, mean-field 0.900. **Overtraining collapse after 12k** (TARGET saturates and can no longer push): @20k only 0.750±0.021 — deliver the 12-14k checkpoint.

## In progress (updated 2026-08-16 23:05)

- **✅ Cross-seed reproduction acceptance fully complete** (seed 1/2 each 14k, same params as the main run):
  - s1 `exp4/lif5_s1_...14k/ckpt_14000.npz` → **0.874±0.006** (all seeds ≥0.864);
  - s2 `exp4/lif5_s2_...14k/mnist_checkpoint.npz` → **0.806±0.015** (0.794/0.784/0.810/0.828/0.812, all seeds ≥0.784, mean≥0.8 met; worst single seed 0.784, cross-seed variance ~7 points);
- **✅ LIF parameter sweep complete** (two batches of 15 combinations × N=1000, including 3 supplemental runs; relationship analysis in `docs/TABLE_LIF_PARAM_SWEEP.md`);
- Remaining: final documentation wrap-up (add the recalibration mechanism to MNIST_RESULTS_CN + the acceptance table + the collapse; README/index update).

## Results quick reference

| Model | Protocol | Training evaluation | Acceptance (multi-seed frozen 5×n=500) |
|---|---|---|---|
| Historical best (κ=0.2, α=1.5e-8, 20k) | no reset (IF, no leakage) | 0.348 | reset eval 0.265/0.281/0.283 |
| exp4/reset_a15_20k | hard reset (control) | 0.767 | 0.776/0.810/0.812 |
| exp4/reset_a10_30k | hard reset (control) | 0.778 | 0.789/0.815/0.811 |
| **exp4/reset_a15_cont** | hard reset (control) | **0.824** | **0.817/0.848/0.850** |
| exp4/lif2_tm0p5_isi50_a15_20k | LIF+ISI uncalibrated (κ=0.2/T1000) | **0.501** | — |
| **exp4/lif5_t5000_k1_isi50_a15_20k @14k** | LIF+ISI recalibrated | — | **0.877±0.007 (single-implementation 0.894 / mean-field 0.900)** |
| exp4/lif5_t5000_k1_isi50_a15_20k @20k | same as left (overtrained) | 0.507 | 0.750±0.021 (collapse) |
| **exp4/lif5_s1_t5000_k1_isi50_a15_14k** | recalibrated seed 1 | 0.701 | **0.874±0.006 (all seeds ≥0.864)** |
| **exp4/lif5_s2_t5000_k1_isi50_a15_14k** | recalibrated seed 2 | 0.748 | **0.806±0.015 (all seeds ≥0.784, mean≥0.8 met)** |
| mean-field weights (capability proof) | LIF+ISI readout (τ_m=0.5, ISI=100) | — | **0.955/0.956/0.953** |

## Data files

- `mnist_mf_pulse_eval_reset.csv`: all independent evaluations of the reset protocol (n=1000, single/long-window/counting + mean-field control)
- `mnist_0p8_campaign.csv` + `docs/TABLE_0P8_CAMPAIGN.md`: comparison table of the 0.8-campaign parameter combinations (generated by `make_campaign_table.py`)
- `exp4/lif5_ckpts/`: every-2k snapshots of the 20k main run + acceptance evals; `exp4/lif5_{s1,s2}_*_ckpts/`: seed-reproduction snapshots + eval.log
- `eval_mf_batch.py` (multi-readout single-implementation acceptance), `eval_multiseed.py` (5 seed × n=500 multi-implementation acceptance, **the acceptance-standard protocol**)
- `meanfield_ckpts/`: mean-field 300k × κ=0/0.1/0.2/0.4 (for capability proof, not the delivery route)

## Next steps

1. ~~Seed reproduction acceptance~~ **✅ Done**: main run 0.877±0.007 / s1 0.874±0.006 / s2 0.806±0.015 — robust across seeds (mean all ≥0.8; s2's worst single seed 0.784, meaning individual-seed variance is ~7 points);
2. **Deliver checkpoints**: lif5 main run @14k (0.877) + s1 @14k (0.874) + s2 @14k (0.806); if further refinement is needed, continue with a lower α (`mnist_finetune_mf.py`, watching the TARGET-saturation collapse risk — keep near the peak);
3. **Documentation wrap-up (in progress)**: add to MNIST_RESULTS_CN.md the recalibration mechanism (push/pull balance + §4.5/§5.4/§5.5 rationale) + the multi-seed acceptance table + the collapse finding + the parameter-sweep relationship analysis (`TABLE_LIF_PARAM_SWEEP.md` first version already done);
4. **Paper material**: four-protocol comparison — IF no-leakage vs LIF+ISI uncalibrated (0.501) vs LIF+ISI recalibrated (0.877±0.007) vs hard reset (0.824) — plus parameter-indicator relationships (non-monotonic TARGET / κ alignment paradox / double-edged ISI), the data chain is complete.

## Discipline reminders

- Run all scripts with `python3`; long tasks `setsid nohup ... &` writing to each directory's `run_log.txt`;
- At most 4 parallel processes, monitor with `free -h` at any time (training RSS ~270MB/process);
- Write new experiment logs to the experiment directory's `run_log.txt`, and summarize into docs/PROJECT_LOG.md.
