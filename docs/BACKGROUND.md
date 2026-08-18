# Background & Reading Map — Start Here If This Is Your First Time in This Project's Context

> Default target reader: someone with **a bit of general SNN-domain knowledge** (you know what a spiking neuron, a firing rate, and STDP are), but **new to this project**.
> This document's job: to make clear "what context this project sits in, what problem it solves, what route it took, and how to read the repository."
> Plain-language explanations of terms are in [GLOSSARY.md](GLOSSARY.md) (with a brief explanation at each term's first occurrence too).

---

## 1. What context does this project sit in?

**Training spiking neural networks (SNNs) is a recognized hard problem.** Neurons emit only "firing / not firing" spikes, and the derivative that classical backpropagation needs does not exist on a spike. The traditional escape routes are three:

1. **STDP-style local rules** — biologically inspired, but lack an explicit "target" signal, making classification tasks difficult;
2. **surrogate gradient** — uses a smooth approximation to fool the non-differentiable point, but it is still global BP and needs information from the whole network;
3. **backpropagation through time (BPTT)** — exact but computationally expensive, needs to store the entire trajectory, and is biologically implausible.

**This project (and its paper) takes another route: online, local "local backpropagation" on probabilistic spiking synapses.**
The core claim: no global gradient is needed — each synapse only needs to multiply its "local error" with its "eligibility trace" (its own record of how recently it was active) to approximate the effect of backpropagation. This is the most load-bearing beam in the whole cognitive scaffold:
**"Local" is not a compromise but the method itself** — both biological plausibility and hardware (neuromorphic chip) friendliness come from here.

## 2. What questions does this project set out to answer?

1. Can this online local learning scheme — "probabilistic synapse + eligibility trace + local error" — actually learn at MNIST scale? (The paper only gives the theoretical framework; implementation and validation are the project's job.)
2. If it can learn, **what is the bottleneck?** (Answer: cross-sample state pollution — the "echo" of the previous sample disturbs the next one.)
3. How is the bottleneck solved **physically**? (Answer: LIF membrane leakage + inter-sample silent interval, not manual reset.)
4. Why do early configurations **get stuck**? (Answer: output-layer push/pull imbalance → fixed by recalibration.)
5. What is the **relationship** between each hyperparameter (TARGET/κ/τ_m/ISI/α/τ_e) and the final score? (A 16-set parameter sweep gives a quantitative answer.)

## 3. Technical roadmap (how the project got to 0.877)

```
Paper theory (SDE framework: continuous-time form of online local BP on probabilistic synapses)
   │
   ├─ 1. XOR validation (xor_residual_local_bp.py)
   │    local learning works on a toy task as the theory predicts; grid sweep establishes
   │    parameter laws such as R×ΔT
   │
   ├─ 2. MNIST topology implementation (mnist_shared.py)
   │    shared convolution (784→5×5×4 str2→FC32→10) — weight reuse without violating locality
   │    correctness validation: gradient FD 12/12, forward equivalence, mean-field GD comparison
   │
   ├─ 3. Diagnostic toolchain (mnist_table.py / mnist_lif_table.py)
   │    covariance corr(e,d), gradient alignment, update SNR / sign consistency — quantifies "learning quality"
   │
   ├─ 4. Bottleneck localization: cross-sample state pollution
   │    non-leaky IF membrane potential deep negative locking → the hard-reset comparison
   │    version qualifies first (0.824)
   │
   ├─ 5. Physically faithful protocol: LIF membrane leakage + ISI silent interval (fully physical, no hard reset)
   │    fails (0.501) → localize push/pull imbalance → recalibration (TARGET 5000 + κ=1.0) → recovers
   │
   ├─ 6. Main result: 0.877±0.007 (multi-seed acceptance) + seed reproduction (0.874 / 0.806)
   │
   └─ 7. Parameter × metric relationship sweep (16 sets): TARGET non-monotonicity / κ alignment paradox / ISI double-edged
```

The experiment directories corresponding to each node: `exp/` (early) → `exp3/` (transition) → `exp4/` (0.8 sprint and the paper's main result).

## 4. Term map (where core concepts sit along the route)

| Concept | Where it appears in the roadmap | One-liner |
|---|---|---|
| probabilistic synapse | throughout (from step 1 on) | synapse fires with probability; information lives in the average firing rate |
| eligibility trace E | the learning rule itself | the synapse's "recent-activity ledger" |
| local error d | the learning rule itself | "how wrong I am" computed at the neuron's doorstep |
| gradient alignment | step 3 diagnostics | how close the actual update direction is to the theoretically optimal one |
| corr(e,d) | step 3 diagnostics | whether "bookkeeping" and "error correction" are in sync (paper assumes ≈0) |
| cross-sample state pollution | step 4 bottleneck | the previous sample's echo disturbs the next one |
| hard reset | step 4 comparison | manually clear state (comparison approach) |
| LIF membrane leakage | step 5 physically faithful | membrane potential naturally leaks and decays |
| ISI silent interval | step 5 physically faithful | a no-input silent period between samples |
| push/pull imbalance | step 5 failure mechanism | the two forces on the output layer fall out of balance → gets stuck |
| recalibration | step 5 fix | retune TARGET/κ to levels the network can reach |
| frozen evaluation / multi-seed acceptance | step 6 evaluation | lock the weights and average over several test runs |
| mean-field | throughout, comparison | the deterministic ceiling that smooths out noise |
| overtraining collapse | step 6 finding | performance collapses if training goes on too long |

The full 26 explanations: [GLOSSARY.md](GLOSSARY.md).

## 5. How to read this repository (documentation map)

| What you want to know | What to read | When to read it |
|---|---|---|
| What this is, run it in 30 seconds | `README.md` (root) | any time |
| background, roadmap, context | **this file** | first time entering |
| what a term means | `docs/GLOSSARY.md` | look up as needed |
| where things stand now, results at a glance | `docs/EXPERIMENT_PROGRESS.md` | every time you check progress afterward |
| mechanisms, parameters, troubleshooting history, file status | `docs/PROJECT_DOC.md` | before touching code |
| day-by-day decisions and command timeline | `docs/PROJECT_LOG.md` | when tracing history |
| full MNIST experiment record | `docs/MNIST_RESULTS_CN.md` | when writing the paper / digging deep |
| parameter × metric relationships | `docs/TABLE_LIF_PARAM_SWEEP.md` | when discussing hyperparameters |
| want to run it directly | `dist/windows_demo/` (double-click the bat) or `python3 mnist_demo_train.py` | immediately |

## 6. Key conclusions at a glance (for those short on time)

1. **Local online learning genuinely learns to 0.877±0.007 on MNIST** — not a toy, but full-scale convolution;
2. **The bottleneck is cross-sample state pollution**, and it must be solved by physical means (LIF+ISI); hard reset is only the comparison;
3. **Recalibration is the make-or-break move**: the same algorithm goes from uncalibrated 0.501 → calibrated 0.877;
4. **"High gradient alignment" does not mean "good results"** (alignment paradox, ρ=−0.553) — the evaluation metrics of local learning need rethinking;
5. The mean-field (noise-free) ceiling is 0.965; the spiking implementation still has distance to cover — the future room is clear.
