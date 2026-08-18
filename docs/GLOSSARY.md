# Glossary — Plain-Language Explanations of Project-Specific Terms

> **Convention**: Project documents may use project-specific jargon, but every term is
> explained in plain language at its first occurrence (in reading order) and cross-referenced
> back to this glossary afterwards. This file is the authoritative source of explanations.
> For reading order, see [README.md](README.md) (root) and the docs index.
> Chinese version: `../docs/GLOSSARY.md`.

---

### probabilistic spiking synapse
- **Plain language**: a synapse that fires spikes *with probability* — same input, slightly different spike trains each time (random), and information travels in the average firing rate.
- **Paper**: `Local Online Backpropagation.txt` (SDE framework); older draft `一生有爱何惧风飞沙.txt` §4.5.
- **Code**: `spiking_step()` in `mnist_shared.py` (Poisson sampling).

### eligibility trace (E)
- **Plain language**: each synapse's "recent-activity ledger" — how strongly and how recently it was activated; it is the *temporal carrier* of the learning signal.
- **Paper**: §8.5 (eligibility time constant τ_e).
- **Code**: `E1/E2/E3` in `mnist_shared.py`.

### local error / delta (d)
- **Plain language**: how "wrong I am" computed by each neuron from its neighbors only — no global gradient required.
- **Code**: `d_out/d_fc/d_f` in `mnist_shared.py`.

### online local learning
- **Plain language**: learn on-the-fly, each synapse updates using only signals from its immediate neighborhood; no global step that waits for the whole network.
- **Paper**: core claim of the paper (online form of the SDE framework).

### push/pull imbalance
- **Plain language**: the output layer has two forces — *push* the target class's firing rate up and *pull* the wrong classes down; when the target value is set beyond what the network can physically reach, the push never gains traction, weights drift one way, and training stalls.
- **Fix**: recalibration (below).
- **Code**: output error `d_out = f̂ − TARGET·y` in `mnist_shared.py`.

### recalibration
- **Plain language**: retune the learning target (TARGET) and the output-layer global inhibition (κ) to levels the network can physically reach, so the push/pull forces rebalance and training recovers.
- **Values**: TARGET 1000→5000, κ 0.2→1.0 (the paper's main-result configuration).
- **Code**: `mnist_shared.py` argv[8]=TARGET, argv[11]=KAPPA.

### cross-sample state pollution
- **Plain language**: the "echo" of the previous sample (membrane potential, firing-rate residue) has not decayed before the next sample arrives — the network reads the next digit while still in the previous answer's state; distorts both training and evaluation.
- **Key finding**: the biggest bottleneck of the spiking network; physically solved by LIF leakage + silent interval (ISI).

### deep negative locking
- **Plain language**: with no leakage (IF), a strongly inhibited neuron's membrane potential sits pinned at a deep negative value, "playing dead" and never firing again — an extreme form of pollution.
- **Diagnostic evidence**: clearing membrane only → 0.948; clearing rates only → 0.205 (see EXPERIMENT_PROGRESS.md).

### hard reset / reset protocol
- **Plain language**: manually zero all network state after every sample — simple and effective, but it is the "control" approach (not the paper's official route).
- **Code**: `mnist_shared.py` argv[13]=1 (`reset_state()`).

### ISI / silent interval (inter-sample interval)
- **Plain language**: a silent period of "no input, no target" inserted between samples, letting membrane potential and firing rates decay back to baseline naturally — fully physical, no manual reset.
- **Code**: `mnist_shared.py` argv[14]=ISI_STEPS.

### LIF membrane leakage (leaky integrate-and-fire)
- **Plain language**: the neuron's membrane potential leaks/decays with time constant τ_m — if it does not fire, it slowly returns to rest; the physical mechanism that lets pollution dissolve by itself.
- **Paper**: older draft §4.5 (f_out=−λ/ln(1−λθ/I); IF is the λ→0 special case).
- **Code**: `mnist_shared.py` argv[15]=TAU_M.

### κ inhibition pool (κ=KAPPA)
- **Plain language**: a global "brake" on the output layer — the summed activity of all output neurons is scaled back, creating competition (one active class suppresses the others).
- **Code**: `mnist_shared.py` argv[11] (`u -= KAPPA·Σf̂·dt`).

### frozen evaluation / frozen test
- **Plain language**: after training, **freeze the weights** (no more learning) and feed only test images to measure classification — measures what was *learned*, not what was *memorized*.
- **Code**: `eval_multiseed.py`; `frozen_test()` in `mnist_demo_train.py`.

### multi-seed acceptance
- **Plain language**: same weights, sampled 5 times with different random seeds, report mean±std — because spiking networks are stochastic, a single run fluctuates; average over several runs to make it count.
- **Protocol**: 5 seeds × n=500, mean ≥ 0.8 = pass (`eval_multiseed.py`).

### single / multi implementation
- **Plain language**: each run of a spiking network is one realization of a random process; single = look at one run, multi = average over several.
- **Key finding**: at the κ=1.0 low-rate operating point single-run evaluation is pathologically low, so acceptance must be multi-implementation.

### overtraining collapse
- **Plain language**: performance degrades if training goes on too long — here, TARGET saturates and can no longer push, weights keep drifting, and test accuracy falls from its peak.
- **Key finding**: main run peaks at 12–14k samples, collapses by 20k (0.877→0.750).

### mean-field
- **Plain language**: the deterministic limit that "smooths out" random spikes into average rates — what the same architecture could reach under ideal (noise-free) conditions (0.955–0.965 here).
- **Code**: `gd_meanfield_shared.py`, `eval_mf_batch.py`.

### gradient alignment (gradient angle)
- **Plain language**: the cosine of the angle between each actual weight update's direction and the "theoretically optimal direction" (the true gradient) — closer to 1 means "walking straighter".
- **Measure**: per-sample `cos(ΔP, −g)` in `mnist_lif_table.py`.
- **Key finding**: the "alignment paradox" — high alignment (κ=0.2: +0.764) does not imply good end results (0.414); see TABLE_LIF_PARAM_SWEEP.md.

### update SNR / expected efficiency / sign consistency
- **Plain language**: three "weight-update quality" metrics — SNR = signal strength of the mean update relative to its fluctuation; efficiency = actual update relative to the theoretically expected one; sign consistency = fraction of updates whose direction agrees with the true gradient's sign.
- **Measure**: `mnist_lif_table.py`.

### corr(e,d) (eligibility–error covariance)
- **Plain language**: the correlation between the eligibility trace E and the local error d — whether "what to remember" and "what to fix" appear together; the paper hypothesizes they are (nearly) independent (correlation ≈ 0).
- **Measure**: last-5-samples statistics in `mnist_lif_table.py`.

### shared convolution
- **Plain language**: one convolutional kernel's weights are reused at every position of the image — far fewer weights, and it still respects the "local learning" constraint.
- **Code**: topology in `mnist_shared.py` (784→CONV5×5×32 str2→FC128→10; only 64 conv parameters).

### three readouts (single / long-window / counting)
- **Plain language**: three ways to read the same frozen evaluation — output at the last tick (single), average over a window (long-window), and total spike count over the window (counting) — to cross-check and avoid single-point noise.

### in-sample transient adaptation
- **Plain language**: because training learns "while looking," the network can temporarily adapt to the current sample within the sample — so the accuracy measured during training is inflated and cannot be used as the final score.
- **Key finding**: training acc ~0.9, but the frozen test is the real score.

### normalized MSE (normalized mean squared error loss)
- **Plain language**: mean squared error divided by the square of the target value, so losses are comparable across TARGET settings; its plateau ≈ the Poisson noise floor (~0.049 here).
- **Code**: `0.5·((f−TARGET·y)/TARGET)²` in `mnist_shared.py`/`mnist_lif_table.py`.

### training evaluation vs acceptance protocol
- **Plain language**: the in-training evaluation (single implementation, no state reset) is pathologically low and noisy; paper results are always the "multi-seed frozen evaluation". The two numbers are not directly comparable.
