# Local Online Backpropagation — Probabilistic Spiking Neural Network

> 📖 **Terminology Note**: The coined/project-specific terms in this document (recalibration, eligibility trace, cross-sample state pollution, frozen evaluation, multi-seed acceptance, mean-field, gradient alignment, collapse, etc.) are briefly explained at their first occurrence; the authoritative definitions are consolidated in [docs/GLOSSARY.md](docs/GLOSSARY.md).

> **Pure spiking, pure online, pure local backpropagation**: each neuron updates its synaptic weights using only its local error and eligibility trace — no global gradients, no error-propagation network, no pretraining, no weight injection.
> On MNIST, from random initialization, it reaches a frozen test accuracy of **0.877 ± 0.007** (multi-seed acceptance).

[📄 Paper](Local%20Online%20Backpropagation.txt) · [🧭 Background & Reading Map (read first)](docs/BACKGROUND.md) · [📚 Full Document Index](docs/README.md) · [📊 Parameter-Relationship Analysis](docs/TABLE_LIF_PARAM_SWEEP.md) · [📈 Experiment Progress](docs/EXPERIMENT_PROGRESS.md)

---

## 🚀 Quick Start (clone and run — data ships with the repo, zero downloads)

```bash
git clone <this-repo>
cd <repo-directory>
python3 mnist_demo_train.py          # one-click train + live visualization, ~40–60 min (flagship config)
python3 mnist_demo_train.py --samples 1000   # 15–20 min quick demo
```

- **Windows users**: copy the entire `dist/windows_demo/` folder (or the zip in Releases), double-click `mnist_train_demo.bat` to start training — fully offline.
- On finishing, training automatically produces three artifacts: a training-curve PNG, a weight checkpoint, and a result summary; the curve looks like this:

![MNIST demo training curves](docs/figures/demo_training_curves_example.png)

> The demo program `mnist_demo_train.py` fully reproduces the paper's main-result protocol (LIF membrane leakage + inter-sample silent interval (ISI) + output-layer recalibration),
> and supports `--config uncal/reset/if` to reproduce the paper's four comparison protocols in one click.

---

## 🧠 What This Is

An **online local error learning** framework (the paper's SDE framework) on **probabilistic spiking neurons**:

- **Pure spiking**: information travels in Poisson spike trains; weight updates are driven by the online product of **eligibility trace × local error**;
- **Pure local**: each synapse updates using only the local signals of the neurons before and after it — no global gradient information required;
- **Fully physical**: state decay relies on LIF membrane leakage (τ_m) + inter-sample silent interval (ISI), with no hard-reset hack;
- **Result**: MNIST 784 → shared convolution (5×5×32, only 64 weight parameters) → FC128 → 10, random initialization,
  multi-seed frozen acceptance **0.877 ± 0.007** (5 seeds × n=500, all seeds ≥ 0.85).

## 📊 Main Results

| Model | Protocol | Acceptance (multi-seed frozen) |
|---|---|---|
| **lif5 main run @14k** | LIF+ISI recalibrated (TARGET=5000, κ=1.0) | **0.877 ± 0.007** |
| seed 1 reproduction @14k | same as left | 0.874 ± 0.006 |
| seed 2 reproduction @14k | same as left | 0.806 ± 0.015 |
| reset_a15_cont | hard reset (control) | 0.824 |
| lif2 uncalibrated | LIF+ISI not recalibrated (control) | 0.501 |
| mean-field (noise-free upper bound) | — | 0.955–0.965 |

**Key findings** (see [docs/TABLE_LIF_PARAM_SWEEP.md](docs/TABLE_LIF_PARAM_SWEEP.md)):

- **Recalibration**: output-layer push/pull imbalance is the main cause of failure; training recovers once TARGET=5000 + κ=1.0 is fixed;
- **Alignment paradox**: configurations with high per-sample gradient alignment (κ=0.2: +0.764) are in fact worse end-to-end (0.414),
  with Spearman ρ = **−0.553** between align and frozen acc across the 16 parameter sets;
- **ISI double-edged**: silent intervals 0/50/100 → acc 0.540/0.648/0.210 (the spuriously low training loss at ISI=0 is a state-pollution signal).

## 📁 Repository Structure

```
├── mnist_demo_train.py      # paper demo: one-click training + visualization (entry point for reviewers)
├── mnist_shared.py          # training kernel (shared convolution + LIF online local learning)
├── mnist_lif_table.py       # parameter × metric measurement (covariance/gradient alignment/SNR/sign consistency)
├── analyze_lif_sweep.py     # relationship analysis (Spearman/Pearson) → docs/TABLE_LIF_PARAM_SWEEP.md
├── eval_multiseed.py        # acceptance protocol (5 seeds × n=500 frozen evaluation)
├── mnist_data/              # MNIST data (ships with the repo; clone and run)
├── exp4/                    # paper main-result experiment directory (checkpoints not committed; reproducible)
├── docs/                    # all documentation: progress/logs/handoff/results/parameter sweep
└── dist/                    # demo program build artifacts (not committed; via Releases)
```

## 🔬 Reproduction (paper acceptance protocol)

```bash
# Training (flagship config: TARGET=5000, κ=1.0, τ_m=0.5, ISI=50, α=1.5e-8, N=14000)
python3 mnist_shared.py <seed> 14000 1.0 200 1.5e-8 3000 30 5000 1000 0.02 1.0 0 0 50 0.5

# Multi-seed frozen acceptance (5 seeds × n=500)
python3 eval_multiseed.py exp4/lif5_t5000_k1_isi50_a15_20k/ckpt_14000.npz

# Parameter × metric sweep (raw data source for the 16-set relationship analysis)
python3 mnist_lif_table.py shared 1000 1.0 200 1.5e-8 3000 30 5000 0 1.0 0 0.02 50 0.5 0.2
python3 analyze_lif_sweep.py > docs/TABLE_LIF_PARAM_SWEEP.md
```

Full commands and parameter explanations are in [docs/PROJECT_DOC.md](docs/PROJECT_DOC.md).

## 🗺️ Online Repositories

| Platform | URL | Purpose |
|---|---|---|
| **GitHub (this repo)** | https://github.com/nekonekomiaomiao/local-backprop-snn | English mirror (external showcase / reviewers) |
| **Gitee (primary / Chinese)** | https://gitee.com/nekonekomiaomiao/SNN | Primary development & full process records (daily logs, all experiment logs, Chinese docs) |

> Note for maintainers: both repos are the same project. The Gitee (Chinese) repo carries the
> complete provenance (PROJECT_LOG day-by-day, all experiment run logs); this GitHub repo is the
> English build (English docs + code/data/demo), regenerated by `github_prep.sh` from the Gitee tree.

## 📜 License

- **Source code**: MIT (see [LICENSE](LICENSE))
- **Documentation**: CC BY 4.0 (attribution)
- **Paper**: all rights reserved (currently under submission/publication; please cite when referencing)
- **MNIST data**: owned by the original authors, distributed with the package for research use (see LICENSE for attribution)
