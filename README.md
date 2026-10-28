<div align="center">

<img src="assets/emu.png" alt="BioEmu2 logo" width="120">

# BioEmu2

### A foundation model for biomolecular structure and dynamics

**Predict folded structures, sample equilibrium ensembles, and propagate structures forward in time.**

[Installation](#installation) · [Quick start](#quick-start) · [Tutorials](#tutorials) · [Rate study example](#rate-study-example) · [Documentation](docs/)

<br>

<img src="https://github.com/microsoft/bioemu2/releases/download/readme-media/mcl1_puma_clean_grid_v2.gif" alt="BioEmu2 simulating the PUMA BH3 peptide binding MCL-1" width="420">

<sub>BioEmu2 simulating the PUMA BH3 peptide binding MCL-1.</sub>

</div>

---

BioEmu2, short for "biomolecular emulator", is developed by Microsoft Research
AI for Science.

Key features include:

- molecular systems containing proteins, nucleic acids, ligands, and other
  chemical components;
- equilibrium sampling and structure prediction from the command line or
  Python;
- time-lagged trajectory generation with implicit transfer operator models;
- contact conditioning and inference-time steering toward desired properties;
- context-parallel sampling of structures that do not fit on one GPU.

## System requirements

BioEmu2 requires Linux and an NVIDIA GPU; macOS and Windows are not supported.
The `cuda` extra installs CUDA 13 cuEquivariance, which requires compute
capability 7.5 or newer, such as NVIDIA T4, A100, RTX A6000, L40S, H100, and B200
([CUDA 13 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#deprecated-architectures)).
GPUs with compute capability 7.0 (Volta, such as NVIDIA V100) need the CUDA 12
[installation](#installation) instead.
We have tested V100 (compute capability 7.0), T4 (7.5, with the CUDA 12
installation), A100 and RTX A6000 (8.0 and 8.6), and B200 (10.0); other eligible
GPUs have not been verified end to end. Check
your GPU's compute capability with
`nvidia-smi --query-gpu=name,compute_cap --format=csv`.

Context-parallel sampling requires at least four GPUs.

## Installation

Use Python 3.12, 3.13, or 3.14 on a supported Linux machine. We recommend
installing BioEmu2 in an isolated
[virtual environment](https://docs.python.org/3/library/venv.html) or
[conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
To install in an isolated virtual environment, create and activate it from the
repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Replace `python3.12` with the name or path of your Python 3.12, 3.13, or 3.14
executable if needed. Use `python -m pip` so packages are installed into the
active Python environment rather than a different interpreter's `pip`.

Run the installation command for your GPU from the repository root.
For GPUs with compute capability 7.5 or newer, such as NVIDIA T4, A100,
RTX A6000, L40S, H100, and B200, use the `cuda` extra:

```bash
python -m pip install ".[cuda]"
```

Compute capability 7.0 (Volta, such as NVIDIA V100) is the oldest supported;
on these GPUs, use the `cuda12` extra with the PyTorch CUDA 12.6 package index:

```bash
python -m pip install ".[cuda12]" \
  --extra-index-url https://download.pytorch.org/whl/cu126
```

GPUs with compute capability below 7.0 are not supported.

We tested BioEmu2 and evaluated its benchmark metrics with Python 3.12. Python
3.13 and 3.14 pass the test suite, but we have not evaluated benchmark metrics
with them.

For development, see [DEVELOPER.md](DEVELOPER.md).

For Jupyter or VS Code notebooks, follow the [notebook
quickstart](#notebook-quickstart) to install visualization dependencies and
register a kernel.

## Quick start

### CLI

Sample eight equilibrium conformations of adenylate kinase (PDB 1AKE):

```bash
bioemu2 sample-equilibrium \
  --pdb-id 1AKE \
  --out outputs/adk
```

Or predict a folded structure from a PDB entry:

```bash
bioemu2 predict-structure \
  --pdb-id 1UBQ \
  --num-samples 1 \
  --out outputs/ubiquitin
```

Propagate each sampled adenylate kinase conformation by 1 ns:

```bash
bioemu2 propagate \
  --x0-cif outputs/adk/topology.cif \
  --x0-xtc outputs/adk/samples.xtc \
  --lag-time-ns 1 \
  --out outputs/adk-propagated
```

Run `bioemu2 --help` to list all commands.

### Python API using EmuSession

Construct an `EmuSession` to sample and propagate structures:

```python
from pathlib import Path

from bioemu2 import EmuSession, System, load_model, save_frames

model = load_model()

# Predict a folded structure for ubiquitin.
ubiquitin = System.from_pdb_id("1UBQ")
prediction = EmuSession(ubiquitin, model=model, mode="structure").sample(num_samples=1)
save_frames(prediction, Path("outputs/ubiquitin"))

# Sample equilibrium conformations of adenylate kinase.
adk = System.from_pdb_id("1AKE")
emu = EmuSession(adk, model=model, mode="equilibrium")
samples = emu.sample(num_samples=8)
save_frames(samples, Path("outputs/adk"))

# Propagate each adenylate kinase conformation by 1 ns.
endpoints = emu.propagate(samples, lag_time_ns=1.0)
save_frames(endpoints, Path("outputs/adk-propagated"))
```

`save_frames()` writes `samples.xtc` and `topology.cif` to each output
directory.

See [define molecular systems](docs/system.md), [predict and sample
structures](docs/sampling.md), and [propagate structures](docs/propagation.md)
for more options. The functional Python APIs provide
[high-level sampling](docs/functional-api.md#sample-structures) and
[propagation](docs/functional-api.md#propagate-structures) functions similar to
the CLI commands, plus lower-level workflows for explicit control of
embeddings and samplers.

## More workflows

### Steer sampling

Choose a steering sampler to favour structures with desired properties:

```bash
bioemu2 sample-equilibrium \
  --system src/bioemu2/assets/examples/systems/myoglobin_101m.yaml \
  --out outputs/myoglobin \
  --sampler edm_gd
```

See [steer sampling](docs/steering.md) and [define steering
potentials](docs/potentials.md) to select an algorithm and configure its
objective.

### Sample large structures across multiple GPUs

Context parallelism distributes the model's pair representations across a
square mesh of GPUs, reducing the memory required on each device:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc-per-node=4 \
  --module bioemu2.cp sample-equilibrium \
  --pdb-id 1AKE \
  --checkpoint bioemu2-preview \
  --num-samples 3 \
  --out outputs/adk-cp
```

See [sample large structures with context
parallelism](docs/context-parallel-sampling.md) for mesh requirements, the
Python API, and current limitations.

### Inspect and subsample MSAs

Retrieve an alignment from ColabFold or load your own A3M files, inspect the
raw sequences, and select or randomly subsample rows with Python indexing.
Pass the resulting `MSA` directly to `EmuSession` to control the alignment
used for structure prediction, equilibrium sampling, or propagation without
manually constructing embeddings.

See [use multiple sequence alignments](docs/msa.md) for local-file loading,
[MSA subsampling](docs/msa.md#select-rows-and-subsample-an-msa), and
[custom-MSA inference](docs/msa.md#use-a-custom-msa-for-inference).

## Tutorials

- [Define molecular systems](docs/system.md) from PDB IDs, structure files, or sequences.
- [Convert Boltz input YAML](docs/boltz.md) into systems you can sample from.
- [Customise multiple-sequence alignments](docs/msa.md) used to condition sampling.
- [Predict and sample structures](docs/sampling.md) for a system.
- [Improve sampling speed and scale to bigger systems](docs/speed-and-capacity.md) by choosing batch size and precision.
- [Load a model checkpoint](docs/checkpoints.md), choose where checkpoints are cached, or use a local checkpoint.
- [Propagate structures through time](docs/propagation.md).
- [Reuse trunk embeddings](docs/reuse-embeddings.md) across sampling and propagation calls, in memory or on disk.
- [Use the functional Python API](docs/functional-api.md) to sample and propagate without an `EmuSession`.
- [Steer sampling](docs/steering.md) towards structures with desired properties.
- [Define steering potentials](docs/potentials.md) that encode the properties you want to favour.
- [Sample large structures](docs/context-parallel-sampling.md) across multiple GPUs with context parallelism.

## Notebook quickstart

The [`examples`](examples) directory contains complete workflows as regular
Python files split into executable cells with `# %%` markers. This Python
Interactive format provides notebook-style cells while keeping the examples as
plain `.py` files. They require a Linux machine with a supported NVIDIA GPU and
network access.

Activate a Python environment as described in [installation](#installation).
From the repository root, install BioEmu2 with the visualization dependencies
and register a kernel. The `viz` extra includes `ipykernel` and `py3Dmol`:

```bash
python -m pip install ".[cuda,viz]"
python -m ipykernel install --user --name bioemu2 --display-name "BioEmu2"
```

On compute capability 7.0 GPUs, such as NVIDIA V100, replace the first command
with:

```bash
python -m pip install ".[cuda12,viz]" \
  --extra-index-url https://download.pytorch.org/whl/cu126
```

Select the **BioEmu2** kernel in Jupyter or VS Code. If registration succeeds
but the kernel does not appear in VS Code, restart VS Code and try again.
To check that it uses the environment you installed into, run
`import sys; print(sys.executable)` in a cell. The examples load system YAML
from the installed `bioemu2` package,
regardless of the working directory. Relative `outputs/` paths are written
under the current working directory.

Open a single-GPU example in VS Code and select **Run Cell**, or run **Jupyter:
Run Current File in Interactive Window** from the Command Palette. Launch the
context-parallel example with `torchrun` as described in the
[context-parallel sampling guide](docs/context-parallel-sampling.md#python-api).

To run without VS Code, convert an example with
[Jupytext](https://jupytext.readthedocs.io/) and open it in JupyterLab:

```bash
python -m pip install jupytext jupyterlab
jupytext --to notebook examples/sample_structures.py
jupyter lab examples/sample_structures.ipynb
```

Select the **BioEmu2** kernel, then run individual cells or choose **Run All
Cells** from the **Run** menu.

In VS Code, you can instead run **Jupyter: Export Current Python File as Jupyter
Notebook** from the Command Palette. Generated `.ipynb` files are ignored by
Git.

## Rate study example

The `antibody/` directory contains a complete, reproducible protein–ligand
**on/off rate** study (k_on, k_off, K_D) using BioEmu2 steered sampling, built
for the **9FO glycan** receptor–ligand complex (`wild.pdb`: receptor chains
**H+L**, ligand chain **A**). It is a PDC2026 protein-design study. A full
recipe (all commands, CV definitions, rate math, and caveats) is documented in
[antibody/RATES_STUDY.md](antibody/RATES_STUDY.md).

### Coordinates (CVs)

| CV | Definition | Use |
|----|------------|-----|
| **CV1** | ligand(A) COM − protein(H+L) COM, in Å (rigid-motion invariant, no Kabsch) | steering |
| **CV2** | heavy-atom **pair** contact count @ 4.5 Å (protein-heavy × ligand-heavy) | steering |
| **r_min** | min heavy-atom distance ligand→protein | measurement |

### Pipeline (run from `antibody/`)

```bash
# 1) SMC 2D umbrella (bound -> dissociated)
python run_steered_sampling.py --windows "35,260 55,180 75,60 95,0" \
    --n-samples 30 --batch 10 --seed 42

# 2) Free state (push-out guidance, flat U=0)
python run_guidance.py --targets "90 110 130 150" --n-samples 12 --seeds 0 1

# 3) Transition-region r_min gap (4-15 A)
python run_transition_windows.py --targets "80 85" --seeds 0 1 2

# 4) Cleft-entrance association barrier (SMC)
python run_assoc_umbrella.py --n-samples 25 --batch 5 --seeds 0 1 2

# 5) Cleft-exit off-barrier saddle (stiff-well guidance ladder)
python run_umbrella_ladder.py --targets "46 40 52 25 56 15" --k 60 --gw 6 \
    --n-samples 25 --steps 1500 --seeds 0 1 2

# 6) Analyze + rates + plots
python analyze_kon.py
python analyze_kon_final.py
python make_final_plots.py
```

Each driver is idempotent (skips completed windows) and stores `cv.npz` +
`samples.xtc` under `steered/`. Step 5 is the **stiff-well** guidance ladder
(k=60, batch=1, real per-frame U) — a soft SMC (k~0.5, batch 5) cannot relocate
the ligand out of the bound basin and fails to reach this saddle.

### Rate recipe (Eyring, first-order, per-molecule)

```
kBT        = 0.592 kcal/mol          (300 K)
kBT_over_h = 6.21e12 s^-1            (per-molecule kBT/h — NOT KBT/H)
k_off      = 6.21e12 * exp(-dG_off‡/kBT)
           dG_off‡ = PMF(TS) - PMF(bound)  on the high-CV2 (cleft) exit path
k_on       = 6.21e12 * exp(-dG_on‡/kBT)   (TST from fine-grid cleft-entrance PMF)
           dG_on‡ = PMF(gate) - PMF(approach or free)   -> barrier-limited 1e4-1e5 M^-1 s^-1
K_D        = k_off / k_on_phys       (detailed balance)
```

The estimator is **per-window reweighting**: for window *i*,
`B_i(b) = Σ_j exp(U_i,j − U_i,max) / Σ_j exp(U_i,j)`; **mean B_i over windows**.
A single global `exp(U − max)` is wrong (the dissociated window dominates with
exp(800+)).

### Results vs experiment (89 windows, 2254 frames)

| Quantity | Calculated (BioEmu2) | Experimental | Match? |
|----------|----------------------|--------------|--------|
| ΔG_on‡ (gate−approach) | 11.2 kcal/mol | 11.9 kcal/mol | ✓ close |
| **k_on** | 3.9 × 10⁴ M⁻¹s⁻¹ (barrier-limited) | 1.1 × 10⁴ M⁻¹s⁻¹ | ~3.5× high |
| ΔG_off‡ | ≥ 20.3 kcal/mol (floor-limited **lower bound**) | 21.05 kcal/mol | ✓ (bounded) |
| **k_off** | ≤ 8.0 × 10⁻³ s⁻¹ (**upper bound**) | 2.25 × 10⁻³ s⁻¹ | ✓ (bounded) |
| **K_D** (detailed balance) | ~2 × 10⁻⁷ M | 2.04 × 10⁻⁷ M | ✓ ✓ matches |

The **cleft-entrance barrier** (cv1 45–51, cv2 18–45, PMF 10.7–13.1 kcal/mol)
is resolved and robust. The **cleft-exit saddle** reads at the 1e-15 log-floor
cap (20.45 kcal/mol); experimental 21.05 kcal/mol sits just above that cap, so
ΔG_off‡/k_off are **bounds** at this data density. Robust quantities = barrier
height and K_D. To resolve k_off exactly: lower the PMF floor (1e-20 → cap ~27
kcal/mol) + more saddle frames, or metadynamics.

The raw trajectories (`antibody/steered/`, 679 windows), all drivers/analyzers,
the per-window `cv.npz`/`xtc`, and the plotted PMFs (`pmf_final.png/.svg`,
`cleft_entrance_barrier.png`) are committed.

## License

BioEmu2 is released under the [MIT License](LICENSE). It includes code and
reference data adapted from other open-source projects; the [third-party
notices](NOTICES.md) identify those components and their licenses.
