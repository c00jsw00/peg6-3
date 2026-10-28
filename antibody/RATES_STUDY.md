# PDC2026 Rate Study — BioEmu2 Protein–Ligand k_on/k_off/K_D (9FO glycan)

End-to-end example: estimate protein–ligand dissociation free-energy landscape and
**on/off rates** (k_on, k_off, K_D) using BioEmu2 steered sampling for the
**9FO glycan** receptor–ligand complex (`wild.pdb`: receptor chains **H+L**, ligand
chain **A**). This is the complete, reproducible pipeline from the PDC2026 protein
design competition.

## Files in this repo (rate-study specific)

```
wild.pdb                     # system: receptor chains H+L, ligand chain A
wild_system.yaml             # BioEmu2 System definition (receptor H,L + ligand A)
cv_potentials.py             # ComComPotential, ContactCountPotential, PushOutPotential
run_steered_sampling.py      # SMC 2D umbrella driver
run_guidance.py              # free-state push-out guidance driver
run_transition_windows.py    # r_min 4-15 A push-out transition windows
run_assoc_umbrella.py       # cleft-entrance (association) SMC umbrella
run_umbrella_ladder.py       # cleft-exit (off-barrier) stiff-well guidance ladder
analyze_kon.py               # coarse 2D PMF + minimax (k_on cross-check)
analyze_kon_final.py         # fine 2D PMF + detailed balance -> barriers_final.json
make_final_plots.py          # publication figures
pmf_ladder_final/            # results dir (barriers, PMFs, plots)
```

## Coordinates (CVs)

- **CV1** = ligand(A) COM − protein(H+L) COM, in Å. Rigid-motion invariant,
  **no Kabsch** needed (unlike per-atom shape restraints).
- **CV2** = heavy-atom **PAIR** contact count @ 4.5 Å (protein-heavy × ligand-heavy).
- **r_min** = min heavy-atom distance ligand→protein (good *measurement* CV, poor
  steering CV).

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install ".[cuda]"     # compute capability >= 7.5 (e.g. RTX 4090)
python -m pip install mdtraj numpy matplotlib
```

## End-to-end pipeline (6 steps)

### 1) SMC 2D umbrella (bound → dissociated)

```bash
python run_steered_sampling.py --windows "35,260 55,180 75,60 95,0" \
    --n-samples 30 --batch 10 --seed 42
```
Per window in `steered/d{t}_n{cv2t}/`: `cv.npz` (cv1, cv2, U1, U2, U),
`samples.xtc`, `window.json`. Idempotent (reuses `samples.xtc` if present).

### 2) Free state (push-out guidance, store U=0)

```bash
python run_guidance.py --targets "90 110 130 150" --n-samples 12 --seeds 0 1
```
Per-seed dirs `d{t}_n0_g_s{seed}/`. Store a **flat U=0** in the free region —
raw frame counts are the unbiased free plateau.

### 3) Transition-region r_min gap (4–15 Å)

```bash
python run_transition_windows.py --targets "80 85" --seeds 0 1 2
```
For this glycan/receptor, cv1 ≈ 63–76 Å ↔ r_min ≈ 4–14 Å.

### 4) Cleft-entrance association barrier (SMC)

```bash
python run_assoc_umbrella.py --n-samples 25 --batch 5 --seeds 0 1 2
```
Diagonal windows `a{d}_n{n}_s{seed}`:
(70,0)(62,0)(55,15)(50,30)(48,45)(45,60)(44,90)(42,120)(40,150)(36,180).

### 5) Cleft-exit off-barrier saddle (stiff-well guidance ladder)

```bash
python run_umbrella_ladder.py --targets "46 40 52 25 56 15" --k 60 --gw 6 \
    --n-samples 25 --steps 1500 --seeds 0 1 2
```
Per-seed dirs `l{T}_k{k}_n{cv2t}_s{seed}`; `weight_update=guidance`, batch=1,
**real per-frame U stored**. Soft SMC (k~0.5, batch 5) cannot reach this saddle —
use the stiff-well guidance ladder.
Probe first (`--n-samples 15 --seeds 0`, T=48/52 only) to confirm the saddle
(cv1 50-55 / cv2 10-50) is entered before expanding to 3 seeds.

### 6) Analyze + compute rates

```bash
python analyze_kon.py            # coarse 2D PMF + minimax (k_on cross-check)
python analyze_kon_final.py      # fine 2D PMF + detailed balance -> steered/postproc/barriers_final.json
python make_final_plots.py       # pmf_ladder_final/pmf_final.png + cleft_entrance_barrier.png
```

## Rate recipe (Eyring, first-order, per-molecule)

```
kBT         = 0.592 kcal/mol          (298/300 K)
kBT_over_h  = 6.21e12 s^-1            (per-molecule kBT/h — NOT KBT/H, which is 152x too high)
```

- `k_off = 6.21e12 * exp(-dG_off‡/kBT)`;
  `dG_off‡ = PMF(TS) − PMF(bound)` on the **high-CV2 (cleft)** dissociation path.
- `k_on` = TST from the **fine-grid cleft-entrance** PMF:
  `dG_on‡ = PMF(gate) − PMF(approach or free)`;
  `k_on = 6.21e12 * exp(-dG_on‡/kBT)` — barrier-limited ~10⁴–10⁵ M⁻¹s⁻¹,
  **NOT** the 1×10⁸ diffusion ceiling.
- `K_D = k_off / k_on_phys` (detailed balance).

## Results (89 windows, 2254 frames)

| Quantity | Calculated | Experimental |
|---|---|---|
| dG_on‡ (gate−approach) | **11.2 kcal/mol** | 11.9 kcal/mol |
| k_on | **3.9e4 M⁻¹s⁻¹** (barrier-limited) | 1.1e4 M⁻¹s⁻¹ |
| dG_off‡ | **≥ 20.3 kcal/mol** (floor-limited lower bound) | 21.05 kcal/mol |
| k_off | **≤ 8e-3 s⁻¹** (upper bound) | 2.25e-3 s⁻¹ |
| K_D (detailed balance) | **~2e-7 M** | 2.04e-7 M |

The **cleft-entrance barrier** (cv1 45-51, cv2 18-45, PMF 10.7-13.1 kcal/mol) is
resolved and robust. The **cleft-exit saddle** reads at the 1e-15 log-floor cap
(20.45 kcal/mol); experimental 21.05 sits just above the cap, so ΔG_off‡/k_off are
bounds at this data density.

## Key caveats

- **Barrier height** is the robust quantity; absolute k_off and K_D are bounds /
  order-of-magnitude.
- **Per-window reweighting** (the stable estimator): for window i,
  `B_i(b) = Σ_j exp(U_i,j − U_i,max) / Σ_j exp(U_i,j)`; **mean B_i over windows**.
  Never use a global `exp(U − max)` — the dissociated window dominates with
  exp(800+) and corrupts the PMF.
- **Dedupe** window sets across glob families
  (`d*_n*`, `l*_k*_n*_s*`, `a*_n*_s*`) — `d*_n*` and `d*_n0_g*` both match push-out
  dirs. Naive iteration double-counts them.
- The **1e-15 PMF floor** caps the barrier at 20.45 kcal/mol. To resolve the last
  0.75-2 kcal/mol on k_off: lower the floor (1e-20 → cap ~27) + more saddle frames,
  or metadynamics.

## Raw trajectory data

`steered/` (XTC trajectories, ~86 MB) is **gitignored** — reproducible by re-running
the drivers above. Results and plots are committed in `pmf_ladder_final/`.

## How to reproduce from scratch

1. Clone the repo; `wild.pdb` and `wild_system.yaml` define the system.
2. Build the venv (Setup).
3. Run steps 1–5 (sampling) — each driver is idempotent and skips completed windows.
4. Run step 6 (analyze + plots).
5. Read `pmf_ladder_final/association_barrier.json` (full result incl. off-barrier
   bounds) and `steered/postproc/barriers_final.json` (k_on/k_off/K_D).
