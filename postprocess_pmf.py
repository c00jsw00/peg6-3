#!/usr/bin/env python
"""Post-processing for the wild.pdb dissociation study.

1) Load per-window (d_target, n_target) samples: CV1 (COM-COM A), CV2
   (pair contact count, int), and steering energy U (kcal/mol).
2) Unbiased reweighting (Wu/umbrella-sampling importance sampling).
   Tilted distribution per window: p_tilt_i(x) ~ p_base(x) * exp(-U_i(x)).
   Unbiased: P(x) ~ sum_i p_tilt_i(x) * w_i(x) / Z_i  with w_i=exp(+U_i).
   (FKC SMC returns equal-weight tilted particles -> uniform prior per window.)
3) 2D PMF in (CV1, CV2) with reweighted weights.
4) Identify bound basin (low CV1 + high CV2), transition region (mid CV1,
   falling CV2), and dissociated basin (high CV1, CV2~0); estimate barrier
   height along a 1D CV1 profile (CV2=0 edge / CV2=260 edge) and estimate
   k_off = (kBT/h) * exp(-beta * dG_barrier)  (transition-state rate,
   first-order, per-window-averaged). k_on is left as N/A unless
   rebinding trajectories are available.

Inputs: steered/<d>_<n>/cv.npz or cv.csv, window.json
Output: steered/postproc/{pmf_2d.npy, grid_u.npy, grid_v.npy, pmf.csv,
        profile_1d.npy, profile_1d.csv, barriers.json, pmf_2d.png, profile_1d.png}
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
STEERED = HERE / "steered"
OUT = HERE / "steered" / "postproc"
OUT.mkdir(parents=True, exist_ok=True)
KBT = 0.592  # kcal/mol at 298 K
# Eyring prefactor = kBT/h (per molecule) at 298 K.
# The correct value is ~6.21e12 s^-1. (Old value 9.44e14 s^-1 was 152x too
# high, overestimating k_off; corrected here.)
KBT_OVER_h = 6.21e12  # s^-1  (Eyring prefactor, kBT/h @ 298 K)


def load_windows():
    """Yield (d_target, n_target, cv1, cv2, U) for every window dir.

    Picks up both the original d*_n* SMC/guidance windows AND the
    umbrella-ladder windows (l{T}_k{k}_n{cv2t}_s{seed}, COM-COM only).
    """
    dirs = []
    for pat in ("d*_n*", "l*_k*_n*_s*"):
        for d in STEERED.glob(pat):
            if (d / "cv.npz").is_file() or (d / "cv.csv").is_file():
                dirs.append(d)
    dirs = sorted(set(dirs))
    for d in dirs:
        meta = json.loads((d / "window.json").read_text())
        npz = np.load(d / "cv.npz")
        # Support both original (d_target, n_target) and umbrella-ladder
        # (T, cv2t) window JSON formats.
        d_target = float(meta.get("d_target", meta.get("T", 0.0)))
        n_target = int(meta.get("n_target", meta.get("cv2t", 0)))
        cv1 = npz["cv1"].astype(float)
        cv2 = npz["cv2"].astype(int)
        U = npz["U"].astype(float)
        yield d_target, n_target, cv1, cv2, U, d


def reweight_all():
    windows = list(load_windows())
    print(f"{len(windows)} windows:")
    for dt, nt, cv1, cv2, U, d in windows:
        n = len(cv1)
        print("  %s: d_target=%.1f n_target=%d n=%d cv1 range %.1f..%.1f "
              "cv2 range %d..%d U_mean=%.2f" %
              (str(d), dt, nt, n, cv1.min(), cv1.max(),
               cv2.min(), cv2.max(), U.mean()))
    return windows


def main():
    windows = reweight_all()
    if not windows:
        print("No windows found; run sampling first.")
        return

    # Build a 2D grid spanning the full observed range.
    all_c1 = np.concatenate([w[2] for w in windows])
    all_c2 = np.concatenate([w[3] for w in windows])
    # CV1 grid: from ~10 A (tight) up to the free state.  The free-state
    # windows (guidance push-out) reach CV1 >> 100 A; make the upper edge
    # dynamic so it covers every observed distance, with margin, and always
    # extends to at least 110 A (a genuine free state) even if sampling is thin.
    c1_hi = max(110.0, float(np.nanpercentile(all_c1, 99))) + 10.0
    c1_edges = np.arange(10, round(c1_hi / 5.0) * 5 + 5, 5)  # 5 A bins
    # CV2 grid: 0..max (0, 5, ... )
    c2max = int(np.ceil(np.max(all_c2) / 5)) * 5 if all_c2.size else 300
    c2_edges = np.arange(0, c2max + 5, 5)

    # Stable per-window-normalized umbrella reweighting.
    # For window i, B_i(b) = [sum_{j in b} e^{U_{i,j}}] / [sum_j e^{U_{i,j}}]
    # is an unbiased estimator of p_base(b) and sums to 1 over the grid.
    # Averaging B_i over windows gives a stable, low-variance p_base (no window
    # can be dominated by an astronomically large exp(U)).
    prob_grid = np.zeros((len(c1_edges) - 1, len(c2_edges) - 1), dtype=float)
    n_win = 0
    for (_, _, cv1, cv2, U, _) in windows:
        u = U - U.max()                # per-window shift for numerical stability
        w = np.exp(u)                 # per-frame corrected weights (O(1) after shift)
        h, _, _ = np.histogram2d(cv1, cv2, weights=w, bins=(c1_edges, c2_edges))
        prob_grid += h / (w.sum() + 1e-12)
        n_win += 1
    prob = prob_grid / (prob_grid.sum() + 1e-12)   # combine windows (mean of B_i)
    # Log floor: 1e-15 allows resolving PMF up to -KBT*ln(1e-15) ~ 21 kcal/mol,
    # so a real barrier higher than the old 16.4 (1e-12) cap can be measured.
    # Bins with exactly zero reweighted probability are set to NaN below.
    pmf = -KBT * np.log(prob + 1e-15)
    pmf[prob <= 0] = np.nan

    # grid centres
    grid1 = 0.5 * (c1_edges[:-1] + c1_edges[1:])
    grid2 = 0.5 * (c2_edges[:-1] + c2_edges[1:])

    # raw (unweighted) frame count per bin, for reporting
    counts, _, _ = np.histogram2d(
        np.concatenate([w[2] for w in windows]),
        np.concatenate([w[3] for w in windows]),
        bins=(c1_edges, c2_edges))

    # 1D profile along CV1: PMF(CV1) at each CV1 bin = -KBT ln(sum over CV2 of p)
    # Integrate over CV2 (dissociation path = CV2 -> 0)
    profile1d, c2_int = np.zeros((len(grid1),), dtype=float), None
    p1d = np.sum(prob, axis=1)  # marginal over CV2
    p1d_norm = p1d / (p1d.sum() + 1e-12)
    # 1D profile log floor: 1e-15 (matches 2D PMF; caps at ~20.45 kcal/mol)
    profile1d = -KBT * np.log(p1d_norm + 1e-15)

    # Save
    np.savez(OUT / "pmf_2d.npy", pmf=pmf, grid_u=grid1, grid_v=grid2,
             counts=counts, grid1=c1_edges, grid2=c2_edges)
    np.savez(OUT / "profile_1d.npy", cv1=grid1, pmf=profile1d)

    # write pmf csv
    with open(OUT / "pmf.csv", "w") as f:
        f.write("cv1_A,cv2_contacts,pmf_kcalmol,count\n")
        for i1, (g1, p_row) in enumerate(zip(grid1, pmf)):
            for i2, g2 in enumerate(grid2):
                f.write(f"{g1:.2f},{g2:.1f},{pmf[i1,i2]:.4f},{counts[i1,i2]}\n")
    # write profile csv
    with open(OUT / "profile_1d.csv", "w") as f:
        f.write("cv1_A,pmf_kcalmol\n")
        for g1, p in zip(grid1, profile1d):
            f.write(f"{g1:.2f},{p:.4f}\n")
    print(f"Saved PMF to {OUT}/pmf_2d.npy ; grid CV1 {c1_edges[0]:.0f}-{c1_edges[-1]:.0f} A "
          f"CV2 {c2_edges[0]}-{c2_edges[-1]}")

    # Plot 2D PMF
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(grid2, grid1, pmf, shading="nearest", cmap="viridis",
                       vmin=np.nanpercentile(pmf[pmf < 1e8], 5),
                       vmax=np.nanpercentile(pmf[pmf < 1e8], 95))
    ax.set_xlabel("CV2 (heavy-atom pair contacts)")
    ax.set_ylabel("CV1 (COM-COM distance, A)")
    ax.set_title("Unbiased 2D PMF (kcal/mol) -- wild.pdb dissociation")
    fig.colorbar(im, ax=ax, label="PMF (kcal/mol)")
    ax.invert_yaxis()  # larger CV1 at bottom? keep as-is
    fig.tight_layout()
    fig.savefig(OUT / "pmf_2d.png", dpi=150)

    # 1D profile plot
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(grid1, profile1d, lw=2)
    ax2.set_xlabel("CV1 (COM-COM distance, A)")
    ax2.set_ylabel("PMF (kcal/mol)")
    ax2.set_title("1D free-energy profile along dissociation (CV1)")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(OUT / "profile_1d.png", dpi=150)

    # Identify basins from the 1D CV1 profile.  The bound state is the well at
    # low CV1 (ligand in the pocket, high contacts, CV1 ~ 25-50 A); the free state
    # is the high-CV1, low-contact basin (CV1 > 90 A, reached by the guidance
    # push-out windows).  We locate the bound basin as the profile minimum within
    # the low-CV1 region (so a lower free-state plateau at high CV1 does NOT get
    # picked as the "bound" minimum), and measure the barrier as the maximum PMF
    # on the path between them.
    finite_mask = np.isfinite(profile1d)
    if finite_mask.sum() == 0:
        print("No finite PMF bins.")
        return

    # low-CV1 region: the bound + encounter pocket region, before the free plateau
    low_idx = int(np.searchsorted(c1_edges, 60.0))  # bins with centre < 60 A
    finite_low = np.where(finite_mask[:low_idx])[0]
    if finite_low.size == 0:
        print("No finite PMF in bound (low-CV1) region.")
        return
    bound_idx = int(finite_low[np.argmin(profile1d[finite_low])])
    bound_cv1 = grid1[bound_idx]

    # free basin: rightmost CV1 edge (high COM-COM, low contacts)
    right_idx = len(grid1) - 1

    # Barrier: maximum PMF strictly between the bound well and the free edge.
    search = list(range(bound_idx + 1, right_idx))   # exclude both endpoint bins
    search = [i for i in search if finite_mask[i]]
    if not search:
        print("No bins on the bound->free path.")
        return
    barrier_idx = int(np.argmax(profile1d[np.array(search)]))
    barrier_idx = int(np.array(search)[barrier_idx])
    barrier_cv1 = grid1[barrier_idx]
    barrier_height = float(profile1d[barrier_idx] - profile1d[bound_idx])
    if not np.isfinite(barrier_height) or barrier_height <= 0:
        # free plateau itself is the highest point (barrier is flat/edge)
        barrier_idx = right_idx
        barrier_cv1 = grid1[right_idx]
        barrier_height = float(profile1d[right_idx] - profile1d[bound_idx])

    print("\n=== Basin estimates (CV1, A) ===")
    print(f"  bound basin    CV1 ~ {bound_cv1:.1f} A   (PMF {profile1d[bound_idx]:.2f})")
    print(f"  dissociated    CV1 ~ {grid1[right_idx]:.1f} A  (PMF {profile1d[right_idx]:.2f})")
    print(f"  transition     CV1 ~ {barrier_cv1:.1f} A   "
          f"(barrier height ~ {barrier_height:.1f} kcal/mol)")

    # k_off estimate (first-order, Eyring): k_off = (kBT/h) exp(-beta*ΔG‡)
    # beta = 1/KBT; ΔG‡ in kcal/mol
    if barrier_height is not None and np.isfinite(barrier_height) and barrier_height > 0:
        k_off = KBT_OVER_h * np.exp(-barrier_height / KBT)
        half_life = np.log(2) / k_off  # seconds
        print("\n=== Rate estimate (Eyring, first-order) ===")
        print(f"  ΔG‡ ≈ {barrier_height:.1f} kcal/mol")
        print(f"  k_off ≈ {k_off:.2e}  s^-1  (half-life ≈ {half_life:.1f} s)")
    else:
        print("\n=== Rate estimate ===")
        print(f"  ΔG‡ not reliably estimated (barrier_height={barrier_height})")

    # k_on: require rebinding / free-state trajectories.  This script computes the
    # bound->free barrier (k_off) only; k_on (free->bound, the reverse) is handled
    # by analyze_kon.py which needs the free-state (CV1>90) frames from guidance.
    print("  k_on : see analyze_kon.py (free-state CV1>90 frames required)")

    # Save summary
    summary = {
        "bound_cv1_A": float(bound_cv1),
        "barrier_cv1_A": float(barrier_cv1),
        "barrier_height_kcalmol": float(barrier_height) if np.isfinite(barrier_height) else None,
        "k_off_per_s": float(KBT_OVER_h * np.exp(-barrier_height / KBT)) if (
            barrier_height is not None and np.isfinite(barrier_height) and barrier_height > 0) else None,
        "k_on": None,
        "grid_cv1_A": grid1.tolist(),
        "grid_cv2_contacts": grid2.tolist(),
    }
    (OUT / "barriers.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved barriers to {OUT}/barriers.json")


if __name__ == "__main__":
    main()
