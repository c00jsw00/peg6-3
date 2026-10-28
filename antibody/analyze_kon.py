#!/usr/bin/env python
"""Estimate k_on (association rate) from the 2D PMF (robust version).

Approach (transition-state theory on a 2D free-energy surface):
  k_on ~ (kBT/h) * exp(-dG_on/KBT),  dG_on = G(TS) - G(free)
  TS = lowest saddle = minimax path from BOUND (low CV1, high CV2) to FREE
  (high CV1, low CV2) on the 2D PMF grid.
  free reference = the cell with the lowest PMF in the "bulk free" region
  (CV1 > 90 A) that has REAL frames (not a NaN-interpolated cell).

Confidence: every bin carries a frame count (raw, unweighted). A bin is
"real" if it has >= min_frames frames. The minimax path is only trusted in
bins that are real; the free reference must be a real bin. Bins with zero
frames are linearly interpolated so the grid is connected.

Note on prefactor: k_on is bimolecular (M^-1 s^-1). The TST prefactor
(kBT/h ~ 6.2e12 s^-1 at 298 K) per particle, divided by standard conc c=1 M,
gives the M^-1 s^-1 form (identical numerically for c=1 M).
"""
import heapq
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
STEERED = HERE / "steered"
OUT = HERE / "steered" / "postproc"
OUT.mkdir(parents=True, exist_ok=True)

KBT = 0.592           # kcal/mol at 298 K
KBT_OVER_h = 6.2e12   # s^-1 (k_B T / h)
MIN_FRAMES = 3        # bins with >= this many frames are "real"


def load_windows():
    for pat in ("d*_n*", "l*_k*_n*_s*", "a*_n*_s*", "t*_n*_s*"):
        for d in sorted(STEERED.glob(pat)):
            if not (d / "cv.npz").is_file():
                continue
            npz = np.load(d / "cv.npz")
            yield npz["cv1"].astype(float), npz["cv2"].astype(int), \
                npz["U"].astype(float), d.name


def main():
    windows = list(load_windows())
    print("%d windows" % len(windows))

    # coarse grid.  CV1 edges are 20 A bins; the upper edge is dynamic so it
    # covers the guidance-reached free state (CV1 up to ~150 A), always at least
    # 111 A so a genuine free bin exists.
    all_c1 = np.concatenate([w[0] for w in windows])
    all_c2 = np.concatenate([w[1] for w in windows])
    c1_max = max(111.0, float(np.nanpercentile(all_c1, 99))) + 20.0
    c1_max = max(int(np.ceil(c1_max / 20.0)) * 20, 110)
    c1_edges = np.arange(10, c1_max + 20, 20)            # 20 A bins
    c2max = int(np.ceil(np.max(all_c2) / 30.0)) * 30
    c2_edges = np.arange(0, c2max + 30, 30)      # 30-contact bins

    # per-window-normalized reweighted histogram + raw frame counts
    prob = np.zeros((len(c1_edges) - 1, len(c2_edges) - 1), dtype=float)
    nframes = np.zeros((len(c1_edges) - 1, len(c2_edges) - 1), dtype=float)
    for cv1, cv2, U, name in windows:
        u = U - U.max()
        w = np.exp(u)
        h, _, _ = np.histogram2d(cv1, cv2, weights=w, bins=(c1_edges, c2_edges))
        nh, _, _ = np.histogram2d(cv1, cv2, bins=(c1_edges, c2_edges))
        prob += h / (w.sum() + 1e-12)
        nframes += nh

    prob = prob / (prob.sum() + 1e-12)
    pmf = -KBT * np.log(prob + 1e-12)
    pmf[prob <= 0] = np.nan

    grid1 = 0.5 * (c1_edges[:-1] + c1_edges[1:])
    grid2 = 0.5 * (c2_edges[:-1] + c2_edges[1:])
    real = nframes >= MIN_FRAMES

    # interpolate NaN holes so the grid is fully connected for minimax
    if np.isnan(pmf).any():
        try:
            from scipy.interpolate import RegularGridInterpolator
            yy, xx = np.meshgrid(grid1, grid2, indexing="ij")
            pts = np.column_stack([yy.ravel(), xx.ravel()])
            good = ~np.isnan(pmf)
            interp = RegularGridInterpolator((grid1, grid2), pmf, method="linear")
            vals = interp(pts[good])
            pmf_filled = np.full_like(pmf, np.nan)
            pmf_filled.ravel()[:len(pts)] = vals
            med = float(np.nanmedian(pmf))
            pmf_filled[np.isnan(pmf_filled)] = med
            pmf = pmf_filled
        except Exception:
            med = float(np.nanmedian(pmf))
            pmf = pmf.copy()
            pmf[np.isnan(pmf)] = med

    # print PMF with frame counts (R = real, . = interpolated)
    print("\n=== coarse 2D PMF (rows=CV1_A, cols=CV2_contacts) ===")
    print("       " + "  nframes  " * len(grid2) if False else
          "cv1\\cv2  " + "  " + "  ".join("%6.0f" % g for g in grid2))
    for i, g1 in enumerate(grid1):
        vals = []
        for j in range(len(grid2)):
            tag = "R" if real[i, j] else "."
            vals.append("%6.1f%s " % (pmf[i, j], tag) if tag == "R" else
                        "%6.1f%s " % (pmf[i, j], tag))
        print("%6.0f  " % g1 + "".join(vals))

    n1, n2 = pmf.shape

    # ---- locate basins on REAL bins ----
    # bound = lowest PMF among real high-contact (CV2 > 200) cells
    bound_candidates = [(i, j, pmf[i, j]) for i in range(n1) for j in range(n2)
                        if real[i, j] and grid2[j] > 200.0]
    if not bound_candidates:
        print("No real high-contact bound cells.")
        return
    b_i, b_j, b_pmf = min(bound_candidates, key=lambda t: t[2])
    bound_ij = (b_i, b_j)
    bound = (float(grid1[b_i]), float(grid2[b_j]), float(b_pmf))

    # free = lowest PMF among real high-CV1 (CV1 > 90) cells
    free_candidates = [(i, j, pmf[i, j]) for i in range(n1) for j in range(n2)
                       if real[i, j] and grid1[i] > 90.0]
    if not free_candidates:
        free_candidates = [(i, j, pmf[i, j]) for i in range(n1) for j in range(n2)
                           if grid1[i] > 90.0]   # allow interpolated
    f_i, f_j, f_pmf = min(free_candidates, key=lambda t: t[2])
    free_ij = (f_i, f_j)
    free = (float(grid1[f_i]), float(grid2[f_j]), float(f_pmf))
    free_real = real[f_i, f_j]

    print("\n=== basins (CV1_A, CV2_contacts, PMF_kcalmol) ===")
    print("  bound = (%.1f, %.1f, %.2f)  frames=%d" %
          (bound[0], bound[1], bound[2], nframes[b_i, b_j]))
    print("  free  = (%.1f, %.1f, %.2f)  frames=%d  (real=%s)" %
          (free[0], free[1], free[2], nframes[f_i, f_j], free_real))

    # diagnostics along the binding path (cleft region CV1 40-70)
    path_cells = [(i, j) for i in range(n1) for j in range(n2)
                  if 40 <= grid1[i] <= 70 and real[i, j]]
    if path_cells:
        pc = max(path_cells, key=lambda p: pmf[p[0], p[1]])
        print("  cleft (CV1 40-70) max PMF = %.2f (CV1=%.0f, CV2=%.0f, frames=%d)" %
              (pmf[pc], grid1[pc[0]], grid2[pc[1]], nframes[pc]))

    # ---- minimax transition state ----
    def cell_key(i, j):
        return pmf[i, j]
    start, end = bound_ij, free_ij
    dist = {start: cell_key(*start)}
    pq = [(cell_key(*start), start)]
    parent = {}
    while pq:
        d, c = heapq.heappop(pq)
        if d > dist.get(c, 1e99) + 1e-9:
            continue
        i, j = c
        if c == end:
            break
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < n1 and 0 <= nj < n2:
                cand = max(d, cell_key(ni, nj))
                if cand < dist.get((ni, nj), 1e99):
                    dist[(ni, nj)] = cand
                    parent[(ni, nj)] = c
                    heapq.heappush(pq, (cand, (ni, nj)))
    if free_ij not in dist:
        print("No connected path.")
        return
    path = []
    c = free_ij
    while c is not None:
        path.append(c)
        c = parent.get(c)
    path.reverse()
    saddle_ij = max(path, key=lambda p: cell_key(*p))
    saddle = (float(grid1[saddle_ij[0]]), float(grid2[saddle_ij[1]]),
              float(pmf[saddle_ij]))
    saddle_real = real[saddle_ij[0], saddle_ij[1]]
    # path confidence: fraction of path cells that are real
    real_frac = sum(real[p[0], p[1]] for p in path) / len(path)

    dG_on = saddle[2] - free[2]
    print("\n=== k_on estimate ===")
    print("  free state PMF = %.2f kcal/mol (frames=%d)" % (free[2], nframes[f_i, f_j]))
    print("  TS PMF = %.2f kcal/mol (frames=%d, real=%s)" %
          (saddle[2], nframes[saddle_ij], saddle_real))
    print("  dG_on (free->TS) = %.2f kcal/mol" % dG_on)
    print("  path length=%d cells, %d%% real" % (len(path), int(100 * real_frac)))
    if dG_on < 0:
        print("  (TS below free state: barrier-free / diffusion-limited)")
    if dG_on >= 0:
        k_on = KBT_OVER_h * np.exp(-dG_on / KBT)
        print("  k_on (TST) = %.3e s^-1 = %.3e M^-1 s^-1 (c=1 M)" % (k_on, k_on))
    else:
        k_on = KBT_OVER_h
        print("  k_on: no barrier -> diffusion-limited ceiling %.3e M^-1 s^-1" % k_on)

    summary = {
        "bound": [round(bound[0], 1), round(bound[1], 1), round(bound[2], 2)],
        "free": [round(free[0], 1), round(free[1], 1), round(free[2], 2)],
        "ts_cv1_A": round(saddle[0], 1),
        "ts_cv2_contacts": round(saddle[1], 1),
        "ts_pmf_kcalmol": round(saddle[2], 2),
        "free_pmf_kcalmol": round(free[2], 2),
        "free_frames": int(nframes[f_i, f_j]),
        "dG_on_kcalmol": round(dG_on, 2),
        "k_on_per_M_s": float(k_on),
        "prefactor_M_per_s": float(KBT_OVER_h),
        "path_length": len(path),
        "path_real_fraction": round(real_frac, 2),
        "free_is_real": bool(free_real),
        "ts_is_real": bool(saddle_real),
        "n_windows": len(windows),
    }
    (OUT / "barriers_kon.json").write_text(json.dumps(summary, indent=2))
    print("\nSaved barriers_kon.json")

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 7))
    finite = pmf < 1e8
    im = ax.pcolormesh(grid2, grid1, pmf, shading="nearest", cmap="viridis",
                       vmin=np.nanpercentile(pmf[finite], 5),
                       vmax=np.nanpercentile(pmf[finite], 95))
    ax.set_xlabel("CV2 (contacts)")
    ax.set_ylabel("CV1 (COM-COM, A)")
    ax.set_title("Binding path (TS = saddle) on 2D PMF (R=real, .=interp)")
    fig.colorbar(im, ax=ax, label="PMF (kcal/mol)")
    ax.plot([grid2[j] for i, j in path], [grid1[i] for i, j in path],
            "w-", lw=3, label="minimax binding path")
    ax.plot(saddle[1], saddle[0], "ro", ms=12, label="TS")
    ax.plot(free[0], free[1], "bs", ms=12, label="free")
    ax.plot(bound[0], bound[1], "go", ms=12, label="bound")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "pmf_kon_path.png", dpi=150)
    print("Saved pmf_kon_path.png")


if __name__ == "__main__":
    main()
