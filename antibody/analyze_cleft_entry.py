#!/usr/bin/env python
"""Full-range 2D PMF + cleft-entrance association barrier.

Builds a 2D PMF (cv1 COM-COM vs cv2 contacts) over the FULL range
(cv1 ~10-150 A, cv2 0-255 contacts) using all SMC/umbrella windows (d*, l*,
a*) with per-window normalized reweighting. A 2-A cv1 / 5-contact cv2 grid
resolves the cleft-entrance (cv1 40-60, cv2 20-70) while the free basin
(cv1>90, cv2<10) is covered by the U=0 push-out guidance windows.

Extracts:
  * free  = lowest real PMF in free basin (cv1>90, cv2<10)
  * bound = lowest real PMF in bound basin (cv1<30, cv2>180)
  * association path = Dijkstra minimax (free -> bound); cleft-entrance saddle
    = max PMF on that path.
  * dG_on = G(saddle) - G(free);  k_on = (kBT/h)*exp(-dG_on/kBT), capped 1e8
  * dG_bind = G(bound) - G(free)
Prints the path profile + cleft-entrance slice.

Free and cleft-entrance share the SAME 2D-PMF scale (free-plateau anchored by
the U=0 push-out frames), so dG_on is the absolute barrier to the cleft gate.
"""
import json, heapq
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
STEERED = HERE / "steered"
KBT = 0.592
KBT_OVER_h = 6.21e12     # s^-1 per molecule (kBT/h @ 298 K)
MIN_FRAMES = 3

def load_windows():
    for pat in ("d*_n*", "l*_k*_n*_s*", "a*_n*_s*"):
        for d in sorted(STEERED.glob(pat)):
            if not (d / "cv.npz").is_file():
                continue
            z = np.load(d / "cv.npz")
            yield z["cv1"].astype(float), z["cv2"].astype(int), \
                z["U"].astype(float), d.name

def main():
    windows = list(load_windows())
    print("windows used (incl. free push-out): %d" % len(windows))

    # 2-A cv1 (8..150), 5-contact cv2 (0..255). 2A bins resolve the cleft-
    # entrance saddle (44-56 A) and are adequate for the free plateau.
    c1_edges = np.arange(8.0, 152.0, 2.0)          # 72 bins
    c2_edges = np.arange(0.0, 260.0, 5.0)          # 52 bins
    n1, n2 = len(c1_edges)-1, len(c2_edges)-1
    prob = np.zeros((n1, n2), dtype=float)
    nframes = np.zeros((n1, n2), dtype=float)
    for cv1, cv2, U, name in windows:
        u = U - U.max()
        w = np.exp(u)
        h, _, _ = np.histogram2d(cv1, cv2, weights=w, bins=(c1_edges, c2_edges))
        nh, _, _ = np.histogram2d(cv1, cv2, bins=(c1_edges, c2_edges))
        prob += h / (w.sum() + 1e-12)
        nframes += nh
    prob = prob / (prob.sum() + 1e-12)
    pmf = -KBT * np.log(prob + 1e-15)
    pmf[prob <= 1e-15] = np.nan
    real = nframes >= MIN_FRAMES
    grid1 = 0.5*(c1_edges[:-1] + c1_edges[1:])
    grid2 = 0.5*(c2_edges[:-1] + c2_edges[1:])

    if np.isnan(pmf).any():
        try:
            from scipy.interpolate import RegularGridInterpolator
            yy, xx = np.meshgrid(grid1, grid2, indexing="ij")
            pts = np.column_stack([yy.ravel(), xx.ravel()])
            good = ~np.isnan(pmf)
            if good.sum() >= 3:
                interp = RegularGridInterpolator((grid1, grid2), pmf, method="linear")
                pmf_filled = np.full_like(pmf, np.nan)
                pmf_filled.ravel()[:len(pts)] = interp(pts[good])
                pmf_filled[np.isnan(pmf_filled)] = float(np.nanmedian(pmf))
                pmf = pmf_filled
        except Exception:
            pass
        if np.isnan(pmf).any():
            pmf[np.isnan(pmf)] = float(np.nanmedian(pmf))

    bound_cands = [(i,j,pmf[i,j]) for i in range(n1) for j in range(n2)
                   if real[i,j] and grid1[i] < 30 and grid2[j] > 180]
    if not bound_cands:
        bound_cands = [(i,j,pmf[i,j]) for i in range(n1) for j in range(n2)
                       if real[i,j] and grid1[i] < 35 and grid2[j] > 150]
    b_i, b_j, b_pmf = min(bound_cands, key=lambda t: t[2])
    free_cands = [(i,j,pmf[i,j]) for i in range(n1) for j in range(n2)
                  if real[i,j] and grid1[i] >= 90 and grid2[j] < 10]
    if not free_cands:
        free_cands = [(i,j,pmf[i,j]) for i in range(n1) for j in range(n2)
                      if real[i,j] and grid1[i] >= 85 and grid2[j] < 15]
    f_i, f_j, f_pmf = min(free_cands, key=lambda t: t[2])
    bound = (float(grid1[b_i]), float(grid2[b_j]), float(b_pmf))
    free  = (float(grid1[f_i]), float(grid2[f_j]), float(f_pmf))

    print("\n=== basins ===")
    print("  bound = (cv1=%.1f, cv2=%.1f, PMF=%.2f) frames=%d" %
          (bound[0], bound[1], bound[2], int(nframes[b_i, b_j])))
    print("  free  = (cv1=%.1f, cv2=%.1f, PMF=%.2f) frames=%d" %
          (free[0], free[1], free[2], int(nframes[f_i, f_j])))
    print("  dG_bind (bound-free) = %.2f kcal/mol" % (bound[2]-free[2]))

    # Dijkstra minimax free -> bound
    def cell_key(i, j):
        return pmf[i, j]
    dist = {(f_i, f_j): cell_key(f_i, f_j)}
    parent = {}
    pq = [(cell_key(f_i, f_j), (f_i, f_j))]
    while pq:
        d, c = heapq.heappop(pq)
        i, j = c
        if d > dist.get(c, 1e99) + 1e-9:
            continue
        if c == (b_i, b_j):
            break
        for di, dj in ((-1,0),(1,0),(0,-1),(0,1)):
            ni, nj = i+di, j+dj
            if 0 <= ni < n1 and 0 <= nj < n2:
                cand = max(d, cell_key(ni, nj))
                if cand < dist.get((ni, nj), 1e99):
                    dist[(ni, nj)] = cand
                    parent[(ni, nj)] = c
                    heapq.heappush(pq, (cand, (ni, nj)))

    if (b_i, b_j) not in dist:
        print("  no connected path free->bound"); return
    path = []
    c = (b_i, b_j)
    while c is not None:
        path.append(c); c = parent.get(c)
    path.reverse()
    real_frac = sum(real[p[0], p[1]] for p in path) / len(path)
    saddle_ij = max(path, key=lambda p: cell_key(*p))
    saddle = (float(grid1[saddle_ij[0]]), float(grid2[saddle_ij[1]]), float(pmf[saddle_ij]))
    real_saddle = real[saddle_ij[0], saddle_ij[1]]

    print("\n=== association path (minimax, free->bound) ===")
    print("  %d cells, %.0f%% real" % (len(path), 100*real_frac))
    print("  i    cv1   cv2     PMF")
    for k, (i, j) in enumerate(path):
        print("  %3d  %6.1f  %5.1f  %6.2f  %s" %
              (k, grid1[i], grid2[j], pmf[i,j], "R" if real[i,j] else "."))
    print("\n  saddle = (cv1=%.1f, cv2=%.1f, PMF=%.2f, real=%s)" %
          (saddle[0], saddle[1], saddle[2], real_saddle))

    dG_on = saddle[2] - free[2]
    print("\n=== association barrier ===")
    print("  dG_on (free->cleft saddle) = %.2f kcal/mol" % dG_on)
    if dG_on >= 0:
        k_on = KBT_OVER_h * np.exp(-dG_on / KBT)
        k_on_phys = min(k_on, 1e8)
        print("  k_on (TST) = %.3e M^-1 s^-1 (c=1M)" % k_on)
    else:
        k_on_phys = 1e8
        print("  barrier-free -> k_on PHYS = 1.0e8 M^-1 s^-1 (diffusion-limited)")
    print("  k_on PHYS (capped at diffusion 1e8) = %.3e M^-1 s^-1" % k_on_phys)
    print("  dG_bind (bound-free) = %.2f kcal/mol" % (bound[2]-free[2]))

    print("\n=== cleft-entrance slice (cv1 38-62, cv2 0-80) ===")
    c1_idx = [i for i in range(n1) if 38 <= grid1[i] <= 62]
    c2_idx = [j for j in range(n2) if 0 <= grid2[j] <= 80]
    if c1_idx and c2_idx:
        print("cv1\\cv2 " + "  ".join("%5.0f" % grid2[j] for j in c2_idx[:15]))
        for i in c1_idx:
            vals = ""
            for j in c2_idx[:15]:
                tag = "R" if real[i,j] else "."
                vals += "%5.1f%s" % (pmf[i,j], tag)
            print("%.0f     " % grid1[i] + vals)

    out = HERE / "pmf_ladder_final" / "cleft_entry.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_windows": len(windows),
        "free": [round(free[0],1), round(free[1],1), round(free[2],2)],
        "bound": [round(bound[0],1), round(bound[1],1), round(bound[2],2)],
        "saddle": [round(saddle[0],1), round(saddle[1],1), round(saddle[2],2)],
        "saddle_real": bool(real_saddle),
        "dG_on_kcalmol": round(dG_on,2),
        "dG_bind_kcalmol": round(bound[2]-free[2],2),
        "k_on_TST_M_s": k_on if dG_on>=0 else None,
        "k_on_phys_M_s": k_on_phys,
        "path_length": len(path),
        "path_real_frac": round(real_frac,2),
    }, indent=2, default=float))
    print("\nSaved", out)

if __name__ == "__main__":
    main()
