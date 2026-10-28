#!/usr/bin/env python
import numpy as np, math
from pathlib import Path

HERE = Path(Path(__file__).resolve().parent); STEERED = HERE / "steered"; KBT = 0.592

def load_windows():
    for pat in ("d*_n*", "l*_k*_n*_s*", "a*_n*_s*"):
        for d in sorted(STEERED.glob(pat)):
            if not (d / "cv.npz").is_file():
                continue
            z = np.load(d / "cv.npz")
            yield z["cv1"].astype(float), z["cv2"].astype(int), z["U"].astype(float)

c1_edges = np.arange(8.0, 152.0, 2.0)
c2_edges = np.arange(0.0, 260.0, 5.0)
n1, n2 = len(c1_edges) - 1, len(c2_edges) - 1
prob = np.zeros((n1, n2)); nfr = np.zeros((n1, n2))
for cv1, cv2, U in load_windows():
    u = U - U.max(); w = np.exp(u)
    h, _, _ = np.histogram2d(cv1, cv2, weights=w, bins=(c1_edges, c2_edges))
    nh, _, _ = np.histogram2d(cv1, cv2, bins=(c1_edges, c2_edges))
    prob += h / (w.sum() + 1e-12); nfr += nh
prob /= prob.sum()
pmf = -KBT * np.log(prob + 1e-15)
pmf[prob <= 1e-15] = np.nan
med = float(np.nanmedian(pmf))
pmf = np.where(np.isnan(pmf), med, pmf)   # simple median fill (no interpolation mask bug)
grid1 = 0.5 * (c1_edges[:-1] + c1_edges[1:])
grid2 = 0.5 * (c2_edges[:-1] + c2_edges[1:])
free_pmf = 2.74  # free basin (cv1~137, cv2~2.5) from full-range PMF

def approach(cv1i):   # low-contact approach plateau (cv2 < 8)
    v = [pmf[cv1i, j] for j in range(n2) if grid2[j] < 8]
    return min(v)
def gate(cv1i):       # cleft-entrance gate (cv2 15-50)
    v = [pmf[cv1i, j] for j in range(n2) if 15 <= grid2[j] <= 50]
    return min(v)

print("cv1  approach(cv2<8)  gate(cv2 15-50)  local barrier  vs free")
rows = []
for i in range(n1):
    if 40 <= grid1[i] <= 62:
        a, g = approach(i), gate(i)
        rows.append((round(grid1[i], 1), round(a, 2), round(g, 2),
                     round(g - a, 2), round(g - free_pmf, 2)))
for r in rows:
    print("  %4.1f  %5.2f          %5.2f      %5.2f         %5.2f" % r)
gaps  = [r[3] for r in rows]
gfree = [r[4] for r in rows]
print("\nmin   local barrier (gate-approach) = %.2f kcal/mol" % min(gaps))
print("max   local barrier                 = %.2f kcal/mol" % max(gaps))
print("mean  local barrier                 = %.2f kcal/mol" % float(np.mean(gaps)))
print("min   absolute barrier (gate-free)  = %.2f kcal/mol" % min(gfree))
print("max   absolute barrier (gate-free)  = %.2f kcal/mol" % max(gfree))

print("\n  dG_on‡  ->  k_on (TST, M^-1/s)")
for dg in [4, 6, 8, 9, 10, 11, 12, 12.5]:
    print("    %.1f  ->  %.3e" % (dg, 6.21e12 * math.exp(-dg / KBT)))
