#!/usr/bin/env python
import numpy as np, math
from pathlib import Path

HERE = Path(Path(__file__).resolve().parent); STEERED = HERE / "steered"; KBT = 0.592

def load_windows():
    for pat in ("d*_n*", "l*_k*_n*_s*", "a*_n*_s*"):
        for d in sorted(STEERED.glob(pat)):
            if d.name.endswith("_g_s0") or d.name.endswith("_g_s1"):
                continue
            if not (d / "cv.npz").is_file():
                continue
            z = np.load(d / "cv.npz")
            yield d.name, z["cv1"].astype(float), z["cv2"].astype(int), z["U"].astype(float)

# Fine grid over the cleft-exit TS region.
# cv1 44-62 A (bound ~47.2, exit to ~60), cv2 0-120
c1_e = np.arange(44.0, 63.0, 1.0)
c2_e = np.arange(0.0, 121.0, 5.0)
n1, n2 = len(c1_e) - 1, len(c2_e) - 1
prob, nfr, Usum = np.zeros((n1, n2)), np.zeros((n1, n2)), np.zeros((n1, n2))
names = []
for name, cv1, cv2, U in load_windows():
    u = U - U.max(); w = np.exp(u)
    h, _, _ = np.histogram2d(cv1, cv2, weights=w, bins=(c1_e, c2_e))
    nh, _, _ = np.histogram2d(cv1, cv2, bins=(c1_e, c2_e))
    prob += h / (w.sum() + 1e-12); nfr += nh
g1 = 0.5 * (c1_e[:-1] + c1_e[1:]); g2 = 0.5 * (c2_e[:-1] + c2_e[1:])
pmf = -KBT * np.log(prob + 1e-15)          # current 1e-15 floor
pmf_low = -KBT * np.log(prob + 1e-30)     # low floor
pmf = np.where(prob <= 0, np.nan, pmf)
real = nfr >= 2

print("=== cleft-exit region PMF (cv1 44-62, cv2 0-120, 1A x 5-contact) ===")
print("     cv2:  0  10  20  30  40  50  60  70  80  90  100 110")
print("   cv1")
for i in range(n1):
    row = f"  {g1[i]:4.0f} "
    for j in range(n2):
        if nfr[i, j] >= 2:
            row += f"{pmf[i,j]:5.1f} "
        elif nfr[i, j] == 1:
            row += f"{pmf[i,j]:5.0f} "
        else:
            row += "     "
    print(row)

# cleft-exit path: bound (cv1~47, high cv2) -> exit (cv2 < 5)
# Find the max PMF on the connected cleft-exit path from bound to exit.
# Define path region: cv1 44-60, cv2 10-70 (the cleft).
print("\n=== TS peak search (cv1 45-58, cv2 15-70, frames>=2) ===")
best = -np.inf; bestpos = None
for i in range(n1):
    if not (45 <= g1[i] <= 58): continue
    for j in range(n2):
        if not (15 <= g2[j] <= 70) or nfr[i, j] < 2: continue
        if pmf[i, j] > best:
            best = pmf[i, j]; bestpos = (i, j)
print(f"  max PMF (1e-15 floor) = {best:.2f} kcal/mol at cv1={g1[bestpos[0]]:.0f}, cv2={g2[bestpos[1]]:.0f} frames={int(nfr[bestpos])}")

# bound reference: cv1 44-49, cv2 20-50 min
bounds = []
for i in range(n1):
    if not (44 <= g1[i] < 49): continue
    for j in range(n2):
        if not (20 <= g2[j] <= 50) or nfr[i, j] < 2: continue
        bounds.append(pmf[i, j])
if bounds:
    G_bound = min(bounds)
    print(f"  bound min (cv1 44-49, cv2 20-50) = {G_bound:.2f}")
    print(f"  dG_off = {best - G_bound:.2f} kcal/mol")
    k_off = 6.21e12 * math.exp(-(best - G_bound) / KBT)
    print(f"  k_off = {k_off:.4e} s^-1")

# low floor peak
best2 = -np.inf; best2pos = None
for i in range(n1):
    if not (45 <= g1[i] <= 58): continue
    for j in range(n2):
        if not (15 <= g2[j] <= 70) or nfr[i, j] < 1: continue
        if pmf_low[i, j] > best2:
            best2 = pmf_low[i, j]; best2pos = (i, j)
print(f"\n  max PMF (1e-30 floor)  = {best2:.2f} kcal/mol at cv1={g1[best2pos[0]]:.0f}, cv2={g2[best2pos[1]]:.0f} frames={int(nfr[best2pos])}")
print(f"  floor 1e-15 caps at: {KBT * -math.log(1e-15):.2f} kcal/mol")
