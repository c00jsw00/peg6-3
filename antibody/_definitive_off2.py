#!/usr/bin/env python
# Definitive cleft-exit (off-rate) barrier from the 2D PMF over all SMC/umbrella
# windows. Prints the cleft saddle PMF at two floors and the resulting dG_off/k_off.
import numpy as np
from pathlib import Path

KBT = 0.592
KBT_OVER_h = 6.21e12  # s^-1
S = Path(__file__).resolve().parent / "steered"

smc = []
for pat in ("d*_n*", "l*_k*_n*_s*", "a*_n*_s*", "t*_n*_s*"):
    for d in sorted(S.glob(pat)):
        if d.name.endswith("_g_s0") or d.name.endswith("_g_s1"):
            continue
        if not (d / "cv.npz").is_file():
            continue
        z = np.load(d / "cv.npz")
        smc.append((z["cv1"].astype(float), z["cv2"].astype(int), z["U"].astype(float), d.name))

n = len(smc)
frames = sum(len(w[0]) for w in smc)
print("%d windows, %d frames" % (n, frames))

c1e = np.arange(10.0, 120.0, 1.0)
c2e = np.arange(0.0, 270.0, 5.0)
prob = np.zeros((len(c1e) - 1, len(c2e) - 1))
nfr = np.zeros((len(c1e) - 1, len(c2e) - 1), dtype=float)
for cv1, cv2, U, name in smc:
    u = U - U.max(); w = np.exp(u)
    h, _, _ = np.histogram2d(cv1, cv2, weights=w, bins=(c1e, c2e))
    nh, _, _ = np.histogram2d(cv1, cv2, bins=(c1e, c2e))
    prob += h / (w.sum() + 1e-12); nfr += nh
g1 = 0.5 * (c1e[:-1] + c1e[1:])   # row centers
g2 = 0.5 * (c2e[:-1] + c2e[1:])   # col centers
G15 = -KBT * np.log(prob + 1e-15)  # cap 20.45
G20 = -KBT * np.log(prob + 1e-20)  # cap 27.26

rowmask = g1 < 35
colmask = g2 > 180
Gb = float(np.nanmin(G15[rowmask][:, colmask]))
print("bound basin: G=%.2f (low cv1, cv2>180)" % Gb)

def saddle_report(floor, name):
    G = G15 if "1e-15" in name else G20
    rm = (g1 >= 44) & (g1 <= 60)
    cm = (g2 >= 10) & (g2 <= 90)
    g1m = g1[rm]; g2m = g2[cm]
    sub = G[rm][:, cm]
    real = nfr[rm][:, cm] > 0
    if not real.any():
        print("  %s: NO real frames in saddle" % name); return
    masked = np.where(real, sub, -1e18)
    best = float(np.nanmax(masked))
    idx = int(np.argmax(masked))
    row, col = divmod(idx, sub.shape[1])
    nframes = int(nfr[rm][row, col])
    print("  %s: max REAL PMF = %.2f at cv1=%.0f cv2=%.0f (frames=%d)" % (
        name, best, float(g1m[row]), float(g2m[col]), nframes))
    dG = best - Gb
    k_off = KBT_OVER_h * np.exp(-dG / KBT)
    print("      dG_off = %.2f kcal/mol -> k_off = %.4e s^-1  t1/2 = %.1f s" % (dG, k_off, np.log(2)/k_off))
    exp_dg = -KBT * np.log(2.25e-3 / KBT_OVER_h)
    print("      exp k_off=2.25e-3 needs dG_off=%.2f; gap = %+.2f kcal/mol" % (exp_dg, dG - exp_dg))

print("\n--- cleft-exit saddle (cv1 44-60, cv2 10-90) ---")
saddle_report(1e-15, "floor=1e-15 (cap 20.45)")
saddle_report(1e-20, "floor=1e-20 (cap 27.26)")

print("\n=== TS frame density: cv1 50.0-54.5, cv2 10-60 ===")
total = 0
for cv1, cv2, U, name in smc:
    m = (cv1 >= 50.0) & (cv1 <= 54.5) & (cv2 >= 10) & (cv2 <= 60)
    if m.any():
        total += int(m.sum())
        print("  %s: %d frames (cv1 %.1f, cv2 %.0f)" % (name, int(m.sum()), float(cv1[m].mean()), float(cv2[m].mean())))
print("  TOTAL frames in cleft-exit TS region: %d" % total)
