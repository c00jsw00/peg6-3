#!/usr/bin/env python
"""Re-analyse the cleft-exit (dissociation) barrier after TS-refinement windows.

Recomputes the 1D exit-path PMF (bound -> TS) using all SMC/umbrella windows
INCLUDING the new t*_n*_s* TS-refinement windows.  Prints k_off at the standard
floor (1e-15 -> cap 20.45) AND at a lower floor (1e-20 -> cap 24.9) to see whether
the barrier is floor-limited or genuinely under-resolved.

Experimental targets: k_off=2.25e-3 s^-1 (dG_off=21.05), k_on=1.1e4, K_D=2.04e-7.
"""
import numpy as np, math
from pathlib import Path

HERE = Path(Path(__file__).resolve().parent); STEERED = HERE / "steered"; KBT = 0.592
KBT_OVER_h = 6.21e12   # s^-1 (kBT/h @ 298 K)

def load_windows():
    for pat in ("d*_n*", "l*_k*_n*_s*", "a*_n*_s*", "t*_n*_s*"):
        for d in sorted(STEERED.glob(pat)):
            if d.name.endswith("_g_s0") or d.name.endswith("_g_s1"):
                continue
            if not (d / "cv.npz").is_file():
                continue
            z = np.load(d / "cv.npz")
            yield d.name, z["cv1"].astype(float), z["cv2"].astype(float), z["U"].astype(float)

# 1D exit-path PMF (cv1 10-92, 0.5 A) - same as analyze_kon_final
c1_e = np.arange(10.0, 92.0, 0.5)
wsum = np.zeros(len(c1_e) - 1)
ns = []
frames = 0
for name, cv1, cv2, U in load_windows():
    u = U - U.max(); w = np.exp(u)
    ws, _ = np.histogram(cv1, bins=c1_e, weights=w)
    wsum += ws / (w.sum() + 1e-12)
    ns.append(name); frames += len(cv1)
P = wsum / len(ns)
P = P / (P.sum() + 1e-15)
c1c = 0.5 * (c1_e[:-1] + c1_e[1:])

print(f"{len(ns)} windows, {frames} frames (incl. t* TS-refinement)")
print(f"  t* windows: {[n for n in ns if n.startswith('t')]}")

def compute(floor):
    G = -KBT * np.log(P + floor)   # zero-prob cells -> floor value (barrier-limited, as in analyze_kon_final)
    def argmin_safe(a, m):
        idx = np.where(m)[0]
        a = np.asarray(a[idx], dtype=float)
        return int(idx[np.nanargmin(a)])
    ib = argmin_safe(G, (c1c >= 10) & (c1c < 50))
    Gb = G[ib]
    iTS = argmin_safe(-G, (c1c >= 50) & (c1c < 85))  # max G
    GTS = G[iTS]
    dG = GTS - Gb
    k_off = KBT_OVER_h * np.exp(-dG / KBT)
    t_half = math.log(2) / k_off
    return ib, iTS, Gb, GTS, dG, k_off, t_half, c1c

for floor, cap in [(1e-15, KBT * -math.log(1e-15)),
                   (1e-20, KBT * -math.log(1e-20))]:
    ib, iTS, Gb, GTS, dG, k_off, t_half, cc = compute(floor)
    print(f"\n=== floor={floor:.0e} (cap {cap:.2f} kcal/mol) ===")
    print(f"  bound CV1={cc[ib]:.1f} A  G={Gb:.2f}")
    print(f"  TS    CV1={cc[iTS]:.1f} A  G={GTS:.2f}")
    print(f"  dG_off‡ = {dG:.2f} kcal/mol")
    print(f"  k_off   = {k_off:.4e} s^-1   (t1/2 = {t_half:.1f} s)")
    exp_dg = -KBT * math.log(2.25e-3 / KBT_OVER_h)
    print(f"  exp k_off=2.25e-3 -> dG_off={exp_dg:.2f}; gap = {dG-exp_dg:+.2f} kcal/mol")

# TS-region frame density (cv1 46-56, cv2 10-50)
print("\n=== cleft-exit TS frame density (cv1 46-56, cv2 10-50) ===")
for name, cv1, cv2, U in load_windows():
    m = (cv1 >= 46) & (cv1 <= 56) & (cv2 >= 10) & (cv2 <= 50)
    if m.any():
        print(f"  {name}: {int(m.sum())} frames  cv1 {cv1[m].mean():.1f} cv2 {cv2[m].mean():.0f}")
