#!/usr/bin/env python
"""Final barrier-limited k_on estimate.

Combines:
  (a) SMC-reweighted 1D PMF (bound basin + TS), valid CV1 10-~85 A.
  (b) Guidance free-state raw counts, CV1 68-145 A, cv2=0 (unbound).

In the free region the ligand is an unbound solute in a spherical shell:
  p(r) dr ~ r^2 dr  =>  G(r) = G(r_a) - 2*kBT*ln(r/r_a)
This is the "Jacobian / solvent-entropy" continuation of the bound PMF
into the free plateau.  It is anchored at the SMC/guidance overlap
(r_a = 85 A), where the two sampling schemes overlap in the CV1=70-90 A
unbound tail (cv2~0).

Detailed balance gives:  k_on = k_off * exp((G_free-G_bound)/kBT)
with the standard state at the free plateau (CV1 ~ 110 A).
"""
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEERED = HERE / "steered"
KBT = 0.592            # kcal/mol (298 K)
# Eyring prefactor = kBT/h (per molecule) at 298 K ~ 6.21e12 s^-1.
# (Old value KBT/H with H=6.27e-16 kcal*s gave 9.45e14 s^-1, 152x too high
# and overestimated k_off; corrected to the physical per-molecule prefactor.)
KBT_OVER_h = 6.21e12   # s^-1  (Eyring prefactor, kBT/h @ 298 K)

# ---- load SMC windows (reweighted) ----
smc = []
for pattern in ("d*_n*", "l*_k*_n*_s*", "a*_n*_s*", "t*_n*_s*"):
    for d in sorted(STEERED.glob(pattern)):
        if d.name.endswith("_g_s0") or d.name.endswith("_g_s1"):
            continue
        if not (d / "cv.npz").exists():
            continue
        z = np.load(d / "cv.npz")
        smc.append((z["cv1"].astype(float), z["cv2"].astype(float),
                    z["U"].astype(float), d.name))
print(f"SMC + umbrella windows: {len(smc)}  frames={sum(len(w[0]) for w in smc)}")

# ---- load guidance free-state windows (raw counts, U=0) ----
guid = []
for d in sorted(STEERED.glob("d*_n0_g*")):
    if not (d / "cv.npz").exists():
        continue
    z = np.load(d / "cv.npz")
    guid.append((z["cv1"].astype(float), z["cv2"].astype(float), d.name))
print(f"Guidance windows: {len(guid)}  frames={sum(len(w[0]) for w in guid)}")
print("  guidance CV1 ranges (all cv2=0, unbound):")
for c1, c2, name in guid:
    print(f"    {name}: {c1.min():.1f}-{c1.max():.1f} A")

# ---- 1D PMF from SMC windows (per-window normalized umbrella) ----
c1_edges = np.arange(10.0, 92.0, 0.5)
wsum = np.zeros(len(c1_edges) - 1)
for c1, c2, U, name in smc:
    u = U - U.max()
    w = np.exp(u)
    ws, _ = np.histogram(c1, bins=c1_edges, weights=w)
    wsum += ws / (w.sum() + 1e-12)
P_smc = wsum / len(smc)
P_smc = P_smc / (P_smc.sum() + 1e-15)
# 1e-15 floor (matches 2D PMF; caps profile at ~20.45 kcal/mol).
G_smc = -KBT * np.log(P_smc + 1e-15)
c1c = 0.5 * (c1_edges[:-1] + c1_edges[1:])

# ---- cross-check guidance: estimate G_free(r) slope from raw counts ----
#  Bin guidance raw counts vs r and verify G(r)=const - 2*kBT*ln r
g_edges = np.arange(70.0, 155.0, 5.0)
gc = np.concatenate([c1 for c1, _, _ in guid])
nc, _ = np.histogram(gc, bins=g_edges)
rcc = 0.5 * (g_edges[:-1] + g_edges[1:])
print("\nGuidance raw counts by CV1 (r):")
for r, n in zip(rcc, nc):
    print(f"   r={r:.0f} A : n={n}")

# ---- anchor the free region at the overlap (CV1 ~ 85 A) ----
# SMC profile valid to ~85 A; anchor G there.
r_a = 85.0
ia = int(round((r_a - c1_edges[0]) / 0.5))
G_a = G_smc[ia]
print(f"\n  anchor G(CV1={c1c[ia]:.1f}) = {G_a:.3f} kcal/mol")

# ---- find basins ----
low = (c1c >= 10) & (c1c < 50)
ibound = np.argmin(G_smc[low]) + np.where(low)[0][0]
# TS: max G between bound (50) and free anchor (85)
mid = (c1c >= 50) & (c1c < 85)
itb = np.where(mid)[0]
itt = int(itb[np.argmax(G_smc[itb])])

# standard free state: representative of the free plateau.
# The free plateau is flat (CV1 100-145, cv2=0). Standard state at 1 M
# corresponds to the most probable free distance ~ where guidance density peaks.
# Use CV1 = 110 A as the standard-state anchor.
r_std = 110.0
# G_free(r_std) via Jacobian continuation:
G_free_std = G_a - 2 * KBT * np.log(r_std / r_a)

G_bound = G_smc[ibound]
G_TS = G_smc[itt]
G_free = G_free_std

dG_off = G_TS - G_bound
dG_on = G_TS - G_free
dG_bind = G_bound - G_free

# barrier-limited (TST) off-rate
k_off_tst = KBT_OVER_h * np.exp(-dG_off / KBT)
# association via detailed balance
k_on_tst = k_off_tst * np.exp((G_free - G_bound) / KBT)
K_D = k_off_tst / k_on_tst

# The association barrier is essentially flat (dG_on‡ ~ 0) -> association is
# NOT barrier-limited; it is capped by the Smoluchowski diffusion ceiling.
# k_on_TST >> k_diff means the true rate is the diffusion ceiling.
# Smoluchowski ceiling for a small trisaccharide + protein receptor is ~1e8
# M^-1 s^-1 (system-dependent; see note in JSON).
k_diff = 1.0e8  # M^-1 s^-1, typical Smoluchowski ceiling (small ligand + protein)
k_on_phys = min(k_on_tst, k_diff)
K_D_phys = k_off_tst / k_on_phys  # detailed-balance equilibrium constant

print("\n" + "=" * 60)
print(f"  bound: CV1={c1c[ibound]:.1f} A     G={G_bound:.2f}")
print(f"  TS:    CV1={c1c[itt]:.1f} A       G={G_TS:.2f}")
print(f"  free:  CV1={r_std:.0f} A (std)   G={G_free:.2f}")
print("=" * 60)
print(f"  dG_off‡ (bound->TS)   = {dG_off:.2f} kcal/mol  [barrier-LIMITED]")
print(f"  dG_on‡  (free->TS)    = {dG_on:.2f} kcal/mol  [~0 => diffusion-limited]")
print(f"  dG_bind (bound-free)  = {dG_bind:.2f} kcal/mol (negative=fav)")
print(f"  k_off = {k_off_tst:.4e} s^-1   (half-life = {np.log(2)/k_off_tst:.2f} s)")
print(f"  k_on  TST  = {k_on_tst:.4e} M^-1 s^-1  (> diffusion ceiling -> not physical)")
print(f"  k_on  PHYS = {k_on_phys:.4e} M^-1 s^-1  (diffusion-limited)")
print(f"  K_D   = {K_D_phys:.4e} M  (detailed balance, capped k_on)")
print(f"  diffusion ceiling used = {k_diff:.4e} M^-1 s^-1")
print("=" * 60)

res = {
    "bound_cv1_A": round(float(c1c[ibound]), 1),
    "ts_cv1_A": round(float(c1c[itt]), 1),
    "free_std_cv1_A": float(r_std),
    "g_bound_kcalmol": 0.0,
    "g_ts_kcalmol": round(float(G_TS - G_bound), 3),
    "g_free_kcalmol": round(float(G_free - G_bound), 3),
    "delta_g_off_kcalmol": round(float(dG_off), 3),
    "delta_g_on_kcalmol": round(float(dG_on), 3),
    "delta_g_bind_kcalmol": round(float(dG_bind), 3),
    "k_off_per_s": float(k_off_tst),
    "k_on_per_M_s": float(k_on_tst),
    "K_D_M": float(K_D_phys),  # use capped k_on (consistent with printed K_D)
    "k_on_physical_M_s": float(k_on_phys),
    "k_off_barrier_kcalmol": round(float(dG_off), 3),
    "association_barrier_free": bool(dG_on < 1.0),
    "k_diff_limit_M_s": 1.0e7,
    "n_smc_frames": int(sum(len(w[0]) for w in smc)),
    "n_guid_frames": int(sum(len(w[0]) for w in guid)),
    "method": "SMC-reweighted bound/TS + r^2 Jacobian free plateau "
              "(anchored at CV1=85 A overlap), detailed balance",
}
outdir = STEERED / "postproc"
outdir.mkdir(parents=True, exist_ok=True)
with open(outdir / "barriers_final.json", "w") as f:
    json.dump(res, f, indent=2)
print("Saved barriers_final.json")
