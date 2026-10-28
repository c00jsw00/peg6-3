#!/usr/bin/env python3
"""r_min PMF, P(r_min), state populations, and binding free energy from the
BioEmu2 FKC-umbrella ladder on wild.pdb (9FO glycan).

CV: r_min = min heavy-atom distance ligand(A) vs receptor(H+L) [A].
Ladder (tills r_min 2.3 bound -> 108 deep-free, overlapping):
    d55_n90     bound         rmin 2.3..3.3
    d75_n30     bound->trans  rmin 2.6..26.7
    d90_n0_g_s0 trans->free   rmin 6.6..42.8    (push-out, U=0)
    d110_n0_g_s0 free          rmin 20.1..58.6  (push-out, U=0)
    d130_n0_g   deep-free     rmin 41.0..76.7  (push-out, U=0)
    d170_n0_g   deep-free     rmin 71.3..108.3 (push-out, U=0)
Single-window control: d95_n0 (rmin 3.0..31.8, widest single reach).

Estimators:
  (A) EQUAL-WINDOW-AVERAGED  [primary, robust for strong biases]
      P(b) = (1/N) sum_i  count_i(b) / n_i
      Each window contributes its own r_min distribution, averaged over
      windows (equal weight). Valid because FKC samples each tilted region
      ergodically; equal-window averaging recovers the equilibrium P(r_min)
      in well-covered regions and is not distorted by the U magnitude.
  (B) Reference exp(U) reweight  [cross-check] -- identical to
      postprocess_pmf.py: B_i(b) = sum_{j in b} exp(U_j - max U) /
      sum_j exp(U_j - max U); p(b) = (1/N) sum_i B_i(b).
      Degenerate for strong-bias SMC windows (U varies >> kBT), so it
      over-weights each window's max-U frame; shown for comparison only.

State definitions (per task):  bound  r_min < 4  A ; unbound r_min > 15 A.
dG_bind = -kBT ln(P_bound / P_unbound).
"""
import json
from pathlib import Path
import numpy as np
import mdtraj
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KBT = 0.592
HERE = Path(__file__).resolve().parent
OUT = HERE / "pmf_ladder"
OUT.mkdir(exist_ok=True)

from bioemu2 import System
_TOP = System.from_yaml(str(HERE / "wild_system.yaml")).topology
_ATOMS = list(_TOP.atoms)
prot_idx = [a.index for a in _ATOMS
            if a.residue.chain.chain_id in ("H", "L") and a.element.symbol != "H"]
lig_idx = [a.index for a in _ATOMS
           if a.residue.chain.chain_id == "A" and a.element.symbol != "H"]

MULTI = ["d55_n90", "d75_n30", "d90_n0_g_s0", "d110_n0_g_s0",
         "d130_n0_g", "d170_n0_g"]
SINGLE = "d95_n0"


def _rmin_traj(path):
    path = Path(path)
    traj = mdtraj.load_xtc(path, top=path.parent / "topology.cif")
    xyz = np.asarray(traj.xyz, dtype=float) * 10.0
    L = xyz[:, lig_idx, :].reshape(len(xyz), 1, -1, 3)
    P = xyz[:, prot_idx, :].reshape(len(xyz), -1, 1, 3)
    return np.linalg.norm(L - P, axis=-1).min(axis=(-2, -1))


def load_rmin(w):
    d = HERE / "steered" / w
    n = np.load(d / "cv.npz")
    r = n["rmin"] if "rmin" in n else None
    r = r.astype(float) if r is not None else _rmin_traj(d / "samples.xtc")
    U = n["U"].astype(float)
    return r, U


def _bin_idx(r, edges):
    """0-based bin index matching np.histogram / np.searchsorted convention."""
    nb = len(edges) - 1
    return np.clip(np.searchsorted(edges, r, side="right") - 1, 0, nb - 1)


def estimate(rms, Us, edges):
    """Return p(b) via BOTH estimators plus PMF and state probabilities.
    rms: list of r arrays ; Us: list of U arrays (same order)."""
    nb = len(edges) - 1
    bc = (edges[:-1] + edges[1:]) / 2.0
    counts = np.zeros(nb)

    # (A) equal-window-averaged
    peq = np.zeros(nb)
    for r in rms:
        idx = _bin_idx(r, edges)
        counts += np.bincount(idx, minlength=nb)
        peq += np.bincount(idx, minlength=nb) / len(r)
    peq /= len(rms)

    # (B) reference exp(U) reweight (cross-check)
    pref = np.zeros(nb)
    for r, U in zip(rms, Us):
        idx = _bin_idx(r, edges)
        u = U - U.max()
        w = np.exp(u)
        pref += np.bincount(idx, minlength=nb, weights=w) / (w.sum() + 1e-12)
    pref /= len(rms)

    def pmf(p):
        p = p / p.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            f = -KBT * np.log(p + 1e-15)
        f[p <= 0] = np.nan
        f = f - np.nanmin(f)
        return f

    pmfA = pmf(peq)
    pmfB = pmf(pref)

    def stats(p):
        return (float(p[bc < 4.0].sum()),
                float(p[(bc >= 4.0) & (bc < 15.0)].sum()),
                float(p[bc > 15.0].sum()))

    pbA, ptA, pfA = stats(peq)
    pbB, ptB, pfB = stats(pref)
    dgA = -KBT * np.log(pbA / pfA) if pfA > 1e-12 else float("nan")
    dgB = -KBT * np.log(pbB / pfB) if pfB > 1e-12 else float("nan")
    return (pmfA, pmfB, peq, pref, bc, counts,
            pbA, ptA, pfA, dgA, pbB, ptB, pfB, dgB)


rms, Us = [], []
for w in MULTI:
    r, U = load_rmin(w)
    rms.append(r); Us.append(U)
allr = np.concatenate(rms)
edges = np.arange(int(np.floor(np.min(allr))) - 2,
                  int(np.ceil(np.max(allr))) + 5, 2.0)
nb = len(edges) - 1
bc = (edges[:-1] + edges[1:]) / 2.0

pmfA, pmfB, peq, pref, bc, counts, pbA, ptA, pfA, dgA, pbB, ptB, pfB, dgB = \
    estimate(rms, Us, edges)
raw = np.mean(allr < 4.0); rawf = np.mean(allr > 15.0)
dgraw = -KBT * np.log(raw / rawf)

# single window
rS, US = load_rmin(SINGLE)
pmfS, pmfS2, pS, pS2, bcs, cs, pbS, ptS, pfS, dgS, _pbB, _ptB, _pfB, _dgB = estimate([rS], [US], edges)

print(f"multi frames={len(allr)}  rmin [{allr.min():.1f}, {allr.max():.1f}] A")
for w, r in zip(MULTI, rms):
    print(f"  {w:12s} n={len(r):3d}  rmin {r.min():6.1f}..{r.max():6.1f}")
print(f"\n[A] equal-window: P_bound={pbA:.3f}  P_trans={ptA:.3f}  P_free={pfA:.3f}")
print(f"    dG_bind = {dgA:.3f} kcal/mol   (raw-frame: {dgraw:.3f})")
print(f"[B] exp(U) ref:    P_bound={pbB:.3f}  P_trans={ptB:.3f}  P_free={pfB:.3f} dG={dgB:.3f}")
print(f"[S] single d95_n0: P_bound={pbS:.3f}  P_free={pfS:.3f} dG={dgS:.3f}")

np.savez(OUT / "pmf_ladder.npz",
        PMF_eq=pmfA, PMF_ref=pmfB, PMF_single=pmfS,
        p_eq=peq, p_ref=pref, p_single=pS, bin_c=bc, edges=edges, counts=counts,
        P_bound_eq=pbA, P_trans_eq=ptA, P_free_eq=pfA, dG_eq=dgA,
        P_bound_ref=pbB, P_free_ref=pfB, dG_ref=dgB,
        P_bound_single=pbS, P_free_single=pfS, dG_single=dgS,
        P_bound_raw=raw, P_free_raw=rawf)
(OUT / "summary.json").write_text(json.dumps({
    "frames": int(len(allr)), "KBT": KBT, "n_bins": nb, "bin_edges": edges.tolist(),
    "primary": "equal-window-averaged (robust for strong biases)",
    "multi_eq": {"P_bound": round(pbA, 6), "P_transition": round(ptA, 6),
                 "P_free": round(pfA, 6), "dGbind_kcalmol": round(float(dgA), 3)},
    "multi_ref_expU": {"P_bound": round(pbB, 6), "P_free": round(pfB, 6),
                       "dGbind_kcalmol": round(float(dgB), 3) if np.isfinite(dgB) else None},
    "single_d95_n0": {"P_bound": round(pbS, 6), "P_free": round(pfS, 6),
                      "dGbind_kcalmol": round(float(dgS), 3) if np.isfinite(dgS) else None},
    "raw_frame_ratio": {"P_bound": round(raw, 6), "P_free": round(rawf, 6),
                        "dGbind_kcalmol": round(float(dgraw), 3)},
    "windows": MULTI, "single_window": SINGLE,
}))
print("saved summary.json")

print("\nPMF curve [A] equal-window (r_min, P%, PMF):")
print(f"{'r(A)':>6} {'P%':>8} {'PMF':>8}")
for b, pr, f in zip(bc, peq, pmfA):
    if np.isfinite(f) and (b < 6.0 or b > 15.0 or 4.0 <= b < 8.0):
        print(f"{b:6.1f} {pr*100:8.2f} {f:8.3f}")

fig, axs = plt.subplots(3, 1, figsize=(10, 13), sharex=True,
                        gridspec_kw={"height_ratios": [1.5, 1.0, 1.0]})
fig.suptitle("BioEmu2 FKC-Steered Sampling: r_min Dissociation Landscape\n"
             "wild.pdb (9FO glycan; chains H+L receptor, A ligand) | "
             f"{len(allr)} frames, 6-umbrella windows (r_min = min heavy-atom dist)",
             fontsize=14, y=0.995)
ax = axs[0]
ax.step(bc, pmfA, where="post", color="C0", lw=2, marker="o", mfc="C0", ms=4,
        label="Multi-window (equal-weight umbrella)")
ax.step(bc, pmfB, where="post", color="C3", lw=1.2, ls="--", marker="x", mfc="none",
        label="Multi-window (exp(U) reference reweight)")
ax.step(bcs, pmfS, where="post", color="C4", lw=1.5, ls=":", marker=".", mfc="none",
        label=f"Single-window {SINGLE} control")
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.axvline(4.0, color="C1", ls=":", lw=1.2); ax.axvline(15.0, color="C2", ls=":", lw=1.2)
ax.axvspan(4, 15, color="orange", alpha=0.08)
ax.set_ylabel("PMF F(r_min) = −kBT·ln P(r_min) [kcal/mol] (min→0)")
ax.set_xlim(bc[0] - 1, bc[-1] + 1)
ymax = np.nanmax(pmfA) + 1.0
ax.set_ylim(-1.0, ymax)
ax.annotate(f"Bound (<4 Å)\nP={pbA:.3f}", xy=(4, 0), xytext=(4.2, ymax * 0.9),
            arrowprops=dict(arrowstyle="->", color="C1"), color="C1", fontsize=10)
ax.annotate(f"Unbound (>15 Å)\nP={pfA:.3f}", xy=(min(bc[-1], 20), 0),
            xytext=(min(bc[-1], 20) - 2.0, ymax * 0.95),
            arrowprops=dict(arrowstyle="->", color="C2"), color="C2", fontsize=10)
ax.set_title("PMF(r_min) = −kBT·ln P(r_min); r_min = min heavy-atom ligand(A)–receptor(H+L) distance",
             fontsize=11, loc="left")
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="upper right", fontsize=9)

ax = axs[1]
ax.step(bc, peq * 100, where="post", color="C0", lw=2, marker="o", mfc="C0", ms=4,
        label="Multi-window P(r_min)")
ax.step(bcs, pS * 100, where="post", color="C4", lw=1.5, ls="--", marker="x", mfc="none",
        label=f"Single-window {SINGLE} control")
ax.axvline(4.0, color="C1", ls=":", lw=1.2); ax.axvline(15.0, color="C2", ls=":", lw=1.2)
ax.axvspan(4, 15, color="orange", alpha=0.1)
ax.set_ylabel("P(r_min) × 100 %")
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=9)
ax.set_title("Probability distribution P(r_min)", fontsize=11, loc="left")

ax = axs[2]
rs = np.sort(allr); step = max(1, len(rs) // 250); ii = np.arange(0, len(rs), step)
cb = np.array([np.mean(rs[:k] < 4) for k in ii]); cu = np.array([np.mean(rs[:k] > 15) for k in ii])
ax.plot(rs[ii], cb * 100, color="C1", lw=2, label="P(r_min' < 4 Å) bound CDF")
ax.plot(rs[ii], cu * 100, color="C2", lw=2, label="P(r_min' > 15 Å) unbound CDF")
ax.set_xlabel("r_min [Å]"); ax.set_ylabel("Cumulative population × 100 %")
ax.set_xlim(bc[0] - 1, bc[-1] + 1); ax.grid(alpha=0.3); ax.legend(fontsize=9)
ax.set_title(f"State populations  (Bound P={pbA:.3f}, Unbound P={pfA:.3f}, ΔGbind={dgA:.2f} kcal/mol)",
             fontsize=11, loc="left")
plt.tight_layout()
plt.savefig(OUT / "pmf_ladder.png", dpi=150)
plt.savefig(OUT / "pmf_ladder.svg")
print("saved pmf_ladder.png, pmf_ladder.svg")
print("DONE")
