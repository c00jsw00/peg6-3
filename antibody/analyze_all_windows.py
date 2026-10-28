#!/usr/bin/env python3
"""Authoritative 1D r_min PMF using the project's method (postprocess_pmf.py)
applied to r_min, over the FULL window set: all d*_n* SMC, all l*_k80* umbrella,
and all push-out (d*_n0_g*) windows. Reweighting: per-window weight_j =
exp((U_j - max U)/kBT); B_i(b)=sum_{j in b}w_j/sum_j w_j; p(b)=(1/N)sum_i B_i.
Push-out windows store U=0 -> raw frame counts (reference method)."""
import json, time
from pathlib import Path
import numpy as np
import mdtraj
import cv_potentials as cp
from bioemu2 import System

HERE = Path(__file__).resolve().parent
KBT = 0.592
top = System.from_yaml(str(HERE/"wild_system.yaml")).topology
atoms = list(top.atoms)
prot = [a.index for a in atoms if a.residue.chain.chain_id in ("H","L") and a.element.symbol != "H"]
lig  = [a.index for a in atoms if a.residue.chain.chain_id == "A" and a.element.symbol != "H"]

def rmin_contacts(traj):
    xyz = np.asarray(traj.xyz, dtype=float)*10.0
    L = xyz[:, lig, :].reshape(len(xyz),1,-1,3); P = xyz[:, prot, :].reshape(len(xyz),-1,1,3)
    d = np.linalg.norm(L-P, axis=-1)
    return d.min(axis=(-2,-1)), (d<=4.5).sum(axis=(1,2)).astype(int)

# gather all windows (same glob patterns as postprocess_pmf.py)
import glob

dirs = []
for pat in ("d*_n*", "l*_k*_n*_s*", "d*_n0_g_s*"):
    for d in glob.glob(str((HERE / "steered") / pat)):
        d = Path(d)
        if (d/"cv.npz").exists() or (d/"cv.csv").exists() or (d/"samples.xtc").exists():
            dirs.append(d)
dirs = sorted(set(dirs))
print(f"{len(dirs)} windows:", [Path(d).name for d in dirs], flush=True)

rms, Us, names = [], [], []
for d in dirs:
    w = Path(d).name
    n = np.load(d/"cv.npz")
    U = n["U"].astype(float)
    # r_min from cv.npz if present else xtc
    r = n["rmin"].astype(float) if "rmin" in n else None
    if r is None:
        traj = mdtraj.load_xtc(str(d/"samples.xtc"), top=str(d/"topology.cif"))
        r, _ = rmin_contacts(traj)
    if r is None:
        print(f"  {w}: could not get r_min, skipping"); continue
    rms.append(r); Us.append(U); names.append(w)

allr = np.concatenate(rms)
lo = int(np.floor(np.min(allr))) - 2
hi = int(np.ceil(np.max(allr))) + 4
edges = np.arange(lo, hi + 1, 2.0)
nb = len(edges) - 1
bc = (edges[:-1] + edges[1:]) / 2.0

# reference reweighting: exp(+U) per window, shifted; push-out have U=0
prob = np.zeros(nb); counts = np.zeros(nb); info = {}
for w, r, U in zip(names, rms, Us):
    idx = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, nb - 1)
    counts += np.bincount(idx, minlength=nb)
    u = U - U.max()
    with np.errstate(over="ignore"):
        wgt = np.exp(u)
    ws = float(wgt.sum())
    if not np.isfinite(ws) or ws <= 0:
        wgt = np.ones_like(r); ws = 1.0
    h = np.bincount(idx, minlength=nb, weights=wgt)
    prob += h / (ws + 1e-12)
    info[w] = (r.min(), r.max(), float(U.mean()))
prob = prob / len(rms)
prob = prob / prob.sum()
with np.errstate(divide="ignore", invalid="ignore"):
    pmf = -KBT * np.log(prob + 1e-15)
pmf[prob <= 0] = np.nan
if np.nanmin(pmf) is not None:
    pmf = pmf - np.nanmin(pmf)
Pb = float(prob[bc < 4.0].sum()); Pf = float(prob[bc > 15.0].sum()); Pt = float(prob[(bc>=4)&(bc<15)].sum())
dg = -KBT*np.log(Pb/Pf) if Pf > 1e-12 else float("nan")

print(f"\nframes={len(allr)} rmin[{allr.min():.1f},{allr.max():.1f}] A")
print(f"P_bound(<4)={Pb:.3f}  P_trans={Pt:.3f}  P_free(>15)={Pf:.3f}  dG_bind={dg:.3f} kcal/mol")
print("\nPMF(r_min)  [A, P%, PMF]:")
for b, p, f in zip(bc, prob, pmf):
    if np.isfinite(f) and (b < 6 or b > 15 or (4 <= b < 8)):
        print(f"  {b:6.1f}  {p*100:7.2f}%  {f:8.3f}")
print("\nper-window (r_min range, U mean):")
for w,(rmin,rmax,Um) in info.items():
    print(f"  {w:16s} rmin[{rmin:7.1f},{rmax:7.1f}] U_mean={Um:9.1f}")
print("DONE")
