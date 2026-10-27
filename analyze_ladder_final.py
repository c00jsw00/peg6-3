#!/usr/bin/env python3
"""Final PMF reweighting over the FULL window set (43 windows) using the
project's postprocess_pmf.py method: per-window weight=exp((U-max U)/kBT),
B_i(b)=sum_j w_j / sum_j w_j, p=(1/N) sum_i B_i. Push-out windows store U=0.
New transition-region windows d80_n0_g_s*, d85_n0_g_s* (cv1 ~63-76 A) fill
the r_min 4-15 A gap. Reports P_bound/P_trans/P_free, dG_bind, PMF."""
import json
from pathlib import Path
import numpy as np
import mdtraj
from bioemu2 import System

HERE = Path("/home/c00jsw00/bioemu2")
KBT = 0.592
top = System.from_yaml(str(HERE/"wild_system.yaml")).topology
atoms = list(top.atoms)
prot = [a.index for a in atoms if a.residue.chain.chain_id in ("H","L") and a.element.symbol != "H"]
lig  = [a.index for a in atoms if a.residue.chain.chain_id == "A" and a.element.symbol != "H"]

def rmin_contacts(traj):
    xyz = np.asarray(traj.xyz, dtype=float)*10.0
    L = xyz[:, lig, :].reshape(len(xyz),1,-1,3); P = xyz[:, prot, :].reshape(len(xyz),-1,1,3)
    return np.linalg.norm(L-P, axis=-1).min(axis=(-2,-1))

import glob
wins=[]
for pat in ("d*_n*", "l*_k*_n*_s*", "d*_n0_g_s*"):
    for d in glob.glob(str((HERE/"steered")/pat)):
        d = Path(d)
        if d not in wins and ((d/"cv.npz").exists() or (d/"cv.csv").exists() or (d/"samples.xtc").exists()):
            wins.append(d)

rms=[]; Us=[]; names=[]
for d in wins:
    n = None
    try:
        n = np.load(d/"cv.npz")
    except Exception:
        n = None
    U = n["U"].astype(float) if n is not None and "U" in n else None
    r = n["rmin"].astype(float) if n is not None and "rmin" in n else None
    if r is None and n is not None and "cv1" in n:
        r = n["cv1"].astype(float)
    if r is None:
        try:
            traj = mdtraj.load_xtc(str(d/"samples.xtc"), top=str(d/"topology.cif"))
            r = rmin_contacts(traj)
        except Exception:
            continue
    if U is None:
        continue  # no steering energy available
    rms.append(r); Us.append(U); names.append(d.name)

allr = np.concatenate(rms)
print("frames=%d  r_min[%.1f..%.1f] A  n_windows=%d" % (len(allr), allr.min(), allr.max(), len(names)))
for n in sorted(names): print("  ", n)

lo = int(np.floor(np.min(allr))) - 2
hi = int(np.ceil(np.max(allr))) + 4
edges = np.arange(lo, hi + 1, 2.0)
bc = (edges[:-1] + edges[1:])/2.0
nb = len(edges)-1

prob = np.zeros(nb); counts = np.zeros(nb)
for w, r, U in zip(names, rms, Us):
    idx = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, nb-1)
    counts += np.bincount(idx, minlength=nb)
    u = U - U.max()
    with np.errstate(over="ignore"):
        wgt = np.exp(u)
    ws = float(wgt.sum())
    if not np.isfinite(ws) or ws <= 0:
        wgt = np.ones_like(r); ws = 1.0
    h = np.bincount(idx, minlength=nb, weights=wgt)
    prob += h / (ws + 1e-12)
prob = prob / len(rms)
prob = prob / prob.sum()
with np.errstate(divide="ignore", invalid="ignore"):
    pmf = -KBT * np.log(prob + 1e-15)
pmf = pmf - np.nanmin(pmf)

Pb = float(prob[bc < 4.0].sum()); Pf = float(prob[bc > 15.0].sum()); Pt = float(prob[(bc>=4)&(bc<15)].sum())
dg = -KBT*np.log(Pb/Pf) if Pf > 1e-12 else float("nan")
print("\nP_bound(<4)=%.3f  P_trans(4-15)=%.3f  P_free(>15)=%.3f  dG_bind=%.3f kcal/mol" % (Pb, Pt, Pf, dg))
print("\n  r_min(A) |  P% |  PMF(kcal/mol)")
for b, p, f in zip(bc, prob, pmf):
    print("  %6.1f | %5.1f | %8.3f" % (b, p*100, f))

outdir = Path(HERE/"pmf_ladder_final")
outdir.mkdir(exist_ok=True)
res = {"P_bound":Pb, "P_free":Pf, "P_trans":Pt, "dG_bind":dg,
       "n_frames":int(len(allr)), "n_windows":len(names),
       "pmf":list(pmf), "rmin_A":list(bc),
       "prob_pct":[round(x*100,3) for x in prob]}
(outdir/"summary.json").write_text(json.dumps(res, indent=2))
np.savez(outdir/"pmf_ladder.npz", pmf=pmf, rmin_A=bc, prob=prob)
print("\nWROTE", outdir/"summary.json", outdir/"pmf_ladder.npz")
print("DONE")
