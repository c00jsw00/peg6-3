#!/usr/bin/env python3
"""r_min ladder: COM-COM-driven steering windows, recording r_min (min heavy-atom
ligand-A vs receptor H+L), contacts (4.5 A), and steering U(x) per frame.

Windows (COM-COM target d, contact target n):
  bound/transition  -> SMC (resampling-only):   d=30 n=260, d=55 n=130, d=75 n=60, d=95 n=0
  free plateau      -> guidance push-out:        d=115 n=0, d=135 n=0
r_min is the CV we report; COM-COM drives the separation (r_min steering alone
fails: the ligand deforms to keep one atom near the protein).

Per window outputs (steered/rmin_d{n}/): samples.xtc, topology.cif,
  cv.npz (cv1_comcom, rmin, contacts, U_steering, U_reweight), window.json
"""
import json
import time
from pathlib import Path

import numpy as np
import mdtraj
import torch

from bioemu2 import (
    EmuSession, FKCSampler, MSA, System, check_physicality, load_model,
    save_frames,
)
from cv_potentials import ComComPotential, ContactCountPotential, PushOutPotential

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE / "steered"

SEQ_H = ("QVQLVQSGSELKKPGASVKVSCKASGYTFTNYGMNWVRQAPGQGLEWMGWINTYTGQPIYANDFK"
         "GRFVFSLDTSVSTAYLQISSLKAEDTAVYYCARDWGPYWGQGTLVTVSSASTKGPSVFPLAPSSK"
         "STSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTY"
         "ICNVNHKPSNTKVDKRVEPKSCDK")
SEQ_L = ("DIVMTQSPDSLAVSLGERATINCKSSQSVLYSSNQMNYLAWYQQKPGQPPKLLIYWASTRESGVP"
         "DRFSGSGSGTDFTLTISSLQAEDVAVYYCLQYLSSWTFGGGTKLEIKRTVAAPSVFIFPPSDEQL"
         "KSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHK"
         "VYACEVTHQGLSSPVTKSFNRGEC")

WINDOWS = [
    # (name, d_target, n_target, mode, n_samples, batch)
    ("rmin_d30",   30.0, 260.0, "smc",     10, 1),
    ("rmin_d55",   55.0, 130.0, "smc",    10, 1),
    ("rmin_d75",   75.0,  60.0, "smc",    10, 1),
    ("rmin_d95",   95.0,   0.0, "smc",    10, 1),   # single-window control
    ("rmin_d115", 115.0,   0.0, "push",   12, 1),
    ("rmin_d135", 135.0,   0.0, "push",   12, 1),
]

sys_obj = System.from_yaml(str(HERE / "wild_system.yaml"))
top = sys_obj.topology
atoms = list(top.atoms)
prot = [a.index for a in atoms if a.residue.chain.chain_id in ("H","L") and a.element.symbol!="H"]
lig  = [a.index for a in atoms if a.residue.chain.chain_id == "A" and a.element.symbol!="H"]

def hard_rmin_contacts(traj):
    """r_min (A) = min ligand-heavy/receptor-heavy distance per frame;
    contacts = number of such pairs within 4.5 A per frame (array of shape (F,))."""
    xyz = np.asarray(traj.xyz, dtype=np.float64)*10.0  # (F,N,3) A
    L = xyz[:, lig, :].reshape(len(xyz),1,-1,3)
    P = xyz[:, prot, :].reshape(len(xyz),-1,1,3)
    d = np.linalg.norm(L-P, axis=-1)  # (F,1,Na,Np) A
    rmin = d.min(axis=(-2,-1))  # (F,) A
    cnt = (d <= 4.5).sum(axis=(-2,-1)).astype(np.int64)  # (F,)
    return rmin, cnt

def comcom(traj):
    xyz = np.asarray(traj.xyz, dtype=np.float64)
    pc = xyz[:, prot].mean(axis=1); lc = xyz[:, lig].mean(axis=1)
    return 10.0*np.linalg.norm(lc-pc, axis=1)

def recompute_U(traj, potentials):
    xyz = np.asarray(traj.xyz, dtype=np.float32)*10.0  # A
    coords = torch.from_numpy(np.ascontiguousarray(xyz)).to("cuda", non_blocking=True)
    for p in potentials:
        p.prepare(sys_obj)
    with torch.no_grad():
        U = sum(p(coords, t=0.0) for p in potentials).cpu().numpy()
    return U

def run(name, d, n, mode, n_samples, batch, emu, model, seed=42):
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    xtc = out / "samples.xtc"
    if (out / "cv.npz").exists():
        print(f"[skip] {name} (cv.npz exists)")
        return
    if mode == "smc":
        p1 = ComComPotential(target=float(d), k=0.5, guidance_weight=1.0)
        p2 = ContactCountPotential(target=float(n), k=0.05, cutoff=4.5, alpha=8.0, guidance_weight=1.0)
        pots = [p1, p2]
        sampler = FKCSampler(potentials=pots, weight_update="smc", reheat="gamma",
                             ess_threshold=0.8, num_steps=200)
    else:  # push (guidance, free plateau)
        p1 = PushOutPotential(d_target=float(d), k=20.0, guidance_weight=5.0)
        p2 = ContactCountPotential(target=0.0, k=0.05, cutoff=4.5, alpha=8.0, guidance_weight=3.0)
        pots = [p1, p2]
        sampler = FKCSampler(potentials=pots, weight_update="guidance",
                             ess_threshold=0.8, num_steps=200)
    print(f"[{time.strftime('%H:%M:%S')}] {name} d={d:g} n={n:g} mode={mode} n={n_samples} "
          f"batch={batch} seed={seed} ...", flush=True)
    t0 = time.time()
    samples = emu.sample(num_samples=n_samples, batch_size=batch, seed=seed, sampler=sampler)
    dt = time.time() - t0
    emu.cache_clear()
    save_frames(samples, out, physicality=check_physicality(samples, system=sys_obj))
    traj = mdtraj.load_xtc(str(out/"samples.xtc"), top=str(out/"topology.cif"))
    rmin, cnt = hard_rmin_contacts(traj)
    cv1 = comcom(traj)
    U = recompute_U(traj, pots)
    # reweight U: push windows store flat 0 (free plateau, raw counts) per project recipe;
    # SMC windows keep real (soft) U.
    U_re = U.copy() if mode == "smc" else np.zeros_like(U)
    np.savez(out/"cv.npz", cv1=cv1, rmin=rmin, contacts=cnt, U=U, U_reweight=U_re)
    with open(out/"cv.csv", "w") as f:
        f.write("frame,cv1_A,rmin_A,contacts,U_kcalmol,Urew_kcalmol\n")
        for i in range(len(cv1)):
            f.write(f"{i},{cv1[i]:.3f},{rmin[i]:.3f},{int(cnt[i])},{U[i]:.3f},{U_re[i]:.3f}\n")
    meta = {"name": name, "d_target": d, "n_target": n, "mode": mode,
            "n_samples": int(len(cv1)), "seed": seed, "cv1_range": [float(cv1.min()), float(cv1.max())],
            "rmin_range": [float(rmin.min()), float(rmin.max())],
            "contacts_range": [int(cnt.min()), int(cnt.max())],
            "P_rmin_lt4": float(np.mean(rmin<4)), "P_rmin_gt15": float(np.mean(rmin>15)),
            "time_s": round(dt,1)}
    (out/"window.json").write_text(json.dumps(meta, indent=2))
    print(f"[{time.strftime('%H:%M:%S')}] {name}: rmin {rmin.min():.1f}..{rmin.max():.1f} "
          f"contacts {int(cnt.min())}..{int(cnt.max())} "
          f"P(r<4)={meta['P_rmin_lt4']:.2f} P(r>15)={meta['P_rmin_gt15']:.2f} {dt:.0f}s -> {out}", flush=True)

def main():
    model = load_model(device="cuda", precision="bf16-mixed")
    msa = MSA(unpaired_a3m={"H": f">query\n{SEQ_H}\n", "L": f">query\n{SEQ_L}\n"})
    emu = EmuSession(sys_obj, model=model, mode="equilibrium", msa=msa)
    for name, d, n, mode, n_samples, batch in WINDOWS:
        run(name, d, n, mode, n_samples, batch, emu, model)
    emu.cache_clear()
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
