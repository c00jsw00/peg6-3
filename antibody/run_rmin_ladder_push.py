#!/usr/bin/env python3
"""r_min umbrella ladder driven by COM-COM push-out FKC guidance.

Separation is driven by the COM-COM push-out potential (proven to move the
ligand to the free plateau in this project; SMC resampling gets stuck at the
bound prior). The CV we REPORT and PMF on is r_min = min heavy-atom distance
ligand(A) vs receptor(H+L) [A]. We also record contacts (4.5 A heavy pairs)
and the steering U(x) per frame.

Windows (COM-COM push-out target T, stiffness k):
    T=25 (k=1.5),  T=40 (k=2.5), T=65 (k=4.0), T=90 (k=6.0),
    T=120 (k=8.0), T=150 (k=8.0)
Each window T samples p_base over COM-COM >= T, which maps (monotonically)
to r_min >= ~ (T - offset). Overlapping T tile the r_min axis bound->free.

Outputs per window (steered/rmin_ladder_<name>):
    samples.xtc, topology.cif, cv.npz (cv1_comcom, rmin, contacts, U,
    U_reweight=U), window.json
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
from cv_potentials import PushOutPotential

HERE = Path(__file__).resolve().parent
sys_obj = System.from_yaml(str(HERE / "wild_system.yaml"))
top = sys_obj.topology
atoms = list(top.atoms)
prot = [a.index for a in atoms if a.residue.chain.chain_id in ("H","L") and a.element.symbol!="H"]
lig  = [a.index for a in atoms if a.residue.chain.chain_id == "A" and a.element.symbol!="H"]

SEQ_H = ("QVQLVQSGSELKKPGASVKVSCKASGYTFTNYGMNWVRQAPGQGLEWMGWINTYTGQPIYANDFK"
         "GRFVFSLDTSVSTAYLQISSLKAEDTAVYYCARDWGPYWGQGTLVTVSSASTKGPSVFPLAPSSK"
         "STSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTY"
         "ICNVNHKPSNTKVDKRVEPKSCDK")
SEQ_L = ("DIVMTQSPDSLAVSLGERATINCKSSQSVLYSSNQMNYLAWYQQKPGQPPKLLIYWASTRESGVP"
         "DRFSGSGSGTDFTLTISSLQAEDVAVYYCLQYLSSWTFGGGTKLEIKRTVAAPSVFIFPPSDEQL"
         "KSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHK"
         "VYACEVTHQGLSSPVTKSFNRGEC")

# (name, COM-COM T [A], k [kcal/mol/A^2], guidance_weight)
WINDOWS = [
    ("ladder_T25",  25.0, 1.5, 5.0),
    ("ladder_T40",  40.0, 2.5, 5.0),
    ("ladder_T65",  65.0, 4.0, 5.0),
    ("ladder_T90",  90.0, 6.0, 5.0),
    ("ladder_T120",120.0, 8.0, 5.0),
    ("ladder_T150",150.0, 8.0, 5.0),
]
N_FRAMES = 12

def hard_rmin_contacts(traj):
    xyz = np.asarray(traj.xyz, dtype=np.float64)*10.0  # (F,N,3) A
    L = xyz[:, lig, :].reshape(len(xyz),1,-1,3)
    P = xyz[:, prot, :].reshape(len(xyz),-1,1,3)
    d = np.linalg.norm(L-P, axis=-1)  # (F,1,Na,Np) A
    rmin = d.min(axis=(-2,-1))
    cnt = (d <= 4.5).sum(axis=(-2,-1)).astype(np.int64)
    return rmin, cnt

def comcom(traj):
    xyz = np.asarray(traj.xyz, dtype=np.float64)
    pc = xyz[:, prot].mean(axis=1); lc = xyz[:, lig].mean(axis=1)
    return 10.0*np.linalg.norm(lc-pc, axis=1)

def run(name, T, k, gw, emu, model):
    out = HERE / "steered" / name
    out.mkdir(parents=True, exist_ok=True)
    resampled = False
    if (out / "samples.xtc").exists() and not (out / "cv.npz").exists():
        # already sampled (from a prior run); just recompute CVs + U
        pots = [PushOutPotential(d_target=float(T), k=k, guidance_weight=gw)]
        resampled = True
        dt = 0.0
    elif not (out / "cv.npz").exists():
        pots = [PushOutPotential(d_target=float(T), k=k, guidance_weight=gw)]
        sampler = FKCSampler(potentials=pots, weight_update="guidance",
                             ess_threshold=0.8, num_steps=200)
        print(f"[{time.strftime('%H:%M:%S')}] {name} T={T:g} k={k:g} n={N_FRAMES} seed=42 ...", flush=True)
        t0 = time.time()
        frames = emu.sample(num_samples=N_FRAMES, batch_size=1, seed=42, sampler=sampler)
        dt = time.time() - t0
        emu.cache_clear()
        save_frames(frames, out, physicality=check_physicality(frames, system=sys_obj))
        resampled = True
    if resampled:
        traj = mdtraj.load_xtc(str(out/"samples.xtc"), top=str(out/"topology.cif"))
        rmin, cnt = hard_rmin_contacts(traj)
        cv1 = comcom(traj)
        coords = torch.from_numpy(np.ascontiguousarray(
            10.0*np.asarray(traj.xyz, dtype=np.float32))).to("cuda", non_blocking=True)
        pots[0].prepare(sys_obj)
        with torch.no_grad():
            U = pots[0](coords, t=0.0)
            U = U.cpu().numpy().astype(float)
        np.savez(out/"cv.npz", cv1=cv1, rmin=rmin, contacts=cnt, U=U, U_reweight=U)
        meta = {"name":name,"T":T,"k":k,"n_samples":int(len(cv1)),
                "cv1_range":[float(cv1.min()),float(cv1.max())],
                "rmin_range":[float(rmin.min()),float(rmin.max())],
                "contacts_range":[int(cnt.min()),int(cnt.max())],
                "U_mean":float(U.mean()),"U_range":[float(U.min()),float(U.max())],
                "P_rmin_lt4":float(np.mean(rmin<4)), "P_rmin_gt15":float(np.mean(rmin>15)),
                "time_s":round(dt,1)}
        (out/"window.json").write_text(json.dumps(meta, indent=2))
        print(f"[{time.strftime('%H:%M:%S')}] {name}: cv1 {cv1.min():.1f}..{cv1.max():.1f} "
              f"rmin {rmin.min():.1f}..{rmin.max():.1f} "
              f"contacts {int(cnt.min())}..{int(cnt.max())} "
              f"P(r<4)={meta['P_rmin_lt4']:.2f} P(r>15)={meta['P_rmin_gt15']:.2f} "
              f"U_mean={meta['U_mean']:.1f} -> {out}", flush=True)
    else:
        print(f"[skip] {name} (cv.npz exists)")

def main():
    model = load_model(device="cuda", precision="bf16-mixed")
    msa = MSA(unpaired_a3m={"H": f">query\n{SEQ_H}\n", "L": f">query\n{SEQ_L}\n"})
    emu = EmuSession(sys_obj, model=model, mode="equilibrium", msa=msa)
    for name, T, k, gw in WINDOWS:
        run(name, T, k, gw, emu, model)
    emu.cache_clear()
    print("DONE ladder", flush=True)

if __name__ == "__main__":
    main()
