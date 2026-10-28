#!/usr/bin/env python3
"""Final r_min push-out FKC ladder — targets the ligand actually reaches.

CV: r_min = min heavy-atom distance ligand(A) vs receptor(H+L) [A].
Separation driven by one-sided push-out potential on COM-COM:
    U = 0.5 * k * soft_max(0, T - d_COMCOM)^2
Windows use strong push (k=20, the project's value) with T chosen so the
ligand reaches d_COMCOM >= T in a substantial fraction of frames (where U=0
and samples are unbiased). This makes per-frame importance reweighting
well-conditioned.

Per window (steered/lrN): samples.xtc, topology.cif,
  cv.npz (cv1_comcom, rmin, contacts, U), window.json
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

# (name, T[A], k[kcal/mol/A^2])
WINDOWS = [
    ("lr1",  20.0, 10.0),
    ("lr2",  50.0, 20.0),
    ("lr3",  75.0, 20.0),
    ("lr4",  95.0, 20.0),
    ("lr5", 110.0, 20.0),
    ("lr6", 120.0, 20.0),
]
N_FRAMES = 24
SEED = 42

def hard_rmin_contacts(traj):
    xyz = np.asarray(traj.xyz, dtype=np.float64)*10.0
    L = xyz[:, lig, :].reshape(len(xyz),1,-1,3)
    P = xyz[:, prot, :].reshape(len(xyz),-1,1,3)
    d = np.linalg.norm(L-P, axis=-1)
    rmin = d.min(axis=(-2,-1))
    cnt = (d <= 4.5).sum(axis=(-2,-1)).astype(np.int64)
    return rmin, cnt

def comcom(traj):
    xyz = np.asarray(traj.xyz, dtype=np.float64)
    pc = xyz[:, prot].mean(axis=1); lc = xyz[:, lig].mean(axis=1)
    return 10.0*np.linalg.norm(lc-pc, axis=1)

def run(name, T, k, emu, model):
    out = HERE / "steered" / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "cv.npz").exists():
        print(f"[skip] {name} (cv.npz exists)"); return
    pots = [PushOutPotential(d_target=float(T), k=k, guidance_weight=5.0)]
    sampler = FKCSampler(potentials=pots, weight_update="guidance",
                         ess_threshold=0.8, num_steps=200)
    print(f"[{time.strftime('%H:%M:%S')}] {name} T={T:g} k={k:g} n={N_FRAMES} seed={SEED} ...", flush=True)
    t0 = time.time()
    frames = emu.sample(num_samples=N_FRAMES, batch_size=1, seed=SEED, sampler=sampler)
    dt = time.time() - t0
    emu.cache_clear()
    save_frames(frames, out, physicality=check_physicality(frames, system=sys_obj))
    traj = mdtraj.load_xtc(str(out/"samples.xtc"), top=str(out/"topology.cif"))
    rmin, cnt = hard_rmin_contacts(traj)
    cv1 = comcom(traj)
    coords = torch.from_numpy(np.ascontiguousarray(
        10.0*np.asarray(traj.xyz, dtype=np.float32))).to("cuda", non_blocking=True)
    pots[0].prepare(sys_obj)
    with torch.no_grad():
        U = pots[0](coords, t=0.0).cpu().numpy().astype(float)
    np.savez(out/"cv.npz", cv1=cv1, rmin=rmin, contacts=cnt, U=U)
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
          f"U_mean={meta['U_mean']:.1f} {dt:.0f}s -> {out}", flush=True)

def main():
    model = load_model(device="cuda", precision="bf16-mixed")
    msa = MSA(unpaired_a3m={"H": f">query\n{SEQ_H}\n", "L": f">query\n{SEQ_L}\n"})
    emu = EmuSession(sys_obj, model=model, mode="equilibrium", msa=msa)
    for name, T, k in WINDOWS:
        run(name, T, k, emu, model)
    emu.cache_clear()
    print("DONE final ladder", flush=True)

if __name__ == "__main__":
    main()
