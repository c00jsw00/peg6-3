#!/usr/bin/env python3
"""Transition-region windows: push-out on COM-COM, centered on r_min 4-15 A."""
import json, time
from pathlib import Path
import numpy as np, mdtraj
import cv_potentials as cp
from bioemu2 import (EmuSession, FKCSampler, MSA, System, load_model,
                     check_physicality, save_frames)

HERE = Path(__file__).resolve().parent
sys_obj = System.from_yaml(str(HERE / "wild_system.yaml"))
top = sys_obj.topology
atoms = list(top.atoms)
prot = [a.index for a in atoms if a.residue.chain.chain_id in ("H","L") and a.element.symbol != "H"]
lig  = [a.index for a in atoms if a.residue.chain.chain_id == "A" and a.element.symbol != "H"]
SEQ_H = ("QVQLVQSGSELKKPGASVKVSCKASGYTFTNYGMNWVRQAPGQGLEWMGWINTYTGQPIYANDFK"
         "GRFVFSLDTSVSTAYLQISSLKAEDTAVYYCARDWGPYWGQGTLVTVSSASTKGPSVFPLAPSSK"
         "STSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTY"
         "ICNVNHKPSNTKVDKRVEPKSCDK")
SEQ_L = ("DIVMTQSPDSLAVSLGERATINCKSSQSVLYSSNQMNYLAWYQQKPGQPPKLLIYWASTRESGVP"
         "DRFSGSGSGTDFTLTISSLQAEDVAVYYCLQYLSSWTFGGGTKLEIKRTVAAPSVFIFPPSDEQL"
         "KSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHK"
         "VYACEVTHQGLSSPVTKSFNRGEC")
N_FRAMES = 24

def hard_rmin_contacts(traj):
    xyz = np.asarray(traj.xyz, dtype=np.float64)*10.0
    L = xyz[:, lig, :].reshape(len(xyz),1,-1,3)
    P = xyz[:, prot, :].reshape(len(xyz),-1,1,3)
    d = np.linalg.norm(L-P, axis=-1)
    rmin = d.min(axis=(-2,-1)); cnt = (d <= 4.5).sum(axis=(1,2)).astype(int)
    return rmin, cnt

def run(name, T, k, emu, model):
    out = HERE/"steered"/name
    out.mkdir(parents=True, exist_ok=True)
    if (out/"cv.npz").exists():
        print(f"[skip] {name} (exists)"); return
    pots = [PushOutPotential(d_target=float(T), k=k, guidance_weight=5.0)]
    sampler = FKCSampler(potentials=pots, weight_update="guidance", ess_threshold=0.8, num_steps=200)
    print(f"[{time.strftime('%H:%M:%S')}] {name} T={T:g} k={k:g} n={N_FRAMES} ...", flush=True)
    t0 = time.time()
    frames = emu.sample(num_samples=N_FRAMES, batch_size=1, seed=42, sampler=sampler)
    dt = time.time()-t0
    emu.cache_clear()
    save_frames(frames, out, physicality=check_physicality(frames, system=sys_obj))
    traj = mdtraj.load_xtc(str(out/"samples.xtc"), top=str(out/"topology.cif"))
    rmin, cnt = hard_rmin_contacts(traj)
    coords = np.ascontiguousarray(10.0*np.asarray(traj.xyz, dtype=np.float64))
    with torch.no_grad():
        pots[0].prepare(sys_obj)
        U = pots[0](torch.from_numpy(coords).to("cuda"), t=0.0).cpu().numpy().astype(float)
    np.savez(out/"cv.npz", cv1=np.array([]), rmin=rmin, contacts=cnt, U=U)
    meta = {"name":name,"T":T,"k":k,"n":int(len(cv1:=rmin)),
            "rmin_range":[float(rmin.min()),float(rmin.max())],
            "contacts_range":[int(cnt.min()),int(cnt.max())],
            "P_rmin_lt4":float(np.mean(rmin<4)), "P_rmin_gt15":float(np.mean(rmin>15)),
            "time_s":round(dt,1)}
    (out/"window.json").write_text(json.dumps(meta, indent=2))
    print(f"[{time.strftime('%H:%M:%S')}] {name}: rmin {rmin.min():.1f}..{rmin.max():.1f} "
          f"contacts {int(cnt.min())}..{int(cnt.max())} "
          f"P(r<4)={meta['P_rmin_lt4']:.2f} P(r>15)={meta['P_rmin_gt15']:.2f} {dt:.0f}s", flush=True)

def main():
    import torch
    model = load_model(device="cuda", precision="bf16-mixed")
    msa = MSA(unpaired_a3m={"H": f">query\n{SEQ_H}\n", "L": f">query\n{SEQ_L}\n"})
    emu = EmuSession(sys_obj, model=model, mode="equilibrium", msa=msa)
    for name, T, k in [("tr_80", 80.0, 15.0), ("tr_85", 85.0, 18.0), ("tr_90", 90.0, 20.0),
                        ("tr_95", 95.0, 18.0)]:
        run(name, T, k, emu, model)
    emu.cache_clear()
    print("DONE transition windows")

if __name__ == "__main__":
    main()
