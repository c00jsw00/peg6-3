#!/usr/bin/env python
"""Association umbrella: resolve the cleft-entrance barrier along the
ligand-approach path (low cv2 -> rising cv2 as ligand enters cleft).
Windows along the CV1/CV2 diagonal (cv1 35-70 A, cv2 0-180), overlapping.
Uses FKC SMC (resampling-only), per-window normalized reweight.
Outputs per window (steered/a{T}_n{N}_s{seed}/) so postprocess picks up.
Idempotent: skips windows with existing cv.npz.
"""
import sys, json, time
from pathlib import Path
import numpy as np

# reuse run_steered_sampling's window runner but write to 'a{T}_n{N}_s{seed}' dirs
sys.path.insert(0, str(Path(__file__).parent))
from run_steered_sampling import (run_window, OUT_ROOT, HERE, SEQ_H, SEQ_L)
from cv_potentials import ComComPotential, ContactCountPotential
from bioemu2 import FKCSampler, EmuSession, MSA, System, check_physicality, load_model, save_frames

HERE2 = Path(__file__).resolve().parent

def run_assoc_window(d_target, n_target, seed, out_root, n_samples, batch, model, system):
    out = out_root / f"a{int(round(d_target))}_n{int(round(n_target))}_s{seed}"
    if (out / "cv.npz").exists():
        print(f"[skip] {out}", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] assoc d={d_target:.0f}A n={n_target:.0f} "
          f"batch={batch} n={n_samples} seed={seed} ...", flush=True)
    p1 = ComComPotential(target=float(d_target), k=0.5, guidance_weight=1.0)
    p2 = ContactCountPotential(target=float(n_target), k=0.05, cutoff=4.5,
                               alpha=8.0, guidance_weight=1.0)
    msa = MSA(unpaired_a3m={"H": f">query\n{SEQ_H}\n",
                             "L": f">query\n{SEQ_L}\n"})
    emu = EmuSession(system, model=model, mode="equilibrium", msa=msa)
    sampler = FKCSampler(potentials=[p1, p2], weight_update="smc",
                         reheat="gamma", ess_threshold=0.8, num_steps=200)
    t0 = time.time()
    samples = emu.sample(num_samples=n_samples, batch_size=batch, seed=seed, sampler=sampler)
    dt = time.time() - t0
    emu.cache_clear()
    save_frames(samples, out, physicality=check_physicality(samples, system=system))
    import mdtraj
    traj = mdtraj.load_xtc(str(out / "samples.xtc"), top=str(out / "topology.cif"))
    cv1, cv2 = __import__("run_steered_sampling").compute_cvs(traj)
    U1, U2 = __import__("run_steered_sampling").potential_energy(traj, system, p1, p2)
    U = U1 + U2
    np.savez(out / "cv.npz", cv1=cv1, cv2=cv2, U1=U1, U2=U2, U=U)
    with open(out / "cv.csv", "w") as f:
        for i in range(len(cv1)):
            f.write(f"{i},{cv1[i]:.3f},{int(cv2[i])},{U1[i]:.4f},{U2[i]:.4f},{U[i]:.4f}\n")
    meta = {"d_target": float(d_target), "n_target": int(n_target), "n_samples": int(cv1.size),
            "cv1_range": [float(cv1.min()), float(cv1.max())],
            "cv2_range": [int(cv2.min()), int(cv2.max())], "U_mean": float(U.mean()),
            "seed": seed, "method": "FKC SMC association umbrella"}
    (out / "window.json").write_text(json.dumps(meta, indent=2))
    print(f"[{time.strftime('%H:%M:%S')}] d={d_target:.0f} n={n_target}: cv1 {cv1.min():.1f}..{cv1.max():.1f} "
          f"cv2 {int(cv2.min())}..{int(cv2.max())} U={U.mean():.2f}  {dt:.0f}s -> {out}", flush=True)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=25)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--probe", action="store_true",
                    help="reachability probe: 1 seed, 15 frames, 4 cleft-entrance windows")
    args = ap.parse_args()
    # diagonal of overlapping windows: cv1 35-70, cv2 0-180
    windows = [(70.0, 0.0), (62.0, 0.0), (55.0, 15.0), (50.0, 30.0),
               (48.0, 45.0), (45.0, 60.0), (44.0, 90.0), (42.0, 120.0),
               (40.0, 150.0), (36.0, 180.0)]
    if args.probe:
        windows = [(50.0, 30.0), (48.0, 60.0), (45.0, 90.0), (42.0, 120.0)]
        seeds = [0]
        nsamples = 15
    else:
        seeds, nsamples = args.seeds, args.n_samples
    model = load_model(device="cuda", precision="bf16-mixed")
    sys_obj = System.from_yaml(str(HERE2 / "wild_system.yaml"))
    for d, n in windows:
        for seed in seeds:
            run_assoc_window(d, n, seed, OUT_ROOT, nsamples, args.batch,
                             model, sys_obj)
    model._cache.clear() if hasattr(model, "_cache") else None
    print("DONE association umbrella", windows, flush=True)

if __name__ == "__main__":
    main()
