#!/usr/bin/env python
"""Umbrella ladder for the cleft-opening barrier.

Soft, overlapping COM-COM wells (k ~ 10-20) centered on CV1 44, 48, 52, 56
plus bound (T=28, k=5) and free (T=90, U=0) anchors.  Each window samples a
~1 A wide tilted distribution, so exp(U) reweighting is well-conditioned.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from bioemu2 import (EmuSession, FKCSampler, MSA, System, check_physicality,
                    load_model, save_frames)
import cv_potentials as cp
from run_steered_sampling import SEQ_H, SEQ_L, compute_cvs, potential_energy

HERE = Path(__file__).resolve().parent
OUT = HERE / "steered"
KBT = 0.592

def run_window(emu, system, T, cv2_t, k_well, gw_well, n_samples, steps, seed):
    outdir = OUT / f"l{int(round(T))}_k{int(k_well)}_n{int(cv2_t)}_s{seed}"
    if (outdir / "cv.npz").exists():
        print(f"[skip] {outdir}", flush=True)
        return
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] umbrella T={T:.0f}A cv2t={int(cv2_t)} "
          f"k={k_well} n={n_samples} steps={steps} seed={seed} ...", flush=True)
    p1 = cp.ComComPotential(target=T, k=k_well, guidance_weight=gw_well)
    p2 = cp.ContactCountPotential(target=cv2_t, k=0.0, cutoff=4.5, alpha=8.0,
                                  guidance_weight=0.0)
    sampler = FKCSampler(potentials=[p1, p2], weight_update="guidance",
                         ess_threshold=0.8, num_steps=steps)
    t0 = time.time()
    samples = emu.sample(num_samples=n_samples, batch_size=1, seed=seed,
                         sampler=sampler)
    dt = time.time() - t0
    emu.cache_clear()
    save_frames(samples, outdir, physicality=check_physicality(samples, system=system))
    import mdtraj
    traj = mdtraj.load_xtc(str(outdir / "samples.xtc"), top=str(outdir / "topology.cif"))
    cv1, cv2 = compute_cvs(traj)
    U1, U2 = potential_energy(traj, system, p1, p2)
    U = U1 + U2
    np.savez(outdir / "cv.npz", cv1=cv1.astype(float), cv2=cv2.astype(int),
             U1=U1, U2=U2, U=U)
    meta = {"T": T, "cv2t": cv2_t, "n": int(cv1.size), "k": k_well, "seed": seed,
            "cv1_range": [float(cv1.min()), float(cv1.max())],
            "cv2_range": [int(cv2.min()), int(cv2.max())],
            "U_mean": float(U.mean()), "U_max": float(U.max()),
            "U_spread": float(U.max() - U.min()),
            "U_max_kBT": float((U.max() - U.min()) / KBT),
            "method": "umbrella-ladder (soft COM-COM well + contact, REAL U)"}
    U_spread = float(U.max() - U.min())
    (outdir / "window.json").write_text(json.dumps(meta, indent=2))
    print(f"[{time.strftime('%H:%M:%S')}] T={T:.0f} cv2t={int(cv2_t)}: "
          f"cv1 {cv1.min():.1f}..{cv1.max():.1f} (n={n_samples})  "
          f"cv2 {int(cv2.min())}..{int(cv2.max())}  "
          f"U mean={U.mean():.1f} max={U.max():.1f} spread={U_spread:.1f} kcal/mol "
          f"({U_spread/KBT:.0f} kBT)  {dt:.0f}s -> {outdir}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True,
                    help='T cv2t pairs: "44 25 48 25 52 25 56 25"')
    ap.add_argument("--k", type=float, default=50.0)
    ap.add_argument("--gw", type=float, default=6.0)
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    args = ap.parse_args()
    vals = [float(t) for t in args.targets.replace(',', ' ').split()]
    pairs = [(vals[i], vals[i+1]) for i in range(0, len(vals), 2)]
    print(f"umbrella ladder: {[(round(T), int(ct)) for T, ct in pairs]} "
          f"k={args.k} gw={args.gw} n={args.n_samples} seeds={args.seeds}", flush=True)
    system = System.from_yaml(str(HERE / "wild_system.yaml"))
    model = load_model(device="cuda", precision="bf16-mixed")
    msa = MSA(unpaired_a3m={"H": f">query\n{SEQ_H}\n", "L": f">query\n{SEQ_L}\n"})
    emu = EmuSession(system, model=model, mode="equilibrium", msa=msa)
    for (T, ct) in pairs:
        for seed in args.seeds:
            run_window(emu, system, T=T, cv2_t=ct, k_well=args.k,
                       gw_well=args.gw, n_samples=args.n_samples,
                       steps=args.steps, seed=seed)
    emu.cache_clear()
    print("DONE.", flush=True)

if __name__ == "__main__":
    main()
