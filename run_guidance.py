#!/usr/bin/env python3
"""Guidance push-out sampling of the FREE state (CV1 > 90 A).

The SMC (resampling-only) windows cannot relocate the ligand beyond
CV1 ~ 81 A; the base distribution has no mass at CV1 > 90.  To reach the
genuine free state we use FKCSampler with weight_update="guidance" (backward
pass, cheap at batch=1) plus a one-sided PUSH-OUT potential

    U_push = 0.5*k*max(0, T - d)^2          d = ligand(A)-protein(H+L) COM-COM (A)

whose gradient drags the ligand out of the pocket to CV1 >= T.  The ligand
settles in [T-lag, T] with cv2 ~ 0 (fully unbound).

Reweighting / free-state PMF:
  The push-out U is steep in the region the ligand settles in, so umbrella
  reweighting by U_push would invert the free-region distribution.  Instead we
  estimate the free-state free energy directly from frame counts: we store a
  FLAT free-state potential (U = 0 for every frame) for each guidance window.
  The standard per-window-normalized umbrella reweight then reduces to a raw
  frame count, i.e.  p_base(free) ~ (1/N) * sum_j 1_{x_j in bin}.  The CV1 > 90 A
  bins are covered EXCLUSIVELY by these guidance windows (the SMC windows top
  out at ~81 A), so the free plateau PMF is a clean count and does not mix with
  the well-reweighted bound/cleft region (CV1 < 90 A).

Outputs per window (steered/d{T}_n0_g/): cv.npz, cv.csv, window.json,
samples.xtc, topology.cif, physicality.json  (same layout as the SMC driver).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from bioemu2 import (
    EmuSession, FKCSampler, MSA, System, check_physicality, load_model,
    save_frames,
)

import cv_potentials as cp
from run_steered_sampling import SEQ_H, SEQ_L, compute_cvs

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE / "steered"


def run_window(emu, T, n_samples, steps, seed, push_k, push_gw, out_root, model, system):
    outdir = out_root / f"d{int(round(T))}_n0_g_s{seed}"
    if (outdir / "cv.npz").exists():
        print(f"[skip] {outdir} (cv.npz exists)", flush=True)
        return
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] push-out d={T:.0f}A  k={push_k} gw={push_gw} "
          f"steps={steps} n={n_samples} seed={seed} ...", flush=True)

    p1 = cp.PushOutPotential(d_target=T, k=push_k, guidance_weight=push_gw)
    p2 = cp.ContactCountPotential(target=0.0, k=0.05, cutoff=4.5, alpha=8.0,
                                  guidance_weight=3.0)
    sampler = FKCSampler(potentials=[p1, p2], weight_update="guidance",
                         ess_threshold=0.8, num_steps=steps)
    t0 = time.time()
    samples = emu.sample(num_samples=n_samples, batch_size=1, seed=seed, sampler=sampler)
    dt = time.time() - t0
    emu.cache_clear()
    save_frames(samples, outdir, physicality=check_physicality(samples, system=system))

    import mdtraj
    traj = mdtraj.load_xtc(str(outdir / "samples.xtc"), top=str(outdir / "topology.cif"))
    cv1, cv2 = compute_cvs(traj)
    # Flat free-state potential (U=0) => umbrella reweight reduces to raw count.
    U = np.zeros_like(cv1, dtype=float)
    np.savez(outdir / "cv.npz", cv1=cv1.astype(float), cv2=cv2.astype(int), U=U)
    with open(outdir / "cv.csv", "w") as f:
        f.write("frame,cv1_A,cv2_contacts,U_kcalmol\n")
        for i in range(len(cv1)):
            f.write(f"{i},{cv1[i]:.3f},{int(cv2[i])},0.0\n")
    meta = {"d_target": float(T), "n_target": 0, "n_samples": int(cv1.size),
            "cv1_range": [float(cv1.min()), float(cv1.max())],
            "cv2_range": [int(cv2.min()), int(cv2.max())],
            "U_mean": 0.0, "seed": seed,
            "method": "guidance push-out (flat free-state potential, U=0)"}
    (outdir / "window.json").write_text(json.dumps(meta, indent=2))
    print(f"[{time.strftime('%H:%M:%S')}] d={T:.0f}: cv1 {cv1.min():.1f}..{cv1.max():.1f} A "
          f"cv2 {int(cv2.min())}..{int(cv2.max())}  {dt:.0f}s -> {outdir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Guidance push-out free-state sampling")
    ap.add_argument("--targets", required=True,
                    help='space/comma separated push-out targets in A, e.g. "110 130 150 170"')
    ap.add_argument("--n-samples", type=int, default=12)
    ap.add_argument("--steps", type=int, default=400,
                    help="denoising steps per sample (backward pass each step)")
    ap.add_argument("--push-k", type=float, default=20.0,
                    help="push-out stiffness (kcal/mol/A^2)")
    ap.add_argument("--guidance-weight", type=float, default=5.0)
    ap.add_argument("--seeds", nargs='+', type=int, default=[0])
    args = ap.parse_args()
    targets = [float(t) for t in args.targets.replace(',', ' ').split()]

    system = System.from_yaml(str(HERE / "wild_system.yaml"))
    model = load_model(device="cuda", precision="bf16-mixed")
    msa = MSA(unpaired_a3m={"H": f">query\n{SEQ_H}\n",
                             "L": f">query\n{SEQ_L}\n"})
    emu = EmuSession(system, model=model, mode="equilibrium", msa=msa)

    for T in targets:
        for seed in args.seeds:
            run_window(emu, T, args.n_samples, args.steps, seed,
                       args.push_k, args.guidance_weight, OUT_ROOT, model, system)
    emu.cache_clear()
    print("DONE. free-state (guidance) windows:", OUT_ROOT, flush=True)


if __name__ == "__main__":
    main()
