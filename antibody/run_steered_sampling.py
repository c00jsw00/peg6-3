#!/usr/bin/env python
"""Steered umbrella-sampling driver for the wild.pdb dissociation study.

System: protein receptor = chains H+L, ligand = chain A.
CV1 = ligand(A) COM - protein(H+L) COM distance (A).
CV2 = heavy-atom contact count, cutoff 4.5 A.

Each window (d_target, n_target) is sampled with FKC guidance (weight_update=
"guidance", reheat="sigma") so the returned particles are equal-weight draws
from the tilted distribution p_tilt ~ p_base*exp(-U). CVs and potential
energies (U) are saved per frame for unbiased reweighting.

Outputs per window (out/<d>_<n>/):
    samples.xtc, topology.cif, cv.csv (frame, cv1, cv2_int, U), window.json
"""
from pathlib import Path
import json
import numpy as np
import torch

from bioemu2 import (
    FKCSampler,
    EmuSession,
    MSA,
    System,
    check_physicality,
    load_model,
    save_frames,
)
from cv_potentials import ComComPotential, ContactCountPotential

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE / "steered"


def torch_from_numpy(a):
    return torch.as_tensor(a)

# protein sequences (for MSA)
SEQ_H = ("QVQLVQSGSELKKPGASVKVSCKASGYTFTNYGMNWVRQAPGQGLEWMGWINTYTGQPIYANDFK"
         "GRFVFSLDTSVSTAYLQISSLKAEDTAVYYCARDWGPYWGQGTLVTVSSASTKGPSVFPLAPSSK"
         "STSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTY"
         "ICNVNHKPSNTKVDKRVEPKSCDK")
SEQ_L = ("DIVMTQSPDSLAVSLGERATINCKSSQSVLYSSNQMNYLAWYQQKPGQPPKLLIYWASTRESGVP"
         "DRFSGSGSGTDFTLTISSLQAEDVAVYYCLQYLSSWTFGGGTKLEIKRTVAAPSVFIFPPSDEQL"
         "KSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHK"
         "VYACEVTHQGLSSPVTKSFNRGEC")


def compute_cvs(traj):
    """Return (cv1_A, cv2_int) arrays of shape (F,) for the trajectory."""
    top = traj.topology
    atoms = list(top.atoms)
    prot = [a.index for a in atoms
            if a.residue.chain.chain_id in ("H", "L") and a.element.symbol != "H"]
    lig = [a.index for a in atoms
           if a.residue.chain.chain_id == "A" and a.element.symbol != "H"]
    xyz = np.asarray(traj.xyz, dtype=np.float64)  # (F,N,3) nm

    # CV1
    pc = xyz[:, prot].mean(axis=1)
    lc = xyz[:, lig].mean(axis=1)
    cv1 = 10.0 * np.linalg.norm(lc - pc, axis=1)  # A

    # CV2: per-frame heavy-atom PAIR contact count, cutoff 0.45 nm (4.5 A).
    # Count all (ligand-heavy, protein-heavy) pairs within cutoff.
    L = xyz[:, lig].copy()
    P = xyz[:, prot].copy()
    d = np.sqrt(np.sum((L[:, :, None, :] - P[:, None, :, :]) ** 2, axis=-1))  # (F,Na,Np)
    cv2 = np.sum(d <= 0.45, axis=(1, 2)).astype(np.int64)  # (F,)
    return cv1, cv2


def potential_energy(traj, system, p1, p2):
    """Re-evaluate the steering potentials on final coords -> U per frame."""
    xyz = np.asarray(traj.xyz, dtype=np.float32) * 10.0  # -> A
    coords = torch_from_numpy(xyz).to("cuda", non_blocking=True)
    p1.prepare(system)
    p2.prepare(system)
    U1 = p1(coords, t=0.0).cpu().numpy()
    U2 = p2(coords, t=0.0).cpu().numpy()
    return U1, U2


def run_window(d_target, n_target, out_root, n_samples, batch, seed, model,
               system, use_msa):
    out = out_root / f"d{int(round(d_target))}_n{int(round(n_target))}"
    out.mkdir(parents=True, exist_ok=True)

    # Soft harmonic wells: nudge toward the window target without forcing
    # non-physical conformations. Weaker = more faithful to p_base (needed
    # for clean reweighting); the tilted distribution is then corrected.
    p1 = ComComPotential(target=float(d_target), k=0.5, guidance_weight=1.0)
    p2 = ContactCountPotential(target=float(n_target), k=0.05, cutoff=4.5,
                               alpha=8.0, guidance_weight=1.0)

    xtc = out / "samples.xtc"
    if xtc.exists():
        print(f"skip sampling {out} (samples.xtc exists), reusing")
        samples = None
        emu = None
        # FKC SMC sampler still constructed for potential-energy re-eval
        sampler = None
    else:
        # FKC SMC: resampling-only (no gradient / no denoiser backward pass).
        # Targets the tilted distribution p_tilt ~ p_base * exp(-U) and returns
        # equal-weight samples (resampled to equal weight at the final step),
        # which is exactly what unbiased reweighting needs.
        sampler = FKCSampler(
            potentials=[p1, p2],
            weight_update="smc",
            reheat="gamma",
            ess_threshold=0.8,
            num_steps=200,
        )
        if use_msa:
            msa = MSA(unpaired_a3m={"H": f">query\n{SEQ_H}\n",
                                    "L": f">query\n{SEQ_L}\n"})
            emu = EmuSession(system, model=model, mode="equilibrium", msa=msa)
        else:
            emu = EmuSession(system, model=model, mode="equilibrium")

        samples = emu.sample(num_samples=n_samples, batch_size=batch,
                             seed=seed, sampler=sampler)
        save_frames(samples, out, physicality=check_physicality(samples, system=system))
        emu.cache_clear()
        print(f"sampled {out}")

    traj = __import__("mdtraj").load_xtc(str(out / "samples.xtc"),
                                         top=str(out / "topology.cif"))
    cv1, cv2 = compute_cvs(traj)
    U1, U2 = potential_energy(traj, system, p1, p2)
    U = U1 + U2

    cv = np.column_stack([cv1, cv2, U1, U2, U])
    np.savez(out / "cv.npz", cv1=cv1, cv2=cv2, U1=U1, U2=U2, U=U)
    with open(out / "cv.csv", "w") as f:
        f.write("frame,cv1_A,cv2_contacts,U1_kcalmol,U2_kcalmol,U_kcalmol\n")
        for i in range(len(cv1)):
            f.write(f"{i},{cv1[i]:.3f},{int(cv2[i])},{U1[i]:.4f},{U2[i]:.4f},"
                    f"{U[i]:.4f}\n")
    meta = {
        "d_target": float(d_target), "n_target": int(n_target),
        "n_samples": int(cv1.size), "cv1_range": [float(cv1.min()), float(cv1.max())],
        "cv2_range": [int(cv2.min()), int(cv2.max())],
        "U_mean": float(U.mean()), "seed": seed,
    }
    (out / "window.json").write_text(json.dumps(meta, indent=2))
    print("window d=%.1f n=%d: cv1 range %.1f..%.1f  cv2 %d..%d  U=%.2f kcal/mol -> %s" %
          (d_target, n_target, cv1.min(), cv1.max(),
           cv2.min(), cv2.max(), U.mean(), out))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=str, default=None,
                    help="comma-separated 'd,n' pairs, e.g. '35,260 95,0'")
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-msa", action="store_true", default=True)
    ap.add_argument("--dry", action="store_true",
                    help="print plan only")
    args = ap.parse_args()

    sys_obj = System.from_yaml(str(HERE / "wild_system.yaml"))
    print("system OK:", [(type(e).__name__, getattr(e, "asym_ids", None))
                         for e in sys_obj.entities])

    windows = []
    if args.windows:
        for w in args.windows.split():
            d, n = w.split(",")
            windows.append((float(d), float(n)))
    else:
        # default 2D umbrella grid (bound -> dissociated)
        for d in (35.0, 55.0, 75.0, 95.0):
            for n in (260.0, 130.0, 0.0):
                windows.append((d, n))

    if args.dry:
        print("plan:", windows, "n_samples", args.n_samples)
        return

    model = load_model(device="cuda", precision="bf16-mixed")
    for d, n in windows:
        run_window(d, n, OUT_ROOT, args.n_samples, args.batch, args.seed,
                   model, sys_obj, args.use_msa)
    print("DONE", windows)


if __name__ == "__main__":
    main()
