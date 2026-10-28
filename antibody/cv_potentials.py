#!/usr/bin/env python
"""Steering potentials for the wild.pdb dissociation study.

CV1 = ligand(A) COM - protein(H+L) COM distance  (angstroms, continuous)
CV2 = ligand-heavy/protein-heavy contact count, cutoff 4.5 A (discrete ->
      smooth differentiable proxy for steering, exact int saved in post-proc).

Both are FeatsPotential subclasses with a harmonic well:
    U = 0.5*k1*(CV1 - T1)^2 + 0.5*k2*(CV2 - T2)^2
so the tilted distribution is p_tilt ~ p_base * exp(-beta*U), sampled by
FKC guidance (equal-weight tilted samples) for unbiased reweighting.
"""
import torch

from bioemu2.denoising.steering.potentials import FeatsPotential, Potential


class ComComPotential(FeatsPotential):
    """Harmonic well on ligand(A) vs protein(H+L) COM-COM distance (A).

    Frame-invariant: COM-COM is unchanged by rigid motion, no Kabsch needed.
    """

    def __init__(self, target: float, k: float, guidance_weight: float = 3.0):
        super().__init__(parameters={"guidance_weight": guidance_weight,
                                     "resampling_weight": 1.0})
        self.target = target  # A
        self.k = k            # kcal/mol/A^2
        self._prot_idx = None
        self._lig_idx = None

    def prepare(self, system) -> None:
        super().prepare(system)
        atoms = list(system.topology.atoms)
        def sym(a):
            return a.element.symbol

        self._prot_idx = torch.tensor(
            [i for i, a in enumerate(atoms)
             if a.residue.chain.chain_id in ("H", "L") and sym(a) != "H"],
            dtype=torch.int64)
        self._lig_idx = torch.tensor(
            [i for i, a in enumerate(atoms)
             if a.residue.chain.chain_id == "A"],
            dtype=torch.int64)

    def compute(self, coords, feats, parameters=None):
        # coords: [B, N, 3] in A
        P = coords[:, self._prot_idx, :]
        L = coords[:, self._lig_idx, :]
        pc = P.mean(dim=1)
        lc = L.mean(dim=1)
        d = torch.linalg.norm(lc - pc, dim=-1)
        return 0.5 * self.k * (d - self.target) ** 2


class ContactCountPotential(FeatsPotential):
    """Smooth differentiable proxy for the heavy-atom PAIR contact count.

    CV2_soft = sum_{a in ligand_heavy} sum_{b in protein_heavy}
                  sigmoid(alpha * (cutoff - d_{a,b}))
    This is a smooth differentiable proxy for the spec metric:
    number of (protein-heavy, ligand-heavy) pairs within the 4.5 A cutoff.
    It is the soft counterpart of the exact integer pair count saved in
    post-processing.
    """

    def __init__(self, target: float, k: float, cutoff: float = 4.5,
                 alpha: float = 8.0, guidance_weight: float = 3.0):
        super().__init__(parameters={"guidance_weight": guidance_weight,
                                     "resampling_weight": 1.0})
        self.target = target  # contact units
        self.k = k            # kcal/mol / contact^2
        self.cutoff = cutoff
        self.alpha = alpha
        self._lig_idx = None
        self._prot_idx = None

    def prepare(self, system) -> None:
        super().prepare(system)
        atoms = list(system.topology.atoms)
        def heavy(a):
            return a.element.symbol != "H"
        self._prot_idx = torch.tensor(
            [i for i, a in enumerate(atoms)
             if a.residue.chain.chain_id in ("H", "L") and heavy(a)],
            dtype=torch.int64)
        self._lig_idx = torch.tensor(
            [i for i, a in enumerate(atoms)
             if a.residue.chain.chain_id == "A" and heavy(a)],
            dtype=torch.int64)

    def compute(self, coords, feats, parameters=None):
        L = coords[:, self._lig_idx, :]        # [B, Na, 3]
        Pr = coords[:, self._prot_idx, :]      # [B, Np, 3]
        # pairwise distances ligand-atom x protein-atom:
        # L[:, None, :, :] -> [B,1,Na,3] ; Pr[:, :, None, :] -> [B,Np,1,3]
        d = torch.norm(L[:, None, :, :] - Pr[:, :, None, :], p=2, dim=-1)  # [B,Np,Na]
        # soft indicator of a pair being in contact (<= cutoff)
        indicator = torch.sigmoid(self.alpha * (self.cutoff - d))            # [B,Np,Na]
        soft_count = indicator.sum(dim=(1, 2))                                # [B]
        return 0.5 * self.k * (soft_count - self.target) ** 2


class PushOutPotential(FeatsPotential):
    """One-sided COM-COM repulsion that pushes the ligand away from the protein.

    Frame invariant (COM-COM is unchanged under any rigid motion, no Kabsch).
        U = 0.5*k*soft_max(0, d_target - d)^2   where d = |C_lig - C_prot| (A)
    U == 0 whenever d >= d_target; grows quadratically as the ligand sits closer,
    producing a gradient that drags it out. Follows templates/push_out_potential.py.
    """

    def __init__(self, d_target: float, k: float = 5.0,
                 guidance_weight: float = 5.0):
        super().__init__(parameters={"guidance_weight": guidance_weight,
                                     "resampling_weight": 1.0})
        self.d_target = d_target   # A
        self.k = k
        self._prot_idx = None
        self._lig_idx = None

    def prepare(self, system) -> None:
        super().prepare(system)
        atoms = list(system.topology.atoms)
        def sym(a):
            return a.element.symbol
        self._prot_idx = torch.tensor(
            [i for i, a in enumerate(atoms)
             if a.residue.chain.chain_id in ("H", "L") and sym(a) != "H"],
            dtype=torch.int64)
        self._lig_idx = torch.tensor(
            [i for i, a in enumerate(atoms)
             if a.residue.chain.chain_id == "A" and sym(a) != "H"],
            dtype=torch.int64)

    def compute(self, coords, feats, parameters=None):
        # coords: [B, N, 3] in A
        P = coords[:, self._prot_idx, :]
        L = coords[:, self._lig_idx, :]
        c_prot = P.mean(dim=1)
        c_lig = L.mean(dim=1)
        d = torch.linalg.norm(c_lig - c_prot, dim=-1)  # [B] in A
        diff = torch.clamp(self.d_target - d, min=0.0)
        return 0.5 * self.k * (diff * diff)


def window_potentials(t1: float, t2: float,
                      k1: float, k2: float, cutoff: float = 4.5,
                      alpha: float = 10.0):
    """Return [ComComPotential, ContactCountPotential] for window (t1, t2)."""
    return [ComComPotential(target=t1, k=k1),
            ContactCountPotential(target=t2, k=k2, cutoff=cutoff, alpha=alpha)]
