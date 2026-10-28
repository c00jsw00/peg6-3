#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(Path(__file__).resolve().parent); STEERED = HERE / "steered"; KBT = 0.592

def load_windows():
    for pat in ("d*_n*", "l*_k*_n*_s*", "a*_n*_s*"):
        for d in sorted(STEERED.glob(pat)):
            if not (d / "cv.npz").is_file():
                continue
            z = np.load(d / "cv.npz")
            yield z["cv1"].astype(float), z["cv2"].astype(int), z["U"].astype(float)

c1_e = np.arange(8.0, 152.0, 2.0)
c2_e = np.arange(0.0, 260.0, 5.0)
n1, n2 = len(c1_e) - 1, len(c2_e) - 1
prob = np.zeros((n1, n2)); nfr = np.zeros((n1, n2))
for cv1, cv2, U in load_windows():
    u = U - U.max(); w = np.exp(u)
    h, _, _ = np.histogram2d(cv1, cv2, weights=w, bins=(c1_e, c2_e))
    nh, _, _ = np.histogram2d(cv1, cv2, bins=(c1_e, c2_e))
    prob += h / (w.sum() + 1e-12); nfr += nh
prob /= prob.sum()
pmf = -KBT * np.log(prob + 1e-15)
pmf[prob <= 1e-15] = np.nan
med = float(np.nanmedian(pmf))
pmf = np.where(np.isnan(pmf), med, pmf)
g1 = 0.5 * (c1_e[:-1] + c1_e[1:])
g2 = 0.5 * (c2_e[:-1] + c2_e[1:])

# cleft-entrance slice: cv1 35-65, cv2 0-90
c1i = [i for i in range(n1) if 35 <= g1[i] <= 65]
c2i = [j for j in range(n2) if 0 <= g2[j] <= 90]
print("slice rows(cv1):", len(c1i), "cols(cv2):", len(c2i))

fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
sub = pmf[c1i][:, c2i]
im = ax.imshow(sub, origin="lower", aspect="auto",
               extent=[g1[c1i[0]] - 1, g1[c1i[-1]] + 1, 0, 92],
               cmap="viridis", interpolation="nearest", vmin=0, vmax=14)
ax.axvspan(44, 55, color="red", alpha=0.20, lw=2, linestyle="--")
ax.axhspan(15, 45, color="orange", alpha=0.15)
ax.text(49.5, 25, "cleft gate\ncv1 45-51\ncv2 18-45\nPMF 10.7-12.9",
        color="red", fontsize=11, ha="center", va="center", fontweight="bold")
ax.text(50, 4, "approach pocket\ncv2<8\nPMF~1.7", color="white",
        fontsize=10, ha="center", va="center")
ax.set_xlabel("CV1 = ligand-protein COM-COM distance (A)", fontsize=11)
ax.set_ylabel("CV2 = heavy-atom contacts (4.5 A cutoff)", fontsize=11)
ax.set_title("2D PMF: cleft-entrance association barrier", fontsize=13)
cbar = plt.colorbar(im); cbar.set_label("PMF (kcal/mol)")
fig.tight_layout()
out = HERE / "pmf_ladder_final" / "cleft_entrance_barrier.png"
fig.savefig(out, bbox_inches="tight")
print("saved", out)
