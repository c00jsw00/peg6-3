#!/usr/bin/env python3
"""Publication-quality PMF plots from the final reweighting."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = np.load(HERE/"pmf_ladder_final/pmf_ladder.npz")
pmf = data["pmf"]; rmin = data["rmin_A"]; prob = data["prob"]
KBT = 0.592
Pb = float(prob[rmin < 4.0].sum())
Pf = float(prob[rmin > 15.0].sum())
dg = -KBT*np.log(Pb/Pf) if Pf > 1e-12 else float("nan")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

ax = axes[0]
ax.plot(rmin, pmf, lw=1.6, color="#1f77b4", label=f"PMF F(r)\n(ΔG₀=0)")
ax.set_xlabel("Ligand–receptor distance, r$_{min}$ (Å)")
ax.set_ylabel("Free energy F(r) (kcal/mol)")
ax.set_title("Protein–glycan dissociation PMF")
ax.axvspan(0, 4, alpha=0.12, color="green", label="Bound (r<4 Å)")
ax.axvspan(4, 15, alpha=0.12, color="orange", label="Transition (4–15 Å)")
ax.axvspan(15, rmin.max(), alpha=0.12, color="lightblue", label="Free (r>15 Å)")
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(alpha=0.25)

ax = axes[1]
ax.plot(rmin, prob*100, lw=1.6, color="#1f77b4", label="P(r)$_{min}$")
ax.set_xlabel("Ligand–receptor distance, r$_{min}$ (Å)")
ax.set_ylabel("Probability density (% of frames)")
ax.set_title("Unbiased distribution P(r$_{min}$)")
ax.axvline(4, ls="--", c="green", lw=1)
ax.axvline(15, ls="--", c="lightblue", lw=1)
ax.legend(fontsize=8.5)
ax.grid(alpha=0.25)

ax = axes[2]
bars = ax.bar(["Bound\n(r<4 Å)", "Transition\n(4–15 Å)", "Free\n(r>15 Å)",
               "ΔG$_{bind}$"],
              [Pb*100, float(prob[(rmin>=4)&(rmin<15)].sum())*100, Pf*100, dg],
              color=["#2ca02c","#ff7f0e","#8ecae6","#d62728"], alpha=0.9)
ax.set_ylabel("Value (%) / (kcal/mol)")
ax.set_title("State populations & ΔG$_{bind}$")
ax.grid(alpha=0.25, axis="y")
for i, v in enumerate(bars):
    val = [Pb*100, float(prob[(rmin>=4)&(rmin<15)].sum())*100, Pf*100, dg][i]
    ax.text(v.get_x()+v.get_width()/2, v.get_height(), f"{val:.1f}",
            ha="center", va="bottom", fontsize=9)
ax.set_ylim(0, max(Pb*100, Pf*100, abs(dg))*1.4)

fig.suptitle("BioEmu2 umbrella-sampling PMF — 9FO glycan on receptor (49 windows, 1194 frames)",
             fontsize=12.5, y=1.02)
fig.tight_layout(rect=[0,0,1,0.975])
out = HERE/"pmf_ladder_final"
fig.savefig(out/"pmf_final.png", dpi=150, bbox_inches="tight")
fig.savefig(out/"pmf_final.svg", bbox_inches="tight")
print("WROTE", out/"pmf_final.png", out/"pmf_final.svg")
