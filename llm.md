# LLM Runbook — BioEmu2 蛋白–配體 k_on / k_off / K_D 速率研究

This file tells an LLM (me / you) exactly how to run the protein–ligand rate
study (**k_on, k_off, K_D**) from BioEmu2 steered sampling on **any complex PDB**.
Reference run (9FO glycan) + all code/data/results: [`antibody/`](antibody/),
with the full command recipe and caveats in
[`antibody/RATES_STUDY.md`](antibody/RATES_STUDY.md).
The broader method + this end-to-end example also live in the Hermes
`bioemu2` skill.

## 何時使用 / When to use

When the user gives you a **new PDB** (receptor + ligand) and asks for
k_on / k_off / K_D.

## 需要使用者提供 / What you need from the user

1. 新 PDB 檔案路徑。
2. (若不明顯) 哪些 chain 是 **受體 receptor**（protein）、哪個 chain 是 **配體 ligand**。
3. 實驗參考值（可選但建議）：k_on、k_off、K_D — 用於比較。

## Repo 佈局（rate-study 程式碼在 `antibody/`）

- **drivers**: `run_steered_sampling.py`, `run_guidance.py`,
  `run_transition_windows.py`, `run_rmin_ladder.py`, `run_rmin_ladder_push.py`,
  `run_ladder_final.py`, `run_assoc_umbrella.py`, `run_umbrella_ladder.py`,
  `run_ts_refine.py`
- **analyzers**: `analyze_kon.py`, `analyze_kon_final.py`,
  `make_final_plots.py`（+ ladder analyzers）
- **potentials**: `cv_potentials.py`
- **results**: `antibody/pmf_ladder_final/`（barrier JSON、PMF、plots）
  + `antibody/steered/`（raw XTC trajectories）

## Step 0 — 把 chain ID 改成新 PDB 的 ⚠️ 最關鍵的一步

Current code **hardcodes** 受體 chains **H, L** 和 配體 **A** across ~18 files.
Before running, remap them.

**Detect the new chain IDs**（用 biopython）：

```bash
python - <<'PY'
from Bio.PDB import PDBParser
p = PDBParser(SILENT=1)
st = p.get_structure('x', 'complex.pdb')   # <- 改成新 PDB 路徑
print([ch.id for ch in st.models[0].chains])
PY
```

Literals appear as `in ("H", "L")`（受體集合）和 `== "A"`（配體）. Run this,
setting the four values to match the **new** PDB:

```python
# 從 repo root 執行（cd 到 peg6-3 目錄，使 antibody/ 相對可達）
import re, pathlib
base = pathlib.Path.cwd() / "antibody"
if not base.exists():
    raise SystemExit(f"antibody/ not found under {base.parent} — run from repo root")
files = ["cv_potentials.py","run_steered_sampling.py","run_transition_windows.py",
         "run_rmin_ladder.py","run_rmin_ladder_push.py","run_ladder_final.py",
         "run_guidance.py","run_assoc_umbrella.py","run_umbrella_ladder.py",
         "run_ts_refine.py","analyze_ladder.py","analyze_ladder_final.py",
         "analyze_all_windows.py","_finalize_rates.py"]
old_rec, new_rec = ["H","L"], ["R1","R2"]   # new_rec = 新受體 chain ids（可 1~N）
old_lig, new_lig = "A", "L"                  # new_lig  = 新配體 chain id
def remap(t):
    orl = ", ".join(f'"{c}"' for c in old_rec)
    nrl = ", ".join(f'"{c}"' for c in new_rec)
    t = re.sub(r'in\s*\('+re.escape(orl)+r'\)', f'in ({nrl})', t)
    for q in ("'", '"'):
        t = re.sub(rf'chain_id\s*==\s*{q}{re.escape(old_lig)}{q}',
                   f'chain_id == {q}{new_lig}{q}', t)
    return t
for fn in files:
    p = base / fn
    if not p.exists(): continue
    nt = remap(p.read_text())
    if nt != p.read_text():
        p.write_text(nt); print(f"{fn} -> rewritten")
    else:
        print(f"{fn} -> unchanged")
print("done")
```

> **Note:** if you run this from the repo root, `Path.cwd()/antibody` works.
> If you run it from elsewhere, set `base` to the absolute `antibody/` path
> first (e.g. `base = pathlib.Path("/abs/path/peg6-3/antibody")`).

**Special case — `run_steered_sampling.py`**（只在 `--use-msa` 時需要）：

- 把兩個受體 **序列** 字串 blocks `SEQ_H = (...)`、`SEQ_L = (...)`
  改成新受體 chain 的序列。
- 把 MSA dict keys `{"H": ..., "L": ...}` 改成新受體 chain ids。
- 若不想處理、或不需要 MSA：用 `--use-msa`（預設 True）但保持正確序列，
  或跳過（結構預測品質可能下降）。

**Verify before proceeding** — confirm no stray old literals remain:

```bash
grep -rE 'in \("H", "L"\)|chain_id (==|in ) ["\']A["\']' antibody/
# 上面應回空（無輸出）。再目視確認每個 in(...) / == "..." 都已是新 ids。
```

> **Caution:** `sym(a) != "H"` / `element.symbol != "H"` are **element-symbol**
> filters (exclude hydrogen atoms) — **NOT** chain references. Do **not**
> change those. Only touch the `chain_id in (...)` / `chain_id == "..."` literals.

## Step 1 — 環境

```bash
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install ".[cuda]"          # compute >= 7.5（RTX 4090 等）
python -m pip install mdtraj numpy matplotlib
cd antibody/
```

## Step 2 — 跑 6 步 pipeline（每步 idempotent，可重啟）

```bash
# 1) SMC 2D umbrella: bound -> dissociated
python run_steered_sampling.py --windows "35,260 55,180 75,60 95,0" \
    --n-samples 30 --batch 10 --seed 42

# 2) Free state (push-out guidance, 存 flat U=0)
python run_guidance.py --targets "90 110 130 150" --n-samples 12 --seeds 0 1

# 3) Transition-region r_min gap (4-15 A)
python run_transition_windows.py --targets "80 85" --seeds 0 1 2

# 4) Cleft-entrance association barrier (SMC)
python run_assoc_umbrella.py --n-samples 25 --batch 5 --seeds 0 1 2

# 5) Cleft-exit off-barrier saddle —— 一定要用 stiff-well guidance ladder
python run_umbrella_ladder.py --targets "46 40 52 25 56 15" --k 60 --gw 6 \
    --n-samples 25 --steps 1500 --seeds 0 1 2

# 6) Analyze + rates + plots
python analyze_kon.py
python analyze_kon_final.py
python make_final_plots.py
```

Step 5 用 **stiff-well** guidance ladder（k=60, batch=1, 存 real per-frame U）。
soft SMC（k~0.5, batch 5）無法把配體移離 bound basin，**做不到** off-barrier
saddle — step 5 不要用它。
長時間的 step 用 background 跑並 poll log；單顆 RTX 4090 約每 window ~1 h。

## Step 3 — Rate recipe（Eyring，一級，per-molecule）

```
kBT        = 0.592 kcal/mol
kBT_over_h = 6.21e12 s^-1      # 分子單位的 kBT/h —— NOT KBT/H（差 152×）
k_off      = 6.21e12 * exp(-dG_off‡/kBT)
           dG_off‡ = PMF(TS) - PMF(bound)   在 high-CV2(cleft) 出口 path
k_on       = 6.21e12 * exp(-dG_on‡/kBT)     # TST 自 fine-grid cleft-entrance PMF
           dG_on‡  = PMF(gate) - PMF(approach or free)
K_D        = k_off / k_on
```

Estimator：**per-window reweighting**
`B_i(b) = Σ_j exp(U_i,j − U_i,max) / Σ_j exp(U_i,j)`，再對 windows 取平均。
**絕不能用** 單個 global `exp(U − max)`（dissociated window 會爆掉）。
CVs：CV1 = COM-COM（ligand vs receptor, Å）；CV2 = heavy-atom PAIR contact count @4.5 Å；
r_min = min ligand→receptor heavy-atom distance（measurement CV）。

## Step 4 — 報告結果（與實驗比較表）

讀 `antibody/pmf_ladder_final/association_barrier.json`（`off_barrier` 段）
+ `steered/postproc/barriers_final.json`，報告：

- dG_on‡（gate−approach）、k_on
- dG_off‡（lower bound）、k_off（upper bound）
- K_D（detailed balance）

以表格形式與實驗比較（同 README.md / RATES_STUDY.md 的表）。
注意 floor 限制：在 1e-15 PMF floor 下 barrier 上限 ~20.45 kcal/mol；
若實驗 off-barrier 高於它，k_off / dG_off‡ 是 **bounds**，不是精確值。

## Gotchas / 注意事項

- **Chain IDs 是最大失敗點** — 務必 Step 0 完成並驗證。
- k_off 受 floor 限制（upper bound）；**K_D = k_off/k_on** 才是 robust 量。
- Cleft-entrance（association）barrier 已 resolve 且 robust；
  cleft-exit saddle 受 floor 限制。
- 要精確 k_off：降 PMF floor（1e-20 → cap ~27 kcal/mol）+ 更多 saddle frames，
  或用 metadynamics。
