# REPRODUCE.md

One row per data figure and table of the paper (manuscript of 30 August
2026): the command that regenerates it, the parameter files and records it
reads, the printed value, and the value regenerated in this repository on
14 October 2026. No retraining anywhere: every row reads committed
`results/` files. Paths under "reads" are relative to `results/campaigns/`
unless stated otherwise; `docs/PROVENANCE.md` has the full record map.

Setup once (Python 3.12):

```bash
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
```

The whole set:

```bash
python3 scripts/render_paper_figures.py             # Figs. 3-12 -> results/figures/
python3 scripts/print_paper_tables.py --recompute     # Tables I-VI, checked against the printed values
cd notebooks && jupyter nbconvert --to notebook --execute --inplace central_comparison.ipynb && cd ..
python3 -m pytest tests/ -q                           # 432 tests, ~17 min
```

Test suite: 432 passed in the pinned environment of `requirements.txt`,
verified both in the working environment and in a fresh `python3 -m venv`
(17 min each).

---

## Figures

Figure files carry the manuscript-revision tag they were last re-rendered in
(`_w5` … `_w8`); `scripts/render_paper_figures.py` maps the paper's figure
numbers to them. "Regenerated" states the caption facts echoed by the render
function and the pixel comparison against the PNGs in the manuscript.

| Fig. | content | file (results/figures/) | reads | expected (paper) | regenerated | ✔ |
|---|---|---|---|---|---|---|
| 3 | register size, $((n,3,3))_3$, both benchmarks | `fig_n_w6.png` | `r2_campaign/t4_results.json`, `fig6_panels/measurements.json`, `r11_gapfill/t1a_family_L2.json` | channel rate rises monotonically from n = 5 at every s, both recovery maps | monotone booleans True at all five s; same data, minor tick-label rendering differs (6 % of pixels) | ✅ |
| 4 | qudit dimension at K = 3, n = 5 | `fig_q_w5.png` | `fig6_panels/measurements.json`, `r5_campaign/t23_results.json`, `r9_campaign/t2d_results.json` | Petz: improves with q; projective: worsens with q; $((5,3,3))_5$ bounded band | codes q4_L2_s100 / q5_L2_s100 / central; same data, minor tick-label rendering differs | ✅ |
| 5 | distance trio, repeated rounds | `fig_dist_w8.png` | `r11_gapfill/t4_analysis.json` | per-round infidelity d4 < d3 < d2 at every s | d2/d4 per-round margins 1.0729, 1.035, 1.0079, 1.0029, 1.0014 (s = 0.125 … 1.5); minor rendering differences | ✅ |
| 6 | depth study $L \in \{2,4,8,16\}$ | `fig_depth_w6.png` | `r9_campaign/t5_results.json`, `t5_loss_traces.npz`, `r11_gapfill/t5_sgrid_results.json` | loss falls with depth; i.i.d. LER never improves; channel improves weakly | rendered; pixel-identical to the paper PNG | ✅ |
| 7 | magnitude pruning 40 → 16/12/8 MS | `fig_prune_w6.png` | same as Fig. 6 | 16 ≥ unpruned; 12 close; 8 breaks (×3–4) | rendered; pixel-identical | ✅ |
| 8 | encoder cost by entangling gate | `fig_gatecost_w7.png` | analytic ansatz counts | 60 two-qudit gates (MS/ZZ) vs 30 (LS/CSUM/CEX) at $((6,3,3))_3$ | rendered; pixel-identical | ✅ |
| 9 | entangling-gate study $((6,3,3))_3$ | `fig_gates_w7.png` | `r5_campaign/t1_results.json` + `t1_codes/*.npz`, `r4_campaign/a1_results.json`, `r6_campaign/esc_psweep.json`, `r12_scrutiny/t5_results.json` | weight-2-perfect runs 12/12, 11/12, 9/12, 9/12, 0/12 (MS, ZZ, LS, CSUM, CEX) | tallies 12/11/9/9/0; CEX/MS Petz ratios 4.11 → 1.13 across s; pixel-identical | ✅ |
| 10 | sampler cost–quality frontier $((9,3,3))_3$ | `fig_sampler_frontier_w8.png` | `r10_sampler/t3_analysis.json` | certified budgets 6 435–34 250 op-EVs, quality flat | budgets span 6435–34250; pixel-identical | ✅ |
| 11 | convergence traces behind Fig. 10 | `fig_sampler_traces_w8.png` | `r10_sampler/t1_codes/*.npz`, `t3_analysis.json` | stops at steps 29–183, confirmed residuals 9.0–9.9 × 10⁻⁵ | 20 runs, stops 29–183, residuals 9.01e-5–9.87e-5; pixel-identical | ✅ |
| 12 | paired two-benchmark comparison at n = 5 | `fig_headline_pairs_w8.png` | `r11_gapfill/t2_table1_results.json` | $[[5,1,3]]_{Z_3}$ lowest at every matched point; central leads i.i.d. for p ≤ 0.08 | rendered from the same cells; minor rendering differences (label wrapping) | ✅ |

Figs. 1 and 2 are schematics (TikZ/quantikz in the manuscript, not data).

## Tables

| Table | content | command | reads | expected (paper) | regenerated | ✔ |
|---|---|---|---|---|---|---|
| I | calibrated comparison, s ∈ {0.125, 0.25, 0.5, 1, 1.5}, projective / Petz, both codes | `scripts/print_paper_tables.py I --recompute`; notebook Sec. 5 | `r1_campaign/t1_codes/A_L2_s100.npz`, Eq. (13); record `r9_campaign/t1_results.json` | 5.66/0.70/4.91/0.23/3.05; 11.78/2.05/10.41/0.84/2.45; 23.17/6.03/21.05/3.08/1.96; 39.62/16.22/37.16/10.43/1.56; 49.36/26.32/47.18/19.42/1.36 | all 25 cells and 5 ratios identical to the printed digits (recomputed from the parameters) | ✅ |
| I caption | σ_a² = 0 at s = 1 | notebook Sec. 5 | same | Petz 11.8 / 5.9; cost 37 % / 76 % | 11.8 / 5.9; 37 % / 77 % (76.8 % unrounded; see HANDOFF) | ✅ |
| II | Shor–Laflamme enumerators, 12 rows | `scripts/print_paper_tables.py II --recompute` | parameter files listed in the script; records `table3/table3_enumerators.json`, `w5_t0/w5_enum_imp005.json` | e.g. central 2.4408 / 2.4229 / 2.4631 / 3.1225; stabilizer 2.2222 / 1.9753 with zero deficit; sampler row 4.0050 / 7.1371 / 4.0064 / 7.1883 | all 48 entries identical to 4 decimals (recomputed from the parameters) | ✅ |
| III | encoder resources, 7 rows | `scripts/print_paper_tables.py III` | analytic (`src/central_comparison.py::ansatz_resource_counts`) | e.g. $((5,3,3))_3$: 10 / 90 / 180 / 20 / 20 / 10 / 40 / pruned 12 | all 7 rows identical | ✅ |
| IV | weight-conditioned decomposition, 6 rows | `scripts/print_paper_tables.py IV --recompute`; notebook Secs. 2 and 6 | `r1_campaign/t1_results.json`, `loss_ler_diag/t1c,t1e,t1b_results.json`, `headline_p05_Lsweep/lsweep.json`, `r9_campaign/t1,t4_results.json` | residuals 4.6 / 2.90 / 0.449 / 0.360 / 1.95e-4 / ~1e-30; LER\|w=3 0.289 … 0.729; i.i.d. 3.42e-4 … 8.49e-4; Petz s=0.5 6.03e-2 … 3.08e-2 | residuals and Petz column recomputed, identical; LER cells identical to the records; stabilizer LER\|w=3 = 0.7287 from the notebook probe (not in any record) | ✅ |
| V | stopping rules $((6,3,3))_3$, 4 rules × 4 runs | `scripts/print_paper_tables.py V` | `sampling_budget_v2/b_results.json` (arm W) | stop 3000 / 1107–1159 / 1133–1197 / 2200–2400; confirms 1 (0) / 53–69 (52–68) / 2–3 (1–2) / 0; totals 165 354 / 81 368 / 65 493 / 110 333 | identical | ✅ |
| VI | realized per-step cost, n = 5, 6, 7, 9 | `scripts/print_paper_tables.py VI` | `r4_campaign/a2_cost_table.json` | 221/313/421/685, 1 027 500, 1×; 38/55/71/114, 170 736, 6.0×; 58/81/104/193, 289 290, 3.6×; 27/37/49/74, 111 706, 9.2× | identical | ✅ |

## Text numbers of the central comparison (Sec. VI A, App. C, App. F)

Reproduced by `notebooks/central_comparison.ipynb` (wall time 5.6 min on an
8-core laptop under load; the committed copy carries the outputs):

| quantity | paper | notebook |
|---|---|---|
| i.i.d. LER at p = 0.05, 2 × 10⁶ shots, central / stabilizer | 3.42 × 10⁻⁴ / 8.49 × 10⁻⁴, ratio 2.48 | 3.420e-4 (684 errors) / 8.490e-4 (1698 errors), 2.48 — identical shot stream |
| LER\|w=1, LER\|w=2 (both codes) | 0 | 0 / 200 000 and 0 / 100 000 |
| LER\|w=3 central / stabilizer | 0.289 / 0.729 | 0.2892 / 0.7287 |
| [[5,1,3]] encoder, MS count; codeword residual | 36; ≤ 1e-14 | 36 (+174 single-qudit gates); 9.5e-15 |
| CZ = 4 MS exactness | exact | 3.1e-15 (Frobenius) |
| coherence multipliers Δf = 0, 1, 2 at η = 0.9296 | 1.0 / 0.9642 / 0.8642 | 1.0 / 0.9642 / 0.8642 |
| projective-decoder sectors at s = 1: central; stabilizer | 81 (all 3-dim); 74 / 7 / 7 | 81 {3: 81}; {3: 74, 2: 7, 1: 7}; 92 sectors at s = 0.5 |
| Petz-over-optimal overhead at s = 0.5, stabilizer / central | 1.46 / 1.34 (range 1.42–1.46 / 1.28–1.36) | 1.456 / 1.339; full grid 1.39–1.47 / 1.29–1.35 |
| optimal recovery, stabilizer leads | every cell | certified ratio 3.28, 2.65, 2.13, 1.70, 1.47 at s = 0.125 … 1.5 |
| detection residual of the central code (Table IV) | 4.6 | 4.5994 (full basis); 0.9888 on the completed channel basis |

## Out of scope

Figs. 1–2 (schematics) and the gate-definition equations are typeset, not
computed.

## Test suite

432 passed, 0 skipped, in both the working environment and a fresh virtual
environment built from `requirements.txt` (17 min each).
