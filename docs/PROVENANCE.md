# Provenance: which committed data backs which figure and table

All reported numbers come from committed parameter files (`.npz`), result
records (`.json`) and benchmark CSVs under `results/`. Nothing in this
repository retrains a code to reproduce a figure. The campaign records
live under `results/campaigns/<campaign>/`; each campaign directory also
keeps the script that produced its records (`*_queue.py`, `t*_*.py`), so
every number can be traced to the code that computed it. Paths below are
relative to `results/campaigns/` unless stated otherwise.

## Codes

| code (paper name) | parameter file | protocol |
|---|---|---|
| central $((5,3,3))_3$, MS 40 | `r1_campaign/t1_codes/A_L2_s100.npz` (catalog `central_553_3`) | channel-weighted, importance sampling f = 0.2, L = 2, all-to-all; selected among `A_L2_s100..103` by the weight-2/weight-3 probes (`r1_campaign/t1_results.json: headline`) |
| $[[5,1,3]]_{Z_3}$ | analytical, `src/catalog.py::five_qudit_code_states` (Eq. 13) | its 36-MS encoder: `src/stabilizer_encoder.py` (record `r2_campaign/t6a_circuit.json`) |
| $((n,3,3))_3$, n = 6, 8, 9 | `r2_campaign/t4_codes/n{6,8,9}_L2_s*.npz` | channel-weighted, L = 2 |
| $((7,3,d))_3$, d = 2, 3 | `r2_campaign/t1a_codes/d{2,3}_L2_s100.npz` | channel-weighted, L = 2 |
| $((7,3,4))_3$ | `r3_campaign/t2c_codes/d4_L2_s102.npz` | channel-weighted, L = 2 |
| $((5,3,3))_4$, $((5,3,3))_5$ | `r3_campaign/t2b_codes/q{4,5}_L2_s100.npz` | channel-weighted, L = 2 |
| depth study $L \in \{2,4,8,16\}$ | `r1_campaign/t1_codes/A_L{2,4,8,16}_s{100..103}.npz` | channel-weighted |
| pruned encoders (16/12/8 MS) | `r1_campaign/t2_codes/headline_prune{10,25,50}.npz` | magnitude pruning of the central code |
| unweighted $L = 16$ reference | `campaign/d3_n5_dist3_16L.npz` | unweighted full basis, full batch |
| gate study $((6,3,3))_3$ | `r5_campaign/t1_codes/{ms,zz,ls,csum,cex}_s{100..111}.npz` | 12 runs per entangler |
| sampler study $((9,3,3))_3$ | `r10_sampler/t1_codes/*.npz` | certified stop, 4 runs per strategy |
| n = 9 reference (`n9_reference/`) | `d3_n9_dist3_4L_meth_physical_seed1.npz`, `meth_pauli_weights.npz` | an earlier-draft ((9,3,3))₃ code and its 685-operator channel prior, kept only because `tests/test_petz.py`, `tests/test_weighted_map.py` and `tests/test_benchmark_ler_meth.py` exercise the shot-based Petz and MAP decoders on them; no figure or table reads them |
| Table IV ladder | `loss_ler_diag/n5_hw_L16.npz`, `r1_campaign/t1_codes/B_s202_{plateau,conv}.npz`, `headline_p05_Lsweep/n5_L28.npz` | unweighted variants |

## Tables

| table | record(s) | regenerate |
|---|---|---|
| I (calibrated comparison) | `r9_campaign/t1_results.json` (exact cells, both codes, full grid); σ_a² = 0 check `r12_scrutiny/t1a_v1_results.json` | `notebooks/central_comparison.ipynb` recomputes every cell from the parameters; `scripts/print_paper_tables.py` |
| II (weight enumerators) | `table3/table3_enumerators.json`, `w5_t0/w5_enum_imp005.json` | `scripts/print_paper_tables.py --recompute` (wire-grouped enumerators from the parameter files) |
| III (encoder resources) | analytic | `src/central_comparison.py::ansatz_resource_counts` |
| IV (weight-conditioned decomposition) | `r1_campaign/t1_results.json` (central, plateau/converged rows), `loss_ler_diag/t1c_results.json` + `t1e_results.json` (channel-basis L = 16 row), `headline_p05_Lsweep/lsweep.json` + `loss_ler_diag/t1g_results.json` (deep L = 28 row), `r9_campaign/t1_results.json` + `t4_results.json` (Petz column at s = 0.5) | `scripts/print_paper_tables.py` (LER cells are Monte-Carlo records; residuals and Petz cells recompute) |
| V (stopping rules) | `sampling_budget_v2/b_results.json` (arm `W`, $((6,3,3))_3$) | `scripts/print_paper_tables.py` |
| VI (per-step sampler cost) | `r4_campaign/a2_cost_table.json` (exact replay of the draw streams) | `scripts/print_paper_tables.py`; `r4_campaign/a2_cost_table.py` replays |

Certified optimal recovery (App. F): `r12_scrutiny/t2_results.json`
(`t2_jrf.py`, s ∈ {0.25, 0.5}); the notebook recomputes it on the full grid
with `src/central_comparison.py::certified_optimal_fidelity`.

## Figures

Every data figure of the paper is rendered by `scripts/render_paper_figures.py`
(which calls the per-figure functions in `scripts/paper_figures/`) into
`results/figures/`. Inputs:

| figure | file | data records |
|---|---|---|
| 3 register size | `fig_n_w6.png` | i.i.d.: `r2_campaign/t4_results.json`, `fig6_panels/measurements.json`; channel: `r11_gapfill/t1a_family_L2.json` |
| 4 qudit dimension | `fig_q_w5.png` | `fig6_panels/measurements.json`, `r5_campaign/t23_results.json` (reconstruction, anchors), `r9_campaign/t2d_results.json` (channel) |
| 5 distance (repeated rounds) | `fig_dist_w8.png` | `r11_gapfill/t4_analysis.json` |
| 6 depth | `fig_depth_w6.png` | `r9_campaign/t5_results.json`, `t5_loss_traces.npz`, `r11_gapfill/t5_sgrid_results.json` |
| 7 pruning | `fig_prune_w6.png` | `r9_campaign/t5_results.json`, `r11_gapfill/t5_sgrid_results.json` |
| 8 encoder cost by gate | `fig_gatecost_w7.png` | ansatz counts (analytic) |
| 9 gate study | `fig_gates_w7.png` | `r5_campaign/t1_results.json` + `t1_codes/*.npz` (monitor traces), `r4_campaign/a1_results.json`, `r6_campaign/esc_psweep.json`, `r12_scrutiny/t5_results.json` (channel panel) |
| 10 sampler frontier | `fig_sampler_frontier_w8.png` | `r10_sampler/t3_analysis.json` |
| 11 sampler traces | `fig_sampler_traces_w8.png` | `r10_sampler/t1_codes/*.npz` (monitor logs), `t3_analysis.json` |
| 12 paired benchmarks | `fig_headline_pairs_w8.png` | `r11_gapfill/t2_table1_results.json` |

## Shot budgets and seeds

* Central comparison (Sec. VI A): $2\times10^6$ shots per code at p = 0.05,
  seed 3000 (`src/simulation.py::simulate_ler_with_correction_factored`
  with `src/errors.py::make_hardware_noise_fn`); reproduced shot for shot by
  the notebook.
* Weight-conditioned probes of the central code: 200 000 (w = 1, seed
  1001), 100 000 (w = 2, seed 1002), 100 000 (w = 3, seed 1003).
* Every other i.i.d. cell: ≥ 300 000 shots doubled until ≥ 30 errors
  (App. E); the per-cell shot counts and error counts are stored next to
  each LER in the records (`k`, `n_shots`, `wilson95`).
* Calibrated-channel cells: exact (no shots) for n ≤ 7; at n ≥ 8 the
  sampled sector-averaged estimator with the stored standard errors.
* Optimization runs: four independent runs per setting unless stated
  (`s100..s103` in file names); twelve per entangler in the gate study.
