#!/usr/bin/env python3
"""Print Tables I-VI of the paper from the committed records, next to the
printed values, and flag any mismatch.

Usage (from the repository root):

    PYTHONPATH=. python3 scripts/print_paper_tables.py [--recompute] [I II III IV V VI]

Without --recompute every entry is read from the committed campaign record
that the manuscript was typeset from (docs/PROVENANCE.md).  With
--recompute the entries that can be recomputed cheaply from the parameter
files are recomputed instead of read:

  Table I   exact channel infidelities of both codes on the strength grid
            (projective syndrome decoder and Petz recovery; ~10 s)
  Table II  Shor-Laflamme enumerators of all twelve codes (wire-grouped
            path; ~1 min, the n = 9 rows dominate)
  Table IV  the full-basis residuals and the Petz column at s = 0.5
            (~10 s); the LER cells are Monte-Carlo records and stay as read

Table III is analytic; Tables V and VI are summaries of logged optimisation
runs and are always read from the records.
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import numpy as np

CAMP = os.path.join(REPO, "results", "campaigns")


def jload(*parts):
    with open(os.path.join(CAMP, *parts)) as fh:
        return json.load(fh)


def flag(a, b, tol):
    """'ok' if |a-b| <= tol else 'MISMATCH'."""
    try:
        return "ok" if abs(float(a) - float(b)) <= tol else "MISMATCH"
    except (TypeError, ValueError):
        return "?"


def code_states_from(rel_npz, n):
    import src.jax_backend  # noqa: F401 (64-bit JAX before building encoders)
    from src.decoders._common import encoder_forward
    conns = [[i, j] for i in range(n) for j in range(i + 1, n)]
    cs, *_ = encoder_forward(os.path.join(CAMP, rel_npz), conns)
    return np.asarray(cs)


# ───────────────────────────── Table I ──────────────────────────────────
PAPER_T1 = {0.125: (5.66, 0.70, 4.91, 0.23, 3.05), 0.25: (11.78, 2.05, 10.41, 0.84, 2.45),
            0.5: (23.17, 6.03, 21.05, 3.08, 1.96), 1.0: (39.62, 16.22, 37.16, 10.43, 1.56),
            1.5: (49.36, 26.32, 47.18, 19.42, 1.36)}


def table_i(recompute):
    print("\nTABLE I  calibrated-channel comparison, 1 - F_avg in units of 1e-2 "
          "(one noise round, 10 steps); columns: central proj / Petz, stabilizer proj / Petz, Petz ratio")
    if recompute:
        from src import central_comparison as cc
        cen_cs, stab_cs = cc.central_code_states(), cc.stabilizer_code_states()
        src_label = "recomputed (src.central_comparison.channel_cell)"
    else:
        rec = jload("r9_campaign", "t1_results.json")["cells"]
        src_label = "record r9_campaign/t1_results.json"
    print(f"source: {src_label}")
    print(f"{'s':>6} | {'proj':>7} {'Petz':>7} | {'proj':>7} {'Petz':>7} | {'ratio':>6} | paper | check")
    for s, pv in PAPER_T1.items():
        if recompute:
            from src import central_comparison as cc
            steps = cc.noise_round_steps(5, 3, s)
            a = cc.channel_cell(cen_cs, s, steps=steps)
            b = cc.channel_cell(stab_cs, s, steps=steps)
            vals = (a["projective_rate"], a["petz_rate"], b["projective_rate"], b["petz_rate"])
        else:
            a = rec[f"headline_A_L2_s100@v2_s{s:.3f}"]
            b = rec[f"five_one_three_Z3@v2_s{s:.3f}"]
            vals = (a["projective_rate"], a["petz_rate"], b["projective_rate"], b["petz_rate"])
        vals = tuple(100 * v for v in vals) + (vals[1] / vals[3],)
        checks = [flag(v, p, 0.0051) for v, p in zip(vals, pv)]
        print(f"{s:6.3f} | {vals[0]:7.2f} {vals[1]:7.2f} | {vals[2]:7.2f} {vals[3]:7.2f} | {vals[4]:6.2f} | "
              f"{pv[0]:.2f}/{pv[1]:.2f}/{pv[2]:.2f}/{pv[3]:.2f}/{pv[4]:.2f} | "
              f"{'ok' if all(c == 'ok' for c in checks) else checks}")


# ───────────────────────────── Table II ─────────────────────────────────
# (label, record key in table3_enumerators.json or 'w5', npz relative path, n, d, paper A1 A2 B1 B2)
T2_ROWS = [
    ("((5,3,3))_3 channel-weighted L=2 (central)", "t1head", "r1_campaign/t1_codes/A_L2_s100.npz", 5, 3,
     (2.4408, 2.4229, 2.4631, 3.1225)),
    ("[[5,1,3]]_Z3 (analytical)", "s513", None, 5, 3, (2.2222, 1.9753, 2.2222, 1.9753)),
    ("((5,3,3))_3 unweighted full, L=16", "c_n5", "campaign/d3_n5_dist3_16L.npz", 5, 3,
     (3.5135, 6.4561, 3.5156, 6.5717)),
    ("((6,3,3))_3 L=2", "r2_n6", "r2_campaign/t4_codes/n6_L2_s102.npz", 6, 3, (2.7699, 3.2730, 2.7783, 3.6047)),
    ("((7,3,3))_3 L=2", "r2_d3n7", "r2_campaign/t1a_codes/d3_L2_s100.npz", 7, 3, (3.1706, 4.3496, 3.1735, 4.5137)),
    ("((8,3,3))_3 L=2", "r2_n8", "r2_campaign/t4_codes/n8_L2_s100.npz", 8, 3, (3.6052, 5.7419, 3.6062, 5.8135)),
    ("((9,3,3))_3 L=2", "r2_n9", "r2_campaign/t4_codes/n9_L2_s100.npz", 9, 3, (4.0181, 7.1794, 4.0186, 7.2096)),
    ("((7,3,2))_3 L=2", "r2_d2n7", "r2_campaign/t1a_codes/d2_L2_s100.npz", 7, 3, (3.1101, 4.1689, 3.1195, 4.5015)),
    ("((7,3,4))_3 channel-weighted L=2", "r3_d4n7", "r3_campaign/t2c_codes/d4_L2_s102.npz", 7, 3,
     (3.1867, 4.3922, 3.1923, 4.5561)),
    ("((5,3,3))_4 channel-weighted L=2", "r3_q4", "r3_campaign/t2b_codes/q4_L2_s100.npz", 5, 4,
     (7.7117, 23.9410, 7.7163, 24.2102)),
    ("((5,3,3))_5 channel-weighted L=2", "r3_q5", "r3_campaign/t2b_codes/q5_L2_s100.npz", 5, 5,
     (14.6438, 85.7968, 14.6453, 85.9356)),
    ("((9,3,3))_3 chan.-wtd L=2 (sampler, imp. f=0.05)", "w5", "r10_sampler/t1_codes/importance_0.05_s100.npz", 9, 3,
     (4.0050, 7.1371, 4.0064, 7.1883)),
]


def table_ii(recompute):
    print("\nTABLE II  Shor-Laflamme weight enumerators over the channel error basis (A1 A2 B1 B2)")
    rec = jload("table3", "table3_enumerators.json")["rows"]
    w5 = jload("w5_t0", "w5_enum_imp005.json")
    if recompute:
        from reproduce_tables import enumerators
        from src.catalog import five_qudit_code_states
    print(f"source: {'recomputed from the parameter files' if recompute else 'records table3/table3_enumerators.json, w5_t0/w5_enum_imp005.json'}")
    for label, key, npz, n, d, paper in T2_ROWS:
        if recompute:
            cs = five_qudit_code_states(d) if npz is None else code_states_from(npz, n)
            A, B = enumerators(cs, n, d)
            vals = (A[1], A[2], B[1], B[2])
        elif key == "w5":
            vals = (w5["A1"], w5["A2"], w5["B1"], w5["B2"])
        else:
            r = rec[key]
            vals = (r["A"][1], r["A"][2], r["B"][1], r["B"][2])
        checks = [flag(v, p, 5.1e-5) for v, p in zip(vals, paper)]
        print(f"{label:48s} {vals[0]:8.4f} {vals[1]:8.4f} {vals[2]:8.4f} {vals[3]:8.4f} | "
              f"paper {paper[0]:.4f} {paper[1]:.4f} {paper[2]:.4f} {paper[3]:.4f} | "
              f"{'ok' if all(c == 'ok' for c in checks) else checks}")


# ───────────────────────────── Table III ────────────────────────────────
T3 = [((5, 3), (10, 90, 180, 20, 20, 10, 40)), ((6, 3), (15, 120, 240, 24, 30, 12, 60)),
      ((7, 3), (21, 154, 308, 28, 42, 14, 84)), ((8, 3), (28, 192, 384, 32, 56, 16, 112)),
      ((9, 3), (36, 234, 468, 36, 72, 18, 144)), ((5, 4), (10, 135, 270, 30, 30, 15, 60)),
      ((5, 5), (10, 180, 360, 40, 40, 20, 80))]


def table_iii(recompute):
    from src.central_comparison import ansatz_resource_counts
    print("\nTABLE III  encoder resources at L = 2, all-to-all: |E| P_l params XY/layer MS/layer Z/layer MS budget")
    for (n, q), paper in T3:
        r = ansatz_resource_counts(n, q, L=2)
        vals = (r["E"], r["P_layer"], r["params"], r["XY_per_layer"], r["MS_per_layer"], r["Z_per_layer"], r["MS_budget"])
        print(f"((({n},3,3))_{q}) {vals} | paper {paper} | {'ok' if vals == paper else 'MISMATCH'}")


# ───────────────────────────── Table IV ─────────────────────────────────
def table_iv(recompute):
    print("\nTABLE IV  weight-conditioned decomposition at n=5, q=3: MS, residual, LER|w=2, LER|w=3, iid LER p=0.05, Petz s=0.5")
    r1 = jload("r1_campaign", "t1_results.json")
    r9t1 = jload("r9_campaign", "t1_results.json")["cells"]
    r9t4 = jload("r9_campaign", "t4_results.json")["cells"]
    t1c = jload("loss_ler_diag", "t1c_results.json")["per_L"][0]
    t1e = jload("loss_ler_diag", "t1e_results.json")["hwL16"]
    t1b = jload("loss_ler_diag", "t1b_results.json")["weight_conditioned"]["L28"]
    lsweep = {r["L"]: r for r in jload("headline_p05_Lsweep", "lsweep.json")["per_L"]}
    u = r1["units"]["A_L2_s100"]
    b202 = r1["units"]["B_s202"]["snapshots"]
    head = r1["headline"]["measurement"]
    assert t1c["L"] == 16
    rows = [
        ("((5,3,3))_3 channel-weighted L=2 (central)", 40, u["full_basis_loss"], u["ler_w2"]["ler"], u["ler_w3"]["ler"],
         head["t1_headline"]["ler"], r9t1["headline_A_L2_s100@v2_s0.500"]["petz_rate"],
         (4.6, 0.0, 0.289, 3.42e-4, 6.03e-2), "r1_campaign/t1_codes/A_L2_s100.npz"),
        ("((5,3,3))_3 channel basis, full batch, L=16", 320, t1c["full_basis_loss"], t1e["ler_w2"], t1e["ler_w3"],
         t1c["ler_1M_p05"], r9t4["hw_basis_L16@s0.500"]["petz_rate"],
         (2.90, 0.0, 0.460, 4.97e-4, 5.65e-2), "loss_ler_diag/n5_hw_L16.npz"),
        ("((5,3,3))_3 full basis, residual plateau, L=16", 320, b202["plateau"]["full_basis_loss"],
         b202["plateau"]["ler_w2"]["ler"], b202["plateau"]["ler_w3"]["ler"], b202["plateau"]["ler_iid_p05"]["ler"], None,
         (0.449, 2.8e-3, 0.187, 2.75e-4, None), "r1_campaign/t1_codes/B_s202_plateau.npz"),
        ("  same run, optimized to convergence", 320, b202["conv"]["full_basis_loss"],
         b202["conv"]["ler_w2"]["ler"], b202["conv"]["ler_w3"]["ler"], b202["conv"]["ler_iid_p05"]["ler"], None,
         (0.360, 3.58e-2, 0.140, 8.90e-4, None), "r1_campaign/t1_codes/B_s202_conv.npz"),
        ("((5,3,3))_3 full basis, deep, L=28", 560, lsweep[28]["best_full_loss"], t1b["w2"]["ler"], t1b["w3"]["ler"],
         lsweep[28]["ler_300k"], r9t4["deep_L28@s0.500"]["petz_rate"],
         (1.95e-4, 4.96e-2, 0.125, 1.19e-3, 2.80e-1), "headline_p05_Lsweep/n5_L28.npz"),
        ("[[5,1,3]]_Z3 (analytical)", 36, 0.0, 0.0, None, head["513"]["ler"],
         r9t1["five_one_three_Z3@v2_s0.500"]["petz_rate"],
         (1e-30, 0.0, 0.729, 8.49e-4, 3.08e-2), None),
    ]
    if recompute:
        import src.jax_backend  # noqa: F401
        from src import central_comparison as cc
        from src.errors import ErrorModel
        from src.jax_backend import create_jax_loss_vmap
        import jax.numpy as jnp
        grouped = ErrorModel(d=3, n_qudit=5, distance=3, closed=True, basis="full").build_grouped(verbose=False)
        loss_fn = create_jax_loss_vmap(grouped, K=3, d=3, n_qudit=5, distance=3)
        steps = cc.noise_round_steps(5, 3, 0.5)
        new_rows = []
        for row in rows:
            label, ms, res, w2, w3, iid, petz, paper, npz = row
            cs = cc.stabilizer_code_states() if npz is None else code_states_from(npz, 5)
            res = float(loss_fn(jnp.asarray(cs)))
            if petz is not None:
                petz = cc.channel_cell(cs, 0.5, steps=steps)["petz_rate"]
            new_rows.append((label, ms, res, w2, w3, iid, petz, paper, npz))
        rows = new_rows
        print("source: residual and Petz columns recomputed; LER cells from the records")
    else:
        print("source: records (r1_campaign, loss_ler_diag, headline_p05_Lsweep, r9_campaign)")
    print("note: the stabilizer LER|w=3 = 0.729 is a probe measurement reproduced by notebooks/central_comparison.ipynb "
          "(100 000 shots, seed 1003); no committed record stores it.")
    for label, ms, res, w2, w3, iid, petz, paper, npz in rows:
        f = lambda x, fmt: ("---" if x is None else fmt.format(x))
        checks = [flag(res, paper[0], 0.1 * max(abs(paper[0]), 1e-3) if paper[0] > 1e-20 else 1e-20),
                  flag(w2, paper[1], 1e-4), flag(w3, paper[2], 1e-3) if w3 is not None else "nb",
                  flag(iid, paper[3], 1e-6), flag(petz, paper[4], 6e-4) if petz is not None else "---"]
        print(f"{label:48s} MS {ms:3d} | res {f(res, '{:.3g}'):>8} | w2 {f(w2, '{:.2e}'):>8} | w3 {f(w3, '{:.3f}'):>6} | "
              f"iid {f(iid, '{:.2e}'):>8} | Petz {f(petz, '{:.2e}'):>8} | paper {paper} | {checks}")


# ───────────────────────────── Table V ──────────────────────────────────
T5_PAPER = {"fixed_final": ("(i) preset 3000 + confirm", (3000, 3000), (1, 1), (0, 0), 165354),
            "naive": ("(ii) every crossing", (1107, 1159), (53, 69), (52, 68), 81368),
            "hysteresis": ("(iii) 5 in a row, 50-step gap", (1133, 1197), (2, 3), (1, 2), 65493),
            "growing": ("(iv) growing batch", (2200, 2400), (0, 0), (0, 0), 110333)}


def table_v(recompute):
    print("\nTABLE V  stopping rules on ((6,3,3))_3, channel-weighted, importance f=0.2, target 1e-4, 4 runs per rule")
    print("source: record sampling_budget_v2/b_results.json (arm W)")
    units = jload("sampling_budget_v2", "b_results.json")["units"]
    for pol, (label, stop_p, conf_p, false_p, total_p) in T5_PAPER.items():
        us = [v for k, v in units.items() if k.startswith(f"W|{pol}|")]
        stops = [u["steps_to_stop"] for u in us]
        conf = [u["n_confirms"] for u in us]
        false = [u["n_false_confirms"] for u in us]
        total = np.mean([u["total_op_evs"] for u in us])
        ok = ((min(stops), max(stops)) == stop_p and (min(conf), max(conf)) == conf_p
              and (min(false), max(false)) == false_p and abs(total - total_p) < 1)
        print(f"{label:32s} runs {len(us)} | stop {min(stops)}-{max(stops)} | confirms {min(conf)}-{max(conf)} "
              f"({min(false)}-{max(false)}) | total {total:,.0f} | paper {stop_p} {conf_p} {false_p} {total_p:,} | "
              f"{'ok' if ok else 'MISMATCH'}")


# ───────────────────────────── Table VI ─────────────────────────────────
T6_PAPER = {"full_batch": ("full batch", (221, 313, 421, 685), 1027500, 1.0),
            "importance_0.2": ("importance @f=0.2", (38, 55, 71, 114), 170736, 6.0),
            "importance_0.35": ("importance @f=0.35", (58, 81, 104, 193), 289290, 3.6),
            "full_basis_is_0.1": ("fb-importance @f=0.1", (27, 37, 49, 74), 111706, 9.2)}


def table_vi(recompute):
    print("\nTABLE VI  realized per-step cost (op-EVs) at n = 5, 6, 7, 9 and preset-1500-step totals at n = 9")
    print("source: record r4_campaign/a2_cost_table.json (exact replay of the draw streams)")
    t = jload("r4_campaign", "a2_cost_table.json")["table"]
    full9 = t["9"]["full_batch"]["train_total_op_evs"]
    for strat, (label, per_step_p, total_p, adv_p) in T6_PAPER.items():
        per_step = tuple(int(round(t[n][strat]["realized_ops_per_step_mean"])) for n in ("5", "6", "7", "9"))
        total = t["9"][strat]["train_total_op_evs"]
        adv = full9 / total
        ok = per_step == per_step_p and total == total_p and abs(adv - adv_p) < 0.051
        print(f"{label:22s} {per_step} | n=9 total {total:,} adv {adv:.1f}x | paper {per_step_p} {total_p:,} {adv_p}x | "
              f"{'ok' if ok else 'MISMATCH'}")


TABLES = {"I": table_i, "II": table_ii, "III": table_iii, "IV": table_iv, "V": table_v, "VI": table_vi}

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    recompute = "--recompute" in sys.argv
    for name in (args or list(TABLES)):
        TABLES[name](recompute)
