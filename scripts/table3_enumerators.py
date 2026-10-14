#!/usr/bin/env python3
"""Shor-Laflamme weight enumerators of every code plotted or tabulated during
the campaigns, with per-row figure cross-references (the record behind the
paper's Table II: results/campaigns/table3/table3_enumerators.{json,md}).
scripts/print_paper_tables.py II prints the twelve rows of the paper from it
and recomputes them with --recompute.

Uses reproduce_tables.enumerators -- the WIRE-GROUPED path only (local error
application; the dense-projector path is 6.2 GB at n=9 and will not finish).
Hardware single-qudit error basis, weights j <= 2; A_0 = B_0 = 1.

Rows:
  [[5,1,3]]_Z3 analytic                        -> Fig 4b, Fig 6 panels (benchmark)
  campaign ((n,3,3))_3, n=5..9 (L=16 a2a)      -> Fig 4, Fig 6 panel n
  panel codes ((5,2,3))_3 ((5,3,2))_3 ((5,3,3))_4 ((5,3,3))_5 -> Fig 6 K/dist/q
  weighted ((9,3,3))_3 winner (n9_armB_winner) -> Fig 10
(The five Fig-9 sampler codes are intentionally omitted: Fig 9's claim is their
mutual equivalence, carried by that figure's CIs; one weighted ((9,3,3))_3
representative row -- the Fig-10 code -- stands for the family.)

Missing npz sources are skipped loudly (re-run after fig6_panel_train lands).
Writes results/campaigns/table3/table3_enumerators.json + .md.
"""
from __future__ import annotations
import os, sys, json, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import numpy as np
from reproduce_tables import enumerators

BPR = os.path.join(REPO, "results", "campaigns")
OUTDIR = os.path.join(BPR, "table3")

# (key, label, n, d, source, figures)
ROWS = [
    ("t1head", "((5,3,3))_3 central code L2 a2a", 5, 3,
     ("npz", "r1_campaign/t1_codes/A_L2_s100.npz"),
     "Table II central row; Figs. 3, 4, 12"),
    ("s513",  "[[5,1,3]]_Z3 (analytic)",        5, 3, ("catalog", "five_qudit_d3"),
     "Fig 4b, Fig 6 n/K/q/dist (stabilizer benchmark)"),
    ("c_n5",  "((5,3,3))_3 L16 a2a",  5, 3, ("npz", "campaign/d3_n5_dist3_16L.npz"),
     "Fig 4, Fig 6 n/K/q/dist"),
    ("c_n6",  "((6,3,3))_3 L16 a2a",  6, 3, ("npz", "campaign/d3_n6_dist3_16L.npz"),
     "Fig 4, Fig 6 n"),
    ("c_n7",  "((7,3,3))_3 L16 a2a",  7, 3, ("npz", "campaign/d3_n7_dist3_16L.npz"),
     "Fig 4, Fig 6 n"),
    ("c_n8",  "((8,3,3))_3 L16 a2a",  8, 3, ("npz", "campaign/d3_n8_dist3_16L.npz"),
     "Fig 4, Fig 6 n"),
    ("c_n9",  "((9,3,3))_3 L16 a2a",  9, 3, ("npz", "campaign/d3_n9_dist3_16L.npz"),
     "Fig 4, Fig 6 n"),
    ("K2",    "((5,2,3))_3 L16 a2a",  5, 3, ("npz", "fig6_panels/K2_d3_n5_dist3_16L.npz"),
     "Fig 6 K"),
    ("dist2", "((5,3,2))_3 L16 a2a",  5, 3, ("npz", "fig6_panels/d3_n5_dist2_16L.npz"),
     "Fig 6 dist"),
    ("d4K3",  "((5,3,3))_4 L16 a2a",  5, 4, ("npz", "fig6_panels/d4K3_n5_dist3_16L.npz"),
     "Fig 6 q"),
    ("d5K3",  "((5,3,3))_5 L16 a2a",  5, 5, ("npz", "fig6_panels/d5K3_n5_dist3_16L.npz"),
     "Fig 6 q"),
    ("w_n9",  "((9,3,3))_3 weighted winner", 9, 3,
     ("npz", "weighted_race_test/n9_armB_winner.npz"), "Fig 10"),
    # ---- minimal-depth codes of the register, distance and K studies ----
    ("r2_d2n7", "((7,3,2))_3 distance panel L2", 7, 3,
     ("npz", "r2_campaign/t1a_codes/d2_L2_s100.npz"),
     "distance panel (Fig. 5)"),
    ("r2_d3n7", "((7,3,3))_3 distance panel L2", 7, 3,
     ("npz", "r2_campaign/t1a_codes/d3_L2_s100.npz"),
     "distance panel (Fig. 5) and register family (Fig. 3)"),
    ("r2_d4n7", "((7,3,4))_3 distance panel L2 (unweighted)", 7, 3,
     ("npz", "r2_campaign/t1a_codes/d4_L2_s103.npz"),
     "distance panel (unweighted arm)"),
    ("r2_n6", "((6,3,3))_3 matched L2", 6, 3,
     ("npz", "r2_campaign/t4_codes/n6_L2_s102.npz"),
     "register family (Fig. 3)"),
    ("r2_n8", "((8,3,3))_3 matched L2", 8, 3,
     ("npz", "r2_campaign/t4_codes/n8_L2_s100.npz"),
     "register family (Fig. 3)"),
    ("r2_n9", "((9,3,3))_3 matched L2", 9, 3,
     ("npz", "r2_campaign/t4_codes/n9_L2_s100.npz"),
     "register family (Fig. 3)"),
    ("r2_K2n5", "((5,2,3))_3 minimal-L L2", 5, 3,
     ("npz", "r2_campaign/t2_codes/K2_n5_L2_s101.npz"),
     "K study"),
    ("r2_K2n7", "((7,2,3))_3 minimal-L L2", 7, 3,
     ("npz", "r2_campaign/t2_codes/K2_n7_L2_s100.npz"),
     "K study"),
    ("r2_q4L2", "((5,3,3))_4 minimal-L L2", 5, 4,
     ("npz", "r2_campaign/t3_codes/q4_L2_s0.npz"),
     "depth study at q = 4"),
    # ---- channel-weighted minimal-depth codes ----
    ("r3_q4", "((5,3,3))_4 weighted L2", 5, 4,
     ("npz", "r3_campaign/t2b_codes/q4_L2_s100.npz"),
     "q panel (Fig. 4)"),
    ("r3_q5", "((5,3,3))_5 weighted L2", 5, 5,
     ("npz", "r3_campaign/t2b_codes/q5_L2_s100.npz"),
     "q panel (Fig. 4)"),
    ("r3_d4n7", "((7,3,4))_3 channel-weighted L2", 7, 3,
     ("npz", "r3_campaign/t2c_codes/d4_L2_s102.npz"),
     "distance trio (Fig. 5); the d2/d3 arms reproduce the distance-panel codes bit-identically"),
]


def load_states(n, src):
    kind, ref = src
    if kind == "catalog":
        from src.catalog import load_code
        return np.asarray(load_code(ref)["code_states"])
    path = os.path.join(BPR, ref)
    if not os.path.exists(path):
        return None
    from src.decoders._common import encoder_forward
    conns = [[i, j] for i in range(n) for j in range(i + 1, n)]
    cs, *_ = encoder_forward(path, conns)
    return cs


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    jp = os.path.join(OUTDIR, "table3_enumerators.json")
    doc = json.load(open(jp)) if os.path.exists(jp) else {
        "method": "wire-grouped Shor-Laflamme A_w/B_w (reproduce_tables."
                  "enumerators), hardware single-qudit basis, j<=2; dense "
                  "path never used (6.2 GB at n=9)",
        "rows": {}}
    for key, label, n, d, src, figs in ROWS:
        if key in doc["rows"]:
            print(f"[cache] {key} done -- skip", flush=True)
            continue
        cs = load_states(n, src)
        if cs is None:
            print(f"[skip] {key}: {src[1]} not on disk yet", flush=True)
            continue
        t = time.time()
        A, B = enumerators(cs, n, d)
        doc["rows"][key] = dict(
            label=label, n=n, d=d, K=int(cs.shape[0]), source=src[1],
            figures=figs, A=[float(x) for x in A], B=[float(x) for x in B],
            B_minus_A=[float(b - a) for a, b in zip(A, B)],
            wall_s=round(time.time() - t, 1))
        r = doc["rows"][key]
        print(f"{label:34} A1={r['A'][1]:8.4f} A2={r['A'][2]:9.4f} "
              f"B1={r['B'][1]:8.4f} B2={r['B'][2]:9.4f} "
              f"B1-A1={r['B_minus_A'][1]:8.4f} ({r['wall_s']}s)", flush=True)
        json.dump(doc, open(jp, "w"), indent=2)

    md = ["# Table III — Shor–Laflamme weight enumerators (hardware basis, j ≤ 2)",
          "", "Wire-grouped path (reproduce_tables.enumerators). "
          "B_w − A_w = 0 for w < dist is the (exact-)detection condition; "
          "trained codes satisfy it only approximately (see "
          "the paper, Sec. VII — the residual is a conservative certificate and does not order the measured LER).", "",
          "| code | K | A1 | A2 | B1 | B2 | B1−A1 | B2−A2 | shown in |",
          "|---|---|---|---|---|---|---|---|---|"]
    for key, label, *_ in ROWS:
        if key not in doc["rows"]:
            continue
        r = doc["rows"][key]
        md.append(f"| {r['label']} | {r['K']} | {r['A'][1]:.4f} | {r['A'][2]:.4f} "
                  f"| {r['B'][1]:.4f} | {r['B'][2]:.4f} | {r['B_minus_A'][1]:.4f} "
                  f"| {r['B_minus_A'][2]:.4f} | {r['figures']} |")
    open(os.path.join(OUTDIR, "table3_enumerators.md"), "w").write("\n".join(md) + "\n")
    print(f"wrote {jp} + .md ({len(doc['rows'])}/{len(ROWS)} rows)", flush=True)


if __name__ == "__main__":
    main()
