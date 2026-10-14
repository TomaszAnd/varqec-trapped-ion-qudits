#!/usr/bin/env python3
"""Repeated-rounds analysis over the full strength grid: lambda(s) per code.

Model per cell (as in r9_campaign/t3_analyze.py): F_e(R) - 1/K^2 =
A lambda^R, LS fit on ln(F_e - 1/K^2) over ALL recorded rounds. Cells
with a single recorded round (the F_e < 0.5 floor arrives in round 1)
quote the single-round estimator lambda_1 = (F_e(1) - 1/K^2) / (1 - 1/K^2),
labelled "single_round" -- it equals the fit's lambda when A = 1 and is
the natural R = 1 restriction; estimator kind stated per cell.

Deliverable: lambda vs s tables (projective; Petz where run) feeding the
plain lambda-vs-s log-log figure -- one curve per code, no ratios, no
histograms -- plus the single-round rates (round-1 rate_1mFavg) on the
same grid for the optional inset.

Run after t4_rounds_queue.py completes: python3 t4_analyze.py
Writes t4_analysis.json. Feeds Fig. 5.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
K = 3
K2INV = 1.0 / K ** 2

res = json.load(open(os.path.join(HERE, "t4_rounds_results.json")))
out = {"model": "F_e(R) - 1/9 = A lambda^R (fit over all recorded rounds); "
                "single-round cells: lambda_1 = (F_e(1) - 1/9)/(8/9), "
                "estimator kind stated",
       "cells": {}, "petz_cells": {}}


def analyze(cell):
    R = np.array([r["round"] for r in cell["rounds"]])
    Fe = np.array([r["F_e"] for r in cell["rounds"]])
    y = Fe - K2INV
    row = {"code": cell["code"], "n": cell["n"], "s": cell["s"],
           "rounds_recorded": int(R.size),
           "F_e_by_round": Fe.tolist(),
           "single_round_rate": cell["rounds"][0]["rate_1mFavg"],
           "provenance": cell.get("reused_from", "r11 t4")}
    if R.size >= 2 and np.all(y > 0):
        coef = np.polyfit(R, np.log(y), 1)
        lam = float(np.exp(coef[0]))
        resid = float(np.max(np.abs(np.log(y) - np.polyval(coef, R))))
        row.update({"lambda": lam, "estimator": "fit",
                    "A": float(np.exp(coef[1])),
                    "fit_max_log_resid": resid})
    else:
        lam = float(y[0] / (1.0 - K2INV))
        row.update({"lambda": lam, "estimator": "single_round"})
    row["per_round_infid_1mlambda"] = 1.0 - row["lambda"]
    return row

for key, c in res["cells"].items():
    out["cells"][key] = analyze(c)
for key, c in res["petz_cells"].items():
    out["petz_cells"][key] = analyze(c)

# lambda(s) tables per code (the figure feed)
S_ALL = (0.125, 0.25, 0.5, 1.0, 1.5)
codes = ["d2", "d3", "d4", "headline", "five_one_three_Z3"]
table = {}
for name in codes:
    rows = {}
    for s in S_ALL:
        k = f"{name}@s{s:.2f}"
        if k in out["cells"]:
            c = out["cells"][k]
            rows[f"s{s}"] = {"lambda": c["lambda"],
                             "per_round_infid": c["per_round_infid_1mlambda"],
                             "estimator": c["estimator"],
                             "single_round_rate": c["single_round_rate"],
                             "rounds": c["rounds_recorded"]}
    table[name] = rows
out["lambda_vs_s_projective"] = table

ptable = {}
for name in codes:
    rows = {}
    for s in S_ALL:
        k = f"{name}@s{s:.2f}"
        if k in out["petz_cells"]:
            c = out["petz_cells"][k]
            rows[f"s{s}"] = {"lambda": c["lambda"],
                             "per_round_infid": c["per_round_infid_1mlambda"],
                             "estimator": c["estimator"],
                             "rounds": c["rounds_recorded"]}
    if rows:
        ptable[name] = rows
out["lambda_vs_s_petz"] = ptable

# trio ordering by s (projective)
ordering = {}
for s in S_ALL:
    lam = {a: table[a].get(f"s{s}", {}).get("lambda") for a in
           ("d2", "d3", "d4")}
    if all(v is not None for v in lam.values()):
        ordering[f"s{s}"] = {
            "lambda": lam,
            "order_best_first": sorted(lam, key=lambda a: -lam[a]),
            "d4_best": bool(lam["d4"] >= lam["d3"] >= lam["d2"]
                            or (lam["d4"] >= max(lam["d2"], lam["d3"]))),
            "per_round_ratio_d2_over_d4":
                (1 - lam["d2"]) / (1 - lam["d4"])}
out["trio_ordering_by_s"] = ordering

# central code vs 513 per-round ratio by s (n=5, dose-matched)
hv = {}
for s in S_ALL:
    lh = table["headline"].get(f"s{s}", {}).get("lambda")
    l5 = table["five_one_three_Z3"].get(f"s{s}", {}).get("lambda")
    if lh is not None and l5 is not None:
        hv[f"s{s}"] = {"headline_lambda": lh, "s513_lambda": l5,
                       "ratio_perround_infid_headline_over_513":
                           (1 - lh) / (1 - l5)}
out["headline_vs_513_by_s"] = hv

json.dump(out, open(os.path.join(HERE, "t4_analysis.json"), "w"), indent=2)
print(json.dumps({"trio_ordering_by_s": ordering,
                  "headline_vs_513_by_s": hv}, indent=1))
print("done")
