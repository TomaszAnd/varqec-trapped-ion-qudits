#!/usr/bin/env python3
"""Repeated-rounds analysis (Fig. 5): per-round decay fits and the three
questions (i)-(iii).

Model: F_e(R) - 1/K^2 = A * lambda^R (exponential regime; the fit window
is ALL recorded rounds of a cell -- stated per cell; cells with a single
round get no fit, only the round-1 value). Per-round logical infidelity
is quoted as r_round = 1 - lambda (the per-round entanglement-fidelity
decay factor) and as the per-round rate via favg.
Reads t3_results.json; writes t3_analysis.json.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
K2INV = 1.0 / 9.0   # K = 3

res = json.load(open(os.path.join(HERE, "t3_results.json")))
out = {"model": "F_e(R) - 1/9 = A * lambda^R, LS fit on ln(F_e - 1/9), "
                "fit window = all recorded rounds (stated)",
       "cells": {}}

for key, c in res["cells"].items():
    R = np.array([r["round"] for r in c["rounds"]])
    Fe = np.array([r["F_e"] for r in c["rounds"]])
    y = Fe - K2INV
    cell = {"code": c["code"], "n": c["n"], "s": c["s"],
            "rounds_recorded": int(R.size),
            "F_e_by_round": Fe.tolist()}
    if R.size >= 2:
        coef = np.polyfit(R, np.log(y), 1)
        lam = float(np.exp(coef[0]))
        A = float(np.exp(coef[1]))
        resid = float(np.max(np.abs(np.log(y) - np.polyval(coef, R))))
        cell.update({"lambda_per_round": lam, "A": A,
                     "fit_max_log_resid": resid,
                     "per_round_infid_1mlambda": 1.0 - lam,
                     "fit_window_rounds": [int(R[0]), int(R[-1])]})
    out["cells"][key] = cell

# question (i): does the single-round d4<d3<d2 ordering amplify with rounds?
trio = {}
for s in ("0.25", "0.50"):
    d = {a: out["cells"][f"{a}@s{s}"] for a in ("d2", "d3", "d4")}
    Fes = {a: d[a]["F_e_by_round"] for a in d}
    nr = min(len(v) for v in Fes.values())
    gaps = [Fes["d4"][r] - Fes["d2"][r] for r in range(nr)]
    trio[f"s{s}"] = {
        "Fe_gap_d4_minus_d2_by_round": gaps,
        "ordering_by_round": [
            sorted(("d2", "d3", "d4"),
                   key=lambda a: -Fes[a][r]) for r in range(nr)],
        "lambda": {a: d[a].get("lambda_per_round") for a in d},
        "per_round_infid": {a: d[a].get("per_round_infid_1mlambda")
                            for a in d},
    }
    lam = trio[f"s{s}"]["lambda"]
    if all(v is not None for v in lam.values()):
        trio[f"s{s}"]["per_round_ratio_d2_over_d4"] = (
            (1 - lam["d2"]) / (1 - lam["d4"]))
out["verdict_i_trio"] = trio

# question (ii): per-round ratios trio vs [[5,1,3]] reference (n=5 dose!)
ref = {}
for s in ("0.25", "0.50"):
    row = {}
    for name in ("headline", "five_one_three_Z3"):
        row[name] = out["cells"][f"{name}@s{s}"].get("lambda_per_round")
    row["note"] = ("references run the n=5 dose (10 gate steps/round); "
                   "the trio runs the n=7 dose (21 steps/round) -- "
                   "per-round numbers are NOT dose-matched across n, "
                   "stated by design")
    lam513 = row["five_one_three_Z3"]
    if lam513 is not None:
        for a in ("d2", "d3", "d4"):
            lamt = trio[f"s{s}"]["lambda"][a]
            if lamt is not None:
                row[f"ratio_perround_infid_{a}_over_513"] = (
                    (1 - lamt) / (1 - lam513))
        lamh = row["headline"]
        if lamh is not None:
            row["ratio_perround_infid_headline_over_513"] = (
                (1 - lamh) / (1 - lam513))
    ref[f"s{s}"] = row
out["verdict_ii_vs_513"] = ref

# question (iii): first round with trio |F_e gap| > 1% absolute
iii = {}
for s in ("0.25", "0.50"):
    gaps = trio[f"s{s}"]["Fe_gap_d4_minus_d2_by_round"]
    first = next((i + 1 for i, g in enumerate(gaps) if abs(g) > 0.01), None)
    iii[f"s{s}"] = {
        "first_round_gap_gt_1pct": first,
        "gap_at_that_round": (gaps[first - 1] if first else None),
        "Fe_d4_at_that_round": (out["cells"][f"d4@s{s}"]
                                ["F_e_by_round"][first - 1]
                                if first else None),
        "max_gap_recorded": max(abs(g) for g in gaps),
    }
out["verdict_iii_1pct_round"] = iii

with open(os.path.join(HERE, "t3_analysis.json"), "w") as f:
    json.dump(out, f, indent=2)

print(json.dumps({k: out[k] for k in
                  ("verdict_i_trio", "verdict_ii_vs_513",
                   "verdict_iii_1pct_round")}, indent=1))
