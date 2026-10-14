#!/usr/bin/env python3
"""Residual-ladder crossover analysis.

Merges every exact v2 physical-channel cell for the full residual
ladder (Table IV/V codes + the channel-weighted depth ladder) with
[[5,1,3]] as the reference, over the full s grid, both instruments:

  - central code + [[5,1,3]]: r9_campaign/t1_results.json (full grid)
  - hw_basis_L16 / plateau / converged / deep_L28:
      r9_campaign/t4_results.json (s 0.5, 1.0)
      + r12_scrutiny/t3_ladder_results.json (s 0.125, 0.25, 1.5)
  - depth ladder A_L{2,4,8,16} x seeds 100..103:
      r11_gapfill/t5_sgrid_results.json (full grid); per-code
      full-basis residuals from r1_campaign/t1_results.json units.

Answers: (a) does ANY trained code cross below [[5,1,3]] at any s
under either instrument; (b) the gap of the best trained code per s;
(c) within the channel-weighted depth family, the rate-vs-residual
trend and the extrapolated residual needed to close the gap.

Writes t3_crossover.json. Feeds the residual ladder of the paper.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BR = os.path.abspath(os.path.join(HERE, ".."))

S_GRID = (0.125, 0.25, 0.5, 1.0, 1.5)

r9t1 = json.load(open(os.path.join(BR, "r9_campaign", "t1_results.json")))
r9t4 = json.load(open(os.path.join(BR, "r9_campaign", "t4_results.json")))
t3l = json.load(open(os.path.join(HERE, "t3_ladder_results.json")))
t5g = json.load(open(os.path.join(BR, "r11_gapfill",
                                  "t5_sgrid_results.json")))
r1t1 = json.load(open(os.path.join(BR, "r1_campaign", "t1_results.json")))

# ---- assemble: code -> {residual, family, cells: {s: (proj, petz)}} ----
codes = {}


def put(name, resid, family, s, proj, petz, src):
    c = codes.setdefault(name, {"residual": resid, "family": family,
                                "cells": {}, "sources": set()})
    c["cells"][s] = (proj, petz)
    c["sources"].add(src)


for s in S_GRID:
    tag = f"v2_s{s:.3f}"
    c = r9t1["cells"][f"headline_A_L2_s100@{tag}"]
    put("headline_A_L2_s100", 4.599, "chan-wtd L2 (headline)", s,
        c["projective_rate"], c["petz_rate"], "r9t1")
    c = r9t1["cells"][f"five_one_three_Z3@{tag}"]
    put("five_one_three_Z3", 1e-30, "stabilizer", s,
        c["projective_rate"], c["petz_rate"], "r9t1")

LADDER = ("hw_basis_L16", "plateau_B_s202", "converged_B_s202", "deep_L28")
for src, blob, ss, fmt in (
        ("r9t4", r9t4, (0.5, 1.0), "{n}@s{s:.3f}"),
        ("t3_ladder", t3l, (0.125, 0.25, 1.5), "{n}@v2_s{s:.3f}")):
    for name in LADDER:
        for s in ss:
            c = blob["cells"][fmt.format(n=name, s=s)]
            put(name, c["residual"], "TableIV ladder", s,
                c["projective_rate"], c["petz_rate"], src)

for key, c in t5g["cells"].items():
    name = c["code"]                      # depth_L{L}_s{seed} | prune_*
    if not name.startswith("depth_"):
        continue                          # pruning ladder is out of scope here
    unit = name.replace("depth_", "A_")   # A_L{L}_s{seed}
    resid = r1t1["units"][unit]["full_basis_loss"]
    put(name, resid, f"chan-wtd depth L={c['L']}", c["s"],
        c["projective_rate"], c["petz_rate"], "r11t5")

# ---- crossover scan ----
ref = codes["five_one_three_Z3"]
out = {"meta": {"s_grid": list(S_GRID),
                "metric": "rate = 1 - F_avg, exact v2 cells",
                "question": "does any trained code cross below [[5,1,3]]"},
       "per_s": {}, "codes": {}}

for name, c in codes.items():
    out["codes"][name] = {"residual": c["residual"], "family": c["family"],
                          "sources": sorted(c["sources"]),
                          "cells": {str(s): {"proj": c["cells"][s][0],
                                             "petz": c["cells"][s][1]}
                                    for s in sorted(c["cells"])}}

any_cross = {"proj": [], "petz": []}
for s in S_GRID:
    ref_p, ref_z = ref["cells"][s]
    rows = []
    for name, c in codes.items():
        if name == "five_one_three_Z3" or s not in c["cells"]:
            continue
        p, z = c["cells"][s]
        rows.append({"code": name, "residual": c["residual"],
                     "proj": p, "petz": z,
                     "proj_ratio_vs_513": p / ref_p,
                     "petz_ratio_vs_513": z / ref_z})
        for inst, ratio in (("proj", p / ref_p), ("petz", z / ref_z)):
            if ratio < 1.0:
                any_cross[inst].append((name, s, ratio))
    rows.sort(key=lambda r: r["petz"])
    best = rows[0]
    out["per_s"][str(s)] = {
        "ref_513": {"proj": ref_p, "petz": ref_z},
        "best_trained_by_petz": {"code": best["code"],
                                 "petz_ratio": best["petz_ratio_vs_513"]},
        "best_trained_by_proj": min(rows, key=lambda r: r["proj"])["code"],
        "min_proj_ratio": min(r["proj_ratio_vs_513"] for r in rows),
        "min_petz_ratio": min(r["petz_ratio_vs_513"] for r in rows),
        "ladder_sorted_by_petz": rows,
    }

out["crossovers_found"] = {k: v for k, v in any_cross.items()}
out["verdict_crossover"] = ("NONE — no trained code crosses below "
                            "[[5,1,3]] at any strength under either "
                            "instrument"
                            if not any_cross["proj"] and not any_cross["petz"]
                            else "CROSSOVER FOUND — see crossovers_found")

# ---- within-family trend + extrapolation (channel-weighted depth) ----
import numpy as np  # noqa: E402

trend = {}
for s in S_GRID:
    med = {}
    for L in (2, 4, 8, 16):
        sel = [codes[f"depth_L{L}_s{seed}"] for seed in (100, 101, 102, 103)]
        med[L] = {
            "residual_median": float(np.median([c["residual"] for c in sel])),
            "petz_median": float(np.median([c["cells"][s][1] for c in sel])),
            "proj_median": float(np.median([c["cells"][s][0] for c in sel])),
            "petz_best_seed": float(min(c["cells"][s][1] for c in sel)),
        }
    # power-law exponent gamma: rate ~ residual^gamma from L2 -> L16 medians
    r2, r16 = med[2], med[16]
    gamma = (np.log(r16["petz_median"] / r2["petz_median"])
             / np.log(r16["residual_median"] / r2["residual_median"]))
    ref_z = ref["cells"][s][1]
    gap = r16["petz_median"] / ref_z
    # residual multiple needed to close the remaining gap at this gamma
    needed = (np.log(1.0 / gap) / gamma if gamma != 0 else float("inf"))
    trend[str(s)] = {
        "per_L_medians": med,
        "gamma_petz_rate_vs_residual": float(gamma),
        "gap_L16_median_vs_513_petz": float(gap),
        "log10_residual_drop_needed_to_close_gap":
            float(needed / np.log(10)),
        "best_single_seed_gap_at_L16":
            float(r16["petz_best_seed"] / ref_z),
    }
out["family_trend"] = trend

with open(os.path.join(HERE, "t3_crossover.json"), "w") as f:
    def _default(o):
        raise TypeError(o)
    json.dump(out, f, indent=2)

print(out["verdict_crossover"])
for s in S_GRID:
    ps = out["per_s"][str(s)]
    tr = trend[str(s)]
    print(f"s={s}: min petz ratio {ps['min_petz_ratio']:.3f} "
          f"({ps['best_trained_by_petz']['code']}), "
          f"min proj ratio {ps['min_proj_ratio']:.3f}; "
          f"gamma={tr['gamma_petz_rate_vs_residual']:.4f}, "
          f"log10 resid drop to close: "
          f"{tr['log10_residual_drop_needed_to_close_gap']:.1f}")
