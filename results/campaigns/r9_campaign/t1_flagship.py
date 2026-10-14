#!/usr/bin/env python3
"""Exact calibrated-channel cells for the central comparison (Table I,
Figs. 3-7).

Exact engine, channel v2, strength grid s in {0.125, 0.25, 0.5, 1.0, 1.5}
(eta(s) = e^{-s*0.073}, sigma_a2(s) = s*0.047, s = 1 is the calibrated
point), plus one v1 (sigma_a2 = 0) row at s = 1 for the jitter-effect
statement. Codes: central code A_L2_s100 + its three sibling v1-trained
seeds (s101-103; the best-of-4 on the physical channel is identified in
the output), r8_s300 (v2-trained representative), [[5,1,3]]_Z3.
Projective-MAP + exact Petz rates; ratios vs [[5,1,3]] per strength are
computed by the summary block.

Writes t1_results.json incrementally (resumable per cell).
"""
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

import numpy as np

import src.jax_backend  # noqa: F401  (x64 first)
from src.catalog import five_qudit_code_states
from src.decoders._common import encoder_forward
from src.decoders.exact_channel import build_dose_steps, evolve_code_blocks
from src.decoders.projective_decode import (
    build_petz_exact_dose, build_projective_instrument,
    entanglement_fidelity_petz, entanglement_fidelity_projective,
    sector_outcome_probs,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "t1_results.json")
BR = os.path.join(REPO, "results", "campaigns")

SIGMA_P2, SIGMA_A2_CAL = 0.073, 0.047
D, N = 3, 5
GP = [(i, j, 0, 1) for i in range(N) for j in range(i + 1, N)]
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]

CODES = {
    "headline_A_L2_s100": os.path.join(BR, "r1_campaign", "t1_codes",
                                       "A_L2_s100.npz"),
    "A_L2_s101": os.path.join(BR, "r1_campaign", "t1_codes", "A_L2_s101.npz"),
    "A_L2_s102": os.path.join(BR, "r1_campaign", "t1_codes", "A_L2_s102.npz"),
    "A_L2_s103": os.path.join(BR, "r1_campaign", "t1_codes", "A_L2_s103.npz"),
    "r8_s300": os.path.join(BR, "r8_campaign", "r8_codes", "r8_s300.npz"),
    "five_one_three_Z3": None,
}

S_GRID = (0.125, 0.25, 0.5, 1.0, 1.5)
SETTINGS = ([("v2_s%.3f" % s, float(np.exp(-s * SIGMA_P2)), s * SIGMA_A2_CAL)
             for s in S_GRID]
            + [("v1_s1.000", float(np.exp(-SIGMA_P2)), 0.0)])


def load_results():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return {"meta": {"d": D, "n": N, "gate_steps": len(GP),
                     "s_grid": list(S_GRID),
                     "convention": "eta(s)=e^{-s*0.073}, sigma_a2(s)=s*0.047",
                     "engine": "exact (analytic Gaussian masks)",
                     "metric": "channel infidelity rate = 1 - F_avg "
                               "(NOT a shot fraction)"},
            "cells": {}}


def main():
    res = load_results()
    states = {}
    for name, path in CODES.items():
        if path is None:
            states[name] = five_qudit_code_states(D)
        else:
            cs, d_, n_, K_ = encoder_forward(path, CONNS)
            assert (d_, n_) == (D, N)
            states[name] = cs
        print(f"loaded {name}: K={states[name].shape[0]}", flush=True)

    for tag, eta, sa2 in SETTINGS:
        steps = build_dose_steps(N, D, GP, eta, sigma_a2=sa2)
        for name, cs in states.items():
            key = f"{name}@{tag}"
            if key in res["cells"]:
                continue
            t0 = time.time()
            inst = build_projective_instrument(cs, D, N, eta, sigma_a2=sa2)
            blocks = evolve_code_blocks(cs, steps, D, N)
            proj = entanglement_fidelity_projective(inst, blocks)
            petz = build_petz_exact_dose(cs, steps, D, N)
            pz = entanglement_fidelity_petz(petz, cs, blocks)
            K = cs.shape[0]
            rho_avg = sum(blocks[(i, i)] for i in range(K)) / K
            probs = sector_outcome_probs(inst, rho_avg)
            res["cells"][key] = {
                "code": name, "setting": tag, "eta": eta, "sigma_a2": sa2,
                "projective_rate": proj["rate_1mFavg"],
                "petz_rate": pz["rate_1mFavg"],
                "projective_F_e": proj["F_e"], "petz_F_e": pz["F_e"],
                "petz_leak_max": float(max(pz['recovery_leak_per_codeword'])),
                "p_identity_sector": float(probs[0]),
                "p_rest": float(probs[-1]),
                "wall_s": round(time.time() - t0, 1),
            }
            with open(OUT, "w") as f:
                json.dump(res, f, indent=2)
            print(f"{key}: proj={proj['rate_1mFavg']:.4e} "
                  f"petz={pz['rate_1mFavg']:.4e} "
                  f"({res['cells'][key]['wall_s']}s)", flush=True)

    # summary: ratios vs [[5,1,3]] per setting + best-of-4 v1 seed
    summary = {}
    for tag, eta, sa2 in SETTINGS:
        ref = res["cells"][f"five_one_three_Z3@{tag}"]
        row = {}
        for name in CODES:
            c = res["cells"][f"{name}@{tag}"]
            row[name] = {
                "projective_rate": c["projective_rate"],
                "petz_rate": c["petz_rate"],
                "ratio_proj_vs_513": c["projective_rate"]
                                     / ref["projective_rate"],
                "ratio_petz_vs_513": c["petz_rate"] / ref["petz_rate"],
            }
        a_seeds = {n_: row[n_]["petz_rate"] for n_ in
                   ("headline_A_L2_s100", "A_L2_s101", "A_L2_s102",
                    "A_L2_s103")}
        row["best_v1_seed_by_petz"] = min(a_seeds, key=a_seeds.get)
        summary[tag] = row
    res["summary"] = summary
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print("done")


if __name__ == "__main__":
    main()
