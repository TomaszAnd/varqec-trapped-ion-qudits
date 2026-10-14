#!/usr/bin/env python3
"""Weak-noise asymptotics: small-s asymptotics of the central cells.

Adds exact v2 cells at s in {0.01, 0.03125, 0.0625} for the central code
A_L2_s100 and [[5,1,3]]_Z3 (projective + Petz), then fits the leading
power of the corrected infidelity vs s on the weak end, importing the
committed r9_campaign t1 cells at s in {0.125, 0.25, 0.5} with provenance.

Question: both codes correct every weight-1 error exactly, so genuine
quadratic scaling is expected UNLESS an instrument floor intrudes.
Quantify per code: the projective completion/non-Weyl linear floor
coefficient a (rate ~ a*s + b*s^2), the Petz quadratic coefficient b
(no completion floor), local log-log slopes, and whether the 3.05x
weak-noise Petz margin is a code property or partly an instrument-floor
artifact.

Writes t1a_results.json incrementally (resumable per cell). Feeds the
weak-noise asymptotics of the paper.
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
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "t1a_results.json")
BR = os.path.join(REPO, "results", "campaigns")
R9_T1 = os.path.join(BR, "r9_campaign", "t1_results.json")

SIGMA_P2, SIGMA_A2_CAL = 0.073, 0.047
D, N = 3, 5
GP = [(i, j, 0, 1) for i in range(N) for j in range(i + 1, N)]
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]

S_NEW = (0.01, 0.03125, 0.0625)
S_IMPORT = (0.125, 0.25, 0.5)

CODES = {
    "headline_A_L2_s100": os.path.join(BR, "r1_campaign", "t1_codes",
                                       "A_L2_s100.npz"),
    "five_one_three_Z3": None,
}


def load_results():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return {"meta": {"d": D, "n": N, "gate_steps": len(GP),
                     "s_new": list(S_NEW), "s_imported": list(S_IMPORT),
                     "imported_from": "r9_campaign/t1_results.json",
                     "convention": "eta(s)=e^{-s*0.073}, sigma_a2(s)=s*0.047",
                     "engine": "exact (analytic Gaussian masks)",
                     "metric": "rate = 1 - F_avg (channel infidelity)"},
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

    for s in S_NEW:
        eta, sa2 = float(np.exp(-s * SIGMA_P2)), s * SIGMA_A2_CAL
        steps = build_dose_steps(N, D, GP, eta, sigma_a2=sa2)
        for name, cs in states.items():
            key = f"{name}@v2_s{s:g}"
            if key in res["cells"]:
                continue
            t0 = time.time()
            inst = build_projective_instrument(cs, D, N, eta, sigma_a2=sa2)
            blocks = evolve_code_blocks(cs, steps, D, N)
            proj = entanglement_fidelity_projective(inst, blocks)
            petz = build_petz_exact_dose(cs, steps, D, N)
            pz = entanglement_fidelity_petz(petz, cs, blocks)
            res["cells"][key] = {
                "code": name, "s": s, "eta": eta, "sigma_a2": sa2,
                "projective_rate": proj["rate_1mFavg"],
                "petz_rate": pz["rate_1mFavg"],
                "petz_leak_max": float(max(pz["recovery_leak_per_codeword"])),
                "source": "this run",
                "wall_s": round(time.time() - t0, 1),
            }
            with open(OUT, "w") as f:
                json.dump(res, f, indent=2)
            print(f"{key}: proj={proj['rate_1mFavg']:.6e} "
                  f"petz={pz['rate_1mFavg']:.6e}", flush=True)

    # import committed r9 cells with provenance
    with open(R9_T1) as f:
        r9 = json.load(f)
    for s in S_IMPORT:
        for name in CODES:
            key = f"{name}@v2_s{s:g}"
            if key in res["cells"]:
                continue
            c = r9["cells"][f"{name}@v2_s{s:.3f}"]
            res["cells"][key] = {
                "code": name, "s": s, "eta": c["eta"],
                "sigma_a2": c["sigma_a2"],
                "projective_rate": c["projective_rate"],
                "petz_rate": c["petz_rate"],
                "source": "imported r9_campaign/t1_results.json",
            }

    # ---- analysis ----
    s_all = sorted(S_NEW + S_IMPORT)
    ana = {"s_grid": s_all, "per_code": {}}
    for name in CODES:
        rows = {s: res["cells"][f"{name}@v2_s{s:g}"] for s in s_all}
        proj = np.array([rows[s]["projective_rate"] for s in s_all])
        petz = np.array([rows[s]["petz_rate"] for s in s_all])
        sv = np.array(s_all, dtype=float)
        entry = {"projective_rate": dict(zip(map(str, s_all), proj)),
                 "petz_rate": dict(zip(map(str, s_all), petz))}
        for label, r in (("projective", proj), ("petz", petz)):
            slopes = np.log(r[1:] / r[:-1]) / np.log(sv[1:] / sv[:-1])
            entry[f"{label}_local_slopes"] = {
                f"{s_all[i]}->{s_all[i+1]}": float(slopes[i])
                for i in range(len(slopes))}
        # floor / leading coefficients at the weakest cell
        s0 = sv[0]
        entry["proj_linear_coeff_a_at_s0.01"] = float(proj[0] / s0)
        entry["petz_quadratic_coeff_b_at_s0.01"] = float(petz[0] / s0**2)
        # two-term LS fit rate = a*s + b*s^2 over the four weakest cells
        for label, r in (("projective", proj), ("petz", petz)):
            X = np.stack([sv[:4], sv[:4]**2], axis=1)
            coef, *_ = np.linalg.lstsq(X, r[:4], rcond=None)
            pred = X @ coef
            entry[f"{label}_fit_a_s_plus_b_s2"] = {
                "a": float(coef[0]), "b": float(coef[1]),
                "max_rel_dev": float(np.max(np.abs(pred / r[:4] - 1)))}
        ana["per_code"][name] = entry

    h, f5 = ana["per_code"]["headline_A_L2_s100"], \
        ana["per_code"]["five_one_three_Z3"]
    ana["margins"] = {
        "petz_ratio_per_s": {str(s):
            res["cells"][f"headline_A_L2_s100@v2_s{s:g}"]["petz_rate"]
            / res["cells"][f"five_one_three_Z3@v2_s{s:g}"]["petz_rate"]
            for s in s_all},
        "proj_ratio_per_s": {str(s):
            res["cells"][f"headline_A_L2_s100@v2_s{s:g}"]["projective_rate"]
            / res["cells"][f"five_one_three_Z3@v2_s{s:g}"]["projective_rate"]
            for s in s_all},
        "petz_weak_limit_ratio_b_head_over_b_513":
            h["petz_quadratic_coeff_b_at_s0.01"]
            / f5["petz_quadratic_coeff_b_at_s0.01"],
        "proj_weak_limit_ratio_a_head_over_a_513":
            h["proj_linear_coeff_a_at_s0.01"]
            / f5["proj_linear_coeff_a_at_s0.01"],
    }
    res["analysis"] = ana
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(ana["margins"], indent=1))
    print("done")


if __name__ == "__main__":
    main()
