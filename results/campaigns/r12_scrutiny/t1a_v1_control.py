#!/usr/bin/env python3
"""Weak-noise asymptotics control: weak-s asymptotics on the dephasing-only
(v1) channel (sigma_a2 = 0).

Attribution test for the linear term found in the v2 Petz rates: the
angle-jitter step is a weight-2 (pair) error at first order, which no
distance-3 code corrects exactly, so it should contribute a genuine
O(s) term to the Petz rate; the role-dephasing factors are weight-1 at
first order and are corrected exactly by both codes. If that reading
is right, the v1 Petz slopes must approach 2 (pure quadratic) at weak
s and the v1 linear fit coefficients must collapse toward zero.

Writes t1a_v1_results.json.
"""
import json
import os
import sys

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
OUT = os.path.join(HERE, "t1a_v1_results.json")
BR = os.path.join(REPO, "results", "campaigns")

SIGMA_P2 = 0.073
D, N = 3, 5
GP = [(i, j, 0, 1) for i in range(N) for j in range(i + 1, N)]
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]
S_GRID = (0.01, 0.03125, 0.0625, 0.125)

CODES = {
    "headline_A_L2_s100": os.path.join(BR, "r1_campaign", "t1_codes",
                                       "A_L2_s100.npz"),
    "five_one_three_Z3": None,
}


def main():
    res = {"meta": {"d": D, "n": N, "gate_steps": len(GP),
                    "s_grid": list(S_GRID), "channel": "v1 (sigma_a2 = 0)",
                    "metric": "rate = 1 - F_avg"},
           "cells": {}}
    if os.path.exists(OUT):
        with open(OUT) as f:
            res = json.load(f)
    states = {}
    for name, path in CODES.items():
        states[name] = (five_qudit_code_states(D) if path is None
                        else encoder_forward(path, CONNS)[0])

    for s in S_GRID:
        eta = float(np.exp(-s * SIGMA_P2))
        steps = build_dose_steps(N, D, GP, eta, sigma_a2=0.0)
        for name, cs in states.items():
            key = f"{name}@v1_s{s:g}"
            if key in res["cells"]:
                continue
            inst = build_projective_instrument(cs, D, N, eta, sigma_a2=0.0)
            blocks = evolve_code_blocks(cs, steps, D, N)
            proj = entanglement_fidelity_projective(inst, blocks)
            petz = build_petz_exact_dose(cs, steps, D, N)
            pz = entanglement_fidelity_petz(petz, cs, blocks)
            res["cells"][key] = {
                "code": name, "s": s, "eta": eta, "sigma_a2": 0.0,
                "projective_rate": proj["rate_1mFavg"],
                "petz_rate": pz["rate_1mFavg"]}
            with open(OUT, "w") as f:
                json.dump(res, f, indent=2)
            print(f"{key}: proj={proj['rate_1mFavg']:.6e} "
                  f"petz={pz['rate_1mFavg']:.6e}", flush=True)

    ana = {}
    sv = np.array(S_GRID)
    for name in CODES:
        petz = np.array([res["cells"][f"{name}@v1_s{s:g}"]["petz_rate"]
                         for s in S_GRID])
        proj = np.array([res["cells"][f"{name}@v1_s{s:g}"]["projective_rate"]
                         for s in S_GRID])
        X = np.stack([sv, sv ** 2], axis=1)
        pc, *_ = np.linalg.lstsq(X, petz, rcond=None)
        ana[name] = {
            "petz_local_slopes": [float(x) for x in
                                  np.log(petz[1:] / petz[:-1])
                                  / np.log(sv[1:] / sv[:-1])],
            "proj_local_slopes": [float(x) for x in
                                  np.log(proj[1:] / proj[:-1])
                                  / np.log(sv[1:] / sv[:-1])],
            "petz_fit_a": float(pc[0]), "petz_fit_b": float(pc[1]),
        }
    res["analysis"] = ana
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(ana, indent=1))


if __name__ == "__main__":
    main()
