#!/usr/bin/env python3
"""First exact cells: exact adjudication of the sampled pilot.

Exact (linear-in-rho, unraveling-independent) figures of merit for the
main correlated cells: central code A_L2_s100, r8_s300, [[5,1,3]]_Z3 under
channel v2 at s in {0.75, 1.0, 1.5, 2.0} and v1 (dephasing-only) at
eta in {0.9296, 0.8642}. Projective-MAP instrument + exact Petz, both from
the exact density-matrix engine. Writes t3_results.json incrementally
(resumable: completed (code, setting) cells are skipped on rerun).
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
OUT = os.path.join(HERE, "t3_results.json")
BR = os.path.join(REPO, "results", "campaigns")

SIGMA_P2, SIGMA_A2_CAL = 0.073, 0.047
D, N = 3, 5
GP = [(i, j, 0, 1) for i in range(N) for j in range(i + 1, N)]
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]

CODES = {
    "headline_A_L2_s100": os.path.join(BR, "r1_campaign", "t1_codes",
                                       "A_L2_s100.npz"),
    "r8_s300": os.path.join(BR, "r8_campaign", "r8_codes", "r8_s300.npz"),
    "five_one_three_Z3": None,
}

SETTINGS = ([("v2_s%.2f" % s, float(np.exp(-s * SIGMA_P2)), s * SIGMA_A2_CAL)
             for s in (0.75, 1.0, 1.5, 2.0)]
            + [("v1_eta0.9296", 0.9296, 0.0), ("v1_eta0.8642", 0.8642, 0.0)])


def load_results():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return {"meta": {"d": D, "n": N, "gate_steps": len(GP),
                     "engine": "exact (analytic Gaussian masks)",
                     "metric": "entanglement/Haar-average channel fidelity, "
                               "rate = 1 - F_avg (NOT a shot fraction)"},
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
            rep = dict(inst['report'])
            rep.pop('sector_priors', None)
            rep.pop('sector_claimed_overlap_fro', None)
            res["cells"][key] = {
                "code": name, "setting": tag, "eta": eta, "sigma_a2": sa2,
                "projective": {k: proj[k] for k in
                               ("F_e", "F_avg", "rate_1mFavg")},
                "petz": {k: pz[k] for k in ("F_e", "F_avg", "rate_1mFavg")},
                "petz_leak_max": float(max(pz['recovery_leak_per_codeword'])),
                "petz_support_rank": pz['support_rank'],
                "p_identity_sector": float(probs[0]),
                "p_rest": float(probs[-1]),
                "sector_report": rep,
                "wall_s": round(time.time() - t0, 1),
            }
            with open(OUT, "w") as f:
                json.dump(res, f, indent=2)
            print(f"{key}: proj rate={proj['rate_1mFavg']:.4e} "
                  f"petz rate={pz['rate_1mFavg']:.4e} "
                  f"({res['cells'][key]['wall_s']}s)", flush=True)
    print("done")


if __name__ == "__main__":
    main()
