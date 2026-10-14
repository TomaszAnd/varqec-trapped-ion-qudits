#!/usr/bin/env python3
"""Tight validation of the sampled sector-averaged estimator against the
exact density-matrix engine, BOTH unravelings, at n = 5 and n = 7 -- the
test that the unraveling dependence of the earlier sampled instrument is
gone on the linear instrument.

n = 5: [[5,1,3]]_Z3 and the central code A_L2_s100, calibrated v2.
n = 7: t2c d3 selected code (d3_L2_s100), calibrated v2.

Writes t2d_results.json incrementally (resumable per cell). Feeds the
exact-engine validation of the paper.
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
    build_projective_instrument, entanglement_fidelity_projective,
    sampled_entanglement_fidelity,
)
from src.simulation import make_channel_v2_noise_fn, make_meth_mc_noise_fn

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "t2d_results.json")
BR = os.path.join(REPO, "results", "campaigns")

SIGMA_P2, SIGMA_A2 = 0.073, 0.047
ETA = float(np.exp(-SIGMA_P2))

CELLS = [
    ("n5_513", None, 5, 3000),
    ("n5_headline", os.path.join(BR, "r1_campaign", "t1_codes",
                                 "A_L2_s100.npz"), 5, 3000),
    ("n7_d3sel", os.path.join(BR, "r3_campaign", "t2c_codes",
                              "d3_L2_s100.npz"), 7, 600),
]


def main():
    res = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            res = json.load(f)
    for tag, path, n, n_traj in CELLS:
        if tag in res:
            continue
        d = 3
        gp = [(i, j, 0, 1) for i in range(n) for j in range(i + 1, n)]
        conns = [[i, j] for i in range(n) for j in range(i + 1, n)]
        if path is None:
            cs = five_qudit_code_states(d)
        else:
            cs, *_ = encoder_forward(path, conns)
        t0 = time.time()
        inst = build_projective_instrument(cs, d, n, ETA, sigma_a2=SIGMA_A2)
        steps = build_dose_steps(n, d, gp, ETA, sigma_a2=SIGMA_A2)
        blocks = evolve_code_blocks(cs, steps, d, n)
        exact = entanglement_fidelity_projective(inst, blocks)
        wall_exact = time.time() - t0

        cell = {"n": n, "eta": ETA, "sigma_a2": SIGMA_A2,
                "n_traj_per_ensemble": n_traj,
                "exact": {k: exact[k] for k in
                          ("F_e", "F_avg", "rate_1mFavg")},
                "wall_s_exact": round(wall_exact, 1)}
        for mtag, maker, kw in (
                ("modeB_kraus", make_channel_v2_noise_fn, {"n_max": 5}),
                ("modeA_mc", make_meth_mc_noise_fn, {})):
            nf = maker(n, d, gp, ETA, sigma_a2=SIGMA_A2, **kw)
            t0 = time.time()
            est = sampled_entanglement_fidelity(inst, nf, n_traj,
                                                seed=hash(mtag) % 2 ** 31)
            est["wall_s"] = round(time.time() - t0, 1)
            z = abs(est["F_e"] - exact["F_e"]) / max(est["F_e_se"], 1e-12)
            est["z_vs_exact"] = round(z, 2)
            cell[mtag] = est
            print(f"{tag}/{mtag}: rate={est['rate_1mFavg']:.4e} "
                  f"(exact {exact['rate_1mFavg']:.4e}) z={z:.2f} "
                  f"({est['wall_s']}s)", flush=True)
        dz = (abs(cell["modeA_mc"]["F_e"] - cell["modeB_kraus"]["F_e"])
              / np.hypot(cell["modeA_mc"]["F_e_se"],
                         cell["modeB_kraus"]["F_e_se"]))
        cell["z_modeA_vs_modeB"] = round(float(dz), 2)
        res[tag] = cell
        with open(OUT, "w") as f:
            json.dump(res, f, indent=2)
        print(f"{tag}: modeA vs modeB z = {dz:.2f}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
