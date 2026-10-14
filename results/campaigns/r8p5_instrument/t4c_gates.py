#!/usr/bin/env python3
"""Gate-study bars on the exact density-matrix engine.

All 60 gate-study codes of r5_campaign (ms/zz/ls/csum/cex x seeds
100-111), n = 6, v1 dephasing at eta = 0.8642, exact engine +
projective-MAP + Petz. (No probe-per-gate reduction needed: an exact
n = 6 cell is deterministic and costs well under a minute.) Writes
t4c_results.json incrementally.
"""
import glob
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

import numpy as np

import src.jax_backend  # noqa: F401  (x64 first)
from src.decoders._common import encoder_forward
from src.decoders.exact_channel import build_dose_steps, evolve_code_blocks
from src.decoders.projective_decode import (
    build_petz_exact_dose, build_projective_instrument,
    entanglement_fidelity_petz, entanglement_fidelity_projective,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "t4c_results.json")
CODES_DIR = os.path.join(REPO, "results", "campaigns",
                         "r5_campaign", "t1_codes")
D, N, ETA = 3, 6, 0.8642
GP = [(i, j, 0, 1) for i in range(N) for j in range(i + 1, N)]
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]


def main():
    res = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            res = json.load(f)
    res.setdefault("meta", {"d": D, "n": N, "eta": ETA, "sigma_a2": 0.0,
                            "gate_steps": len(GP),
                            "metric": "rate = 1 - F_avg (channel infidelity)"})
    res.setdefault("cells", {})
    steps = build_dose_steps(N, D, GP, ETA, sigma_a2=0.0)
    paths = sorted(glob.glob(os.path.join(CODES_DIR, "*.npz")))
    print(f"{len(paths)} codes", flush=True)
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        if name in res["cells"]:
            continue
        t0 = time.time()
        cs, *_ = encoder_forward(path, CONNS)
        inst = build_projective_instrument(cs, D, N, ETA, sigma_a2=0.0)
        blocks = evolve_code_blocks(cs, steps, D, N)
        proj = entanglement_fidelity_projective(inst, blocks)
        petz = build_petz_exact_dose(cs, steps, D, N)
        pz = entanglement_fidelity_petz(petz, cs, blocks)
        res["cells"][name] = {
            "gate": name.split("_")[0], "seed": int(name.split("_s")[1]),
            "projective_rate": proj["rate_1mFavg"],
            "petz_rate": pz["rate_1mFavg"],
            "projective_F_e": proj["F_e"], "petz_F_e": pz["F_e"],
            "n_sectors": inst["report"]["n_sectors"],
            "wall_s": round(time.time() - t0, 1)}
        with open(OUT, "w") as f:
            json.dump(res, f, indent=2)
        print(f"{name}: proj={proj['rate_1mFavg']:.4e} "
              f"petz={pz['rate_1mFavg']:.4e} "
              f"({res['cells'][name]['wall_s']}s)", flush=True)
    print("done")


if __name__ == "__main__":
    main()
