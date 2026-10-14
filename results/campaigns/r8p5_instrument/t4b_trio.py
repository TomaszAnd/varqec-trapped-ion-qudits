#!/usr/bin/env python3
"""Distance trio on the exact density-matrix engine.

The three selected t2c codes (d2_L2_s100, d3_L2_s100, d4_L2_s102), n = 7,
v1 dephasing at eta in {0.93, 0.92}, exact engine + projective-MAP + Petz.
Writes t4b_results.json incrementally.
"""
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
OUT = os.path.join(HERE, "t4b_results.json")
T2C = os.path.join(REPO, "results", "campaigns", "r3_campaign",
                   "t2c_codes")
D, N = 3, 7
GP = [(i, j, 0, 1) for i in range(N) for j in range(i + 1, N)]
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]
CODES = {"d2": "d2_L2_s100.npz", "d3": "d3_L2_s100.npz",
         "d4": "d4_L2_s102.npz"}
ETAS = [0.93, 0.92]


def main():
    res = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            res = json.load(f)
    res.setdefault("meta", {"d": D, "n": N, "sigma_a2": 0.0,
                            "gate_steps": len(GP),
                            "codes": {k: os.path.join(T2C, v)
                                      for k, v in CODES.items()},
                            "metric": "rate = 1 - F_avg (channel infidelity)"})
    res.setdefault("cells", {})
    states = {}
    for arm, fname in CODES.items():
        states[arm], *_ = encoder_forward(os.path.join(T2C, fname), CONNS)
        print(f"loaded {arm}", flush=True)
    for eta in ETAS:
        steps = build_dose_steps(N, D, GP, eta, sigma_a2=0.0)
        for arm, cs in states.items():
            key = f"{arm}@eta{eta}"
            if key in res["cells"]:
                continue
            t0 = time.time()
            inst = build_projective_instrument(cs, D, N, eta, sigma_a2=0.0)
            blocks = evolve_code_blocks(cs, steps, D, N)
            proj = entanglement_fidelity_projective(inst, blocks)
            petz = build_petz_exact_dose(cs, steps, D, N)
            pz = entanglement_fidelity_petz(petz, cs, blocks)
            res["cells"][key] = {
                "arm": arm, "eta": eta,
                "projective_rate": proj["rate_1mFavg"],
                "petz_rate": pz["rate_1mFavg"],
                "projective_F_e": proj["F_e"], "petz_F_e": pz["F_e"],
                "petz_leak_max":
                    float(max(pz["recovery_leak_per_codeword"])),
                "n_sectors": inst["report"]["n_sectors"],
                "wall_s": round(time.time() - t0, 1)}
            with open(OUT, "w") as f:
                json.dump(res, f, indent=2)
            print(f"{key}: proj={proj['rate_1mFavg']:.4e} "
                  f"petz={pz['rate_1mFavg']:.4e} "
                  f"({res['cells'][key]['wall_s']}s)", flush=True)
    print("done")


if __name__ == "__main__":
    main()
