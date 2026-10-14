#!/usr/bin/env python3
"""Distance trio single-round, channel v2, full strength grid (exact
calibrated-channel cells).

d2/d3/d4 selected codes at n=7, channel v2, s in
{0.125, 0.25, 0.5, 1.0, 1.5}, exact engine + projective-MAP + Petz.
The summary block reports the monotone-in-distance margins per s under
both instruments (does the relative margin grow as s falls?).
Writes t2b_results.json incrementally (resumable per cell).
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
OUT = os.path.join(HERE, "t2b_results.json")
T2C = os.path.join(REPO, "results", "campaigns", "r3_campaign",
                   "t2c_codes")

SIGMA_P2, SIGMA_A2_CAL = 0.073, 0.047
D, N = 3, 7
GP = [(i, j, 0, 1) for i in range(N) for j in range(i + 1, N)]
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]
CODES = {"d2": "d2_L2_s100.npz", "d3": "d3_L2_s100.npz",
         "d4": "d4_L2_s102.npz"}
S_GRID = (0.125, 0.25, 0.5, 1.0, 1.5)


def main():
    res = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            res = json.load(f)
    res.setdefault("meta", {"d": D, "n": N, "gate_steps": len(GP),
                            "s_grid": list(S_GRID),
                            "convention": "eta(s)=e^{-s*0.073}, "
                                          "sigma_a2(s)=s*0.047",
                            "codes": {k: os.path.join(T2C, v)
                                      for k, v in CODES.items()},
                            "metric": "rate = 1 - F_avg (channel infidelity)"})
    res.setdefault("cells", {})
    states = {}
    for arm, fname in CODES.items():
        states[arm], *_ = encoder_forward(os.path.join(T2C, fname), CONNS)
        print(f"loaded {arm}", flush=True)
    for s in S_GRID:
        eta, sa2 = float(np.exp(-s * SIGMA_P2)), s * SIGMA_A2_CAL
        steps = None
        for arm, cs in states.items():
            key = f"{arm}@s{s:.3f}"
            if key in res["cells"]:
                continue
            if steps is None:
                steps = build_dose_steps(N, D, GP, eta, sigma_a2=sa2)
            t0 = time.time()
            inst = build_projective_instrument(cs, D, N, eta, sigma_a2=sa2)
            blocks = evolve_code_blocks(cs, steps, D, N)
            proj = entanglement_fidelity_projective(inst, blocks)
            petz = build_petz_exact_dose(cs, steps, D, N)
            pz = entanglement_fidelity_petz(petz, cs, blocks)
            res["cells"][key] = {
                "arm": arm, "s": s, "eta": eta, "sigma_a2": sa2,
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

    # margins per strength, both instruments
    summary = {}
    for s in S_GRID:
        row = {}
        for kind in ("projective_rate", "petz_rate"):
            r = {a: res["cells"][f"{a}@s{s:.3f}"][kind] for a in CODES}
            row[kind] = {
                **r,
                "margin_d2_over_d4": r["d2"] / r["d4"],
                "margin_d3_over_d4": r["d3"] / r["d4"],
                "rel_margin_d2_vs_d4": (r["d2"] - r["d4"]) / r["d4"],
                "ordering": sorted(r, key=r.get),
            }
        summary[f"s{s:.3f}"] = row
    res["summary"] = summary
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print("done")


if __name__ == "__main__":
    main()
