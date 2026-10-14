#!/usr/bin/env python3
"""Full-grid depth/pruning channel panels: depth codes + pruning ladder on
the FULL strength grid.

Extends w5_t0 (s in {0.5, 1.0}) with s in {0.125, 0.25, 1.5} for the same
20 codes: depth A_L{2,4,8,16}_s{100..103} + pruning 40/16/12/8. Exact
engine, channel v2, projective-MAP + full-dose Petz, rate = 1 - F_avg.
Existing w5_t0 cells are imported verbatim with provenance; the summary
covers the full 5-point grid so the figure needs no {0.5, 1.0} picks.

Writes t5_sgrid_results.json (resumable per cell). Feeds the depth and
pruning channel panels of the paper.
Launch: cd results/campaigns/r11_gapfill
        nohup nice -n 19 python3 t5_sgrid.py >> t5_sgrid.log 2>&1 &
"""
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

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
OUT = os.path.join(HERE, "t5_sgrid_results.json")
BR = os.path.join(REPO, "results", "campaigns")
W5_T0 = os.path.join(BR, "w5_t0", "t0_results.json")

SIGMA_P2, SIGMA_A2_CAL = 0.073, 0.047
D, N = 3, 5
GP = [(i, j, 0, 1) for i in range(N) for j in range(i + 1, N)]
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]
S_FULL = (0.125, 0.25, 0.5, 1.0, 1.5)
S_NEW = (0.125, 0.25, 1.5)

DEPTH = [(f"depth_L{L}_s{seed}",
          os.path.join(BR, "r1_campaign", "t1_codes", f"A_L{L}_s{seed}.npz"),
          {"L": L, "seed": seed})
         for L in (2, 4, 8, 16) for seed in (100, 101, 102, 103)]
PRUNE = [
    ("prune_gates40",
     os.path.join(BR, "r1_campaign", "t1_codes", "A_L2_s100.npz"),
     {"gates": 40}),
    ("prune_gates16",
     os.path.join(BR, "r2_campaign", "t3_codes", "abl_prune60.npz"),
     {"gates": 16}),
    ("prune_gates12",
     os.path.join(BR, "r2_campaign", "t3_codes", "abl_prune70.npz"),
     {"gates": 12}),
    ("prune_gates8",
     os.path.join(BR, "r2_campaign", "t3_codes", "abl_prune80.npz"),
     {"gates": 8}),
]


def main():
    res = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            res = json.load(f)
    res.setdefault("meta", {
        "d": D, "n": N, "s_grid_full": list(S_FULL),
        "convention": "eta(s)=e^{-s*0.073}, sigma_a2(s)=s*0.047",
        "channel": "v2, exact engine",
        "metric": "rate = 1 - F_avg (channel infidelity)",
        "instruments": "projective-MAP + full-dose Petz",
        "provenance": "s in {0.5,1.0} cells imported from w5_t0/"
                      "t0_results.json verbatim"})
    res.setdefault("cells", {})

    with open(W5_T0) as f:
        w5 = json.load(f)
    for key, cell in w5["cells"].items():
        if key not in res["cells"]:
            cell = dict(cell)
            cell["reused_from"] = "w5_t0/t0_results.json"
            res["cells"][key] = cell
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"imported {len(w5['cells'])} w5_t0 cells", flush=True)

    jobs = DEPTH + PRUNE
    states = {}
    for name, path, meta in jobs:
        cs, d_, n_, K_ = encoder_forward(path, CONNS)
        assert (d_, n_) == (D, N)
        states[name] = cs

    for s in S_NEW:
        eta, sa2 = float(np.exp(-s * SIGMA_P2)), s * SIGMA_A2_CAL
        steps = build_dose_steps(N, D, GP, eta, sigma_a2=sa2)
        for name, path, meta in jobs:
            key = f"{name}@s{s:.3f}"
            if key in res["cells"]:
                continue
            cs = states[name]
            t0 = time.time()
            inst = build_projective_instrument(cs, D, N, eta, sigma_a2=sa2)
            blocks = evolve_code_blocks(cs, steps, D, N)
            proj = entanglement_fidelity_projective(inst, blocks)
            petz = build_petz_exact_dose(cs, steps, D, N)
            pz = entanglement_fidelity_petz(petz, cs, blocks)
            cell = {"code": name, "s": s, "eta": eta, "sigma_a2": sa2,
                    "path": os.path.relpath(path, REPO),
                    "projective_rate": proj["rate_1mFavg"],
                    "petz_rate": pz["rate_1mFavg"],
                    "wall_s": round(time.time() - t0, 1)}
            cell.update(meta)
            res["cells"][key] = cell
            json.dump(res, open(OUT, "w"), indent=2)
            print(f"{key}: proj={proj['rate_1mFavg']:.4e} "
                  f"petz={pz['rate_1mFavg']:.4e} ({cell['wall_s']}s)",
                  flush=True)

    # full-grid summaries
    summary = {"depth": {}, "prune": {}}
    for s in S_FULL:
        sk = f"s{s:.3f}"
        per_L = {}
        for L in (2, 4, 8, 16):
            pj = [res["cells"][f"depth_L{L}_s{seed}@{sk}"]["projective_rate"]
                  for seed in (100, 101, 102, 103)]
            pz = [res["cells"][f"depth_L{L}_s{seed}@{sk}"]["petz_rate"]
                  for seed in (100, 101, 102, 103)]
            per_L[f"L{L}"] = {
                "proj_median": float(np.median(pj)),
                "proj_range": [float(min(pj)), float(max(pj))],
                "petz_median": float(np.median(pz)),
                "petz_range": [float(min(pz)), float(max(pz))]}
        med_pj = [per_L[f"L{L}"]["proj_median"] for L in (2, 4, 8, 16)]
        med_pz = [per_L[f"L{L}"]["petz_median"] for L in (2, 4, 8, 16)]
        summary["depth"][sk] = {
            "per_L": per_L,
            "proj_median_monotone_nonincreasing_in_L":
                bool(all(med_pj[i] >= med_pj[i + 1]
                         for i in range(len(med_pj) - 1))),
            "petz_median_monotone_nonincreasing_in_L":
                bool(all(med_pz[i] >= med_pz[i + 1]
                         for i in range(len(med_pz) - 1)))}
        summary["prune"][sk] = {
            f"gates{g}": {
                "projective_rate":
                    res["cells"][f"prune_gates{g}@{sk}"]["projective_rate"],
                "petz_rate":
                    res["cells"][f"prune_gates{g}@{sk}"]["petz_rate"]}
            for g in (40, 16, 12, 8)}
    res["summary"] = summary
    json.dump(res, open(OUT, "w"), indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
