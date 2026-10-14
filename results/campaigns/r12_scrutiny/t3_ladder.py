#!/usr/bin/env python3
"""Residual ladder: crossover experiment on the full s grid.

Fills the missing exact v2 cells (s in {0.125, 0.25, 1.5}) for the four
Table IV/V ladder codes not covered by r9_campaign/t4_residual.py (which
ran s in {0.5, 1.0}): hw-basis L16, plateau B_s202, converged B_s202,
deep L28. The central code and [[5,1,3]] full-grid cells exist in
r9_campaign/t1_results.json; the channel-weighted depth ladder
A_L{2,4,8,16} x 4 seeds full-grid cells exist in
r11_gapfill/t5_sgrid_results.json.

Writes t3_ladder_results.json incrementally (resumable per cell).
The crossover analysis (merge + summary) is t3_analyze.py.
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
OUT = os.path.join(HERE, "t3_ladder_results.json")
BR = os.path.join(REPO, "results", "campaigns")

SIGMA_P2, SIGMA_A2_CAL = 0.073, 0.047
D, N = 3, 5
GP = [(i, j, 0, 1) for i in range(N) for j in range(i + 1, N)]
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]
S_GRID = (0.125, 0.25, 1.5)

# (name, path, Table-IV residual) — residuals as in r9 t4_residual.py
CODES = [
    ("hw_basis_L16",
     os.path.join(BR, "loss_ler_diag", "n5_hw_L16.npz"), 2.90),
    ("plateau_B_s202",
     os.path.join(BR, "r1_campaign", "t1_codes", "B_s202_plateau.npz"),
     0.449),
    ("converged_B_s202",
     os.path.join(BR, "r1_campaign", "t1_codes", "B_s202_conv.npz"), 0.360),
    ("deep_L28",
     os.path.join(BR, "headline_p05_Lsweep", "n5_L28.npz"), 1.95e-4),
]


def main():
    res = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            res = json.load(f)
    res.setdefault("meta", {"d": D, "n": N, "s_grid": list(S_GRID),
                            "convention": "eta(s)=e^{-s*0.073}, "
                                          "sigma_a2(s)=s*0.047",
                            "complements": "r9 t4 (s 0.5/1.0), r9 t1 "
                                           "(headline+513 full grid), r11 "
                                           "t5_sgrid (depth ladder full grid)",
                            "metric": "rate = 1 - F_avg (channel infidelity)"})
    res.setdefault("cells", {})
    states = {}
    for name, path, resid in CODES:
        cs, d_, n_, K_ = encoder_forward(path, CONNS)
        assert (d_, n_) == (D, N)
        states[name] = (cs, resid)
        print(f"loaded {name}: K={cs.shape[0]}", flush=True)

    for s in S_GRID:
        eta, sa2 = float(np.exp(-s * SIGMA_P2)), s * SIGMA_A2_CAL
        steps = build_dose_steps(N, D, GP, eta, sigma_a2=sa2)
        for name, (cs, resid) in states.items():
            key = f"{name}@v2_s{s:.3f}"
            if key in res["cells"]:
                continue
            t0 = time.time()
            inst = build_projective_instrument(cs, D, N, eta, sigma_a2=sa2)
            blocks = evolve_code_blocks(cs, steps, D, N)
            proj = entanglement_fidelity_projective(inst, blocks)
            petz = build_petz_exact_dose(cs, steps, D, N)
            pz = entanglement_fidelity_petz(petz, cs, blocks)
            res["cells"][key] = {
                "code": name, "residual": resid, "s": s, "eta": eta,
                "sigma_a2": sa2,
                "projective_rate": proj["rate_1mFavg"],
                "petz_rate": pz["rate_1mFavg"],
                "petz_leak_max": float(max(pz["recovery_leak_per_codeword"])),
                "wall_s": round(time.time() - t0, 1),
            }
            with open(OUT, "w") as f:
                json.dump(res, f, indent=2)
            print(f"{key}: proj={proj['rate_1mFavg']:.4e} "
                  f"petz={pz['rate_1mFavg']:.4e} "
                  f"({res['cells'][key]['wall_s']}s)", flush=True)
    print("done")


if __name__ == "__main__":
    main()
