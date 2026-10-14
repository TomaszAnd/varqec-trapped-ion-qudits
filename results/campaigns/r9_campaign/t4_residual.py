#!/usr/bin/env python3
"""Residual vs near-optimal recovery (exact calibrated-channel cells for the
residual-ladder section).

Exact full-dose Petz + projective rates for the Table IV code set -- the
six n=5 codes spanning ~thirty orders of KL residual: central code,
hardware-basis L16, plateau-stopped (B_s202), converged (B_s202), deep
L28, [[5,1,3]]_Z3 -- channel v2 at s in {0.5, 1.0}. Question: does the
training residual order the near-optimal (Petz) recovery rate, while
decoupling from the discrete-benchmark LER?

Residual values are the published Table IV numbers (sources:
loss_ler_diag/t1b_results.json + t1c_results.json,
r1_campaign/t1_results.json, r1_campaign/t3_results.json).
Writes t4_results.json incrementally.
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
OUT = os.path.join(HERE, "t4_results.json")
BR = os.path.join(REPO, "results", "campaigns")

SIGMA_P2, SIGMA_A2_CAL = 0.073, 0.047
D, N = 3, 5
GP = [(i, j, 0, 1) for i in range(N) for j in range(i + 1, N)]
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]
S_GRID = (0.5, 1.0)

# (name, path, Table-IV residual, residual source)
CODES = [
    ("headline_A_L2_s100",
     os.path.join(BR, "r1_campaign", "t1_codes", "A_L2_s100.npz"),
     4.6, "TableIV/r1 t1_results"),
    ("hw_basis_L16",
     os.path.join(BR, "loss_ler_diag", "n5_hw_L16.npz"),
     2.90, "TableIV/loss_ler_diag t1c_results"),
    ("plateau_B_s202",
     os.path.join(BR, "r1_campaign", "t1_codes", "B_s202_plateau.npz"),
     0.449, "TableIV/r1 t1_results"),
    ("converged_B_s202",
     os.path.join(BR, "r1_campaign", "t1_codes", "B_s202_conv.npz"),
     0.360, "TableIV/r1 t1_results"),
    ("deep_L28",
     os.path.join(BR, "headline_p05_Lsweep", "n5_L28.npz"),
     1.95e-4, "TableIV/loss_ler_diag t1b_results"),
    ("five_one_three_Z3", None, 1e-30, "analytical (Table IV ~1e-30)"),
]


def main():
    res = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            res = json.load(f)
    res.setdefault("meta", {"d": D, "n": N, "s_grid": list(S_GRID),
                            "convention": "eta(s)=e^{-s*0.073}, "
                                          "sigma_a2(s)=s*0.047",
                            "metric": "rate = 1 - F_avg (channel infidelity)"})
    res.setdefault("cells", {})
    states = {}
    for name, path, resid, src in CODES:
        if path is None:
            states[name] = five_qudit_code_states(D)
        else:
            cs, d_, n_, K_ = encoder_forward(path, CONNS)
            assert (d_, n_) == (D, N)
            states[name] = cs
        print(f"loaded {name}", flush=True)

    for s in S_GRID:
        eta, sa2 = float(np.exp(-s * SIGMA_P2)), s * SIGMA_A2_CAL
        steps = build_dose_steps(N, D, GP, eta, sigma_a2=sa2)
        for name, path, resid, src in CODES:
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
            res["cells"][key] = {
                "code": name, "s": s, "eta": eta, "sigma_a2": sa2,
                "residual": resid, "residual_source": src,
                "projective_rate": proj["rate_1mFavg"],
                "petz_rate": pz["rate_1mFavg"],
                "wall_s": round(time.time() - t0, 1)}
            with open(OUT, "w") as f:
                json.dump(res, f, indent=2)
            print(f"{key}: resid={resid:g} proj={proj['rate_1mFavg']:.4e} "
                  f"petz={pz['rate_1mFavg']:.4e}", flush=True)

    # ordering summary: is the Petz rate monotone in residual?
    summary = {}
    for s in S_GRID:
        rows = sorted(
            ({"code": n_, "residual": r_,
              "projective_rate":
                  res["cells"][f"{n_}@s{s:.3f}"]["projective_rate"],
              "petz_rate": res["cells"][f"{n_}@s{s:.3f}"]["petz_rate"]}
             for n_, p_, r_, s_ in CODES),
            key=lambda x: -x["residual"])
        petz = [r["petz_rate"] for r in rows]
        proj = [r["projective_rate"] for r in rows]
        summary[f"s{s:.3f}"] = {
            "rows_by_descending_residual": rows,
            "petz_monotone_decreasing_in_residual_rank":
                bool(all(petz[i] >= petz[i + 1]
                         for i in range(len(petz) - 1))),
            "projective_monotone_decreasing_in_residual_rank":
                bool(all(proj[i] >= proj[i + 1]
                         for i in range(len(proj) - 1))),
        }
    res["summary"] = summary
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print("done")


if __name__ == "__main__":
    main()
