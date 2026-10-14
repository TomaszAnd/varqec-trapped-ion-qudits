#!/usr/bin/env python3
"""q-dimension exact correlated cells.

((5,3,3))_4 and ((5,3,3))_5 panel codes (r3_campaign/t2b_codes, reused by
the q panel) + the q=3 central code, channel v2, full strength grid,
exact engine (dims 1024 / 3125; peak RSS is printed per cell -- q=4 runs
first as the memory canary). Projective-MAP + Petz. Alongside each cell
the per-gate infidelity axis values I_q(s) are stored using the
r7_campaign/r7_t3_axis.py role convention (role_infid on the role
dephasing Kraus at eta(s); target tier = the canonical per-gate per-qudit
number, stated in r7_t3_results.json).

Writes t2d_results.json incrementally (resumable per cell). Feeds the
q panel of the paper.
"""
import json
import os
import resource
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

import numpy as np

import src.jax_backend  # noqa: F401  (x64 first)
from src.correlated_noise import (control_qudit_kraus, spectator_qudit_kraus,
                                  target_qudit_kraus)
from src.decoders._common import encoder_forward
from src.decoders.exact_channel import build_dose_steps, evolve_code_blocks
from src.decoders.projective_decode import (
    build_petz_exact_dose, build_projective_instrument,
    entanglement_fidelity_petz, entanglement_fidelity_projective,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "t2d_results.json")
BR = os.path.join(REPO, "results", "campaigns")

SIGMA_P2, SIGMA_A2_CAL = 0.073, 0.047
N = 5
GP = [(i, j, 0, 1) for i in range(N) for j in range(i + 1, N)]
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]
S_GRID = (0.125, 0.25, 0.5, 1.0, 1.5)

CODES = [  # q=4 first: the RSS canary before any q=5 allocation
    ("q4_L2_s100", 4, os.path.join(BR, "r3_campaign", "t2b_codes",
                                   "q4_L2_s100.npz")),
    ("q5_L2_s100", 5, os.path.join(BR, "r3_campaign", "t2b_codes",
                                   "q5_L2_s100.npz")),
    ("q3_headline_A_L2_s100", 3, os.path.join(BR, "r1_campaign", "t1_codes",
                                              "A_L2_s100.npz")),
]


def role_infid(kraus_diags, q):
    """Convention of r7_campaign/r7_t3_axis.py: average gate infidelity of a
    diagonal-Kraus role channel."""
    s = sum(abs(np.sum(dg)) ** 2 for dg in kraus_diags)
    return float(1 - (s + q) / (q ** 2 + q))


def axis_values(q, eta):
    return {
        "control": role_infid(control_qudit_kraus(q, 0, eta, 12), q),
        "target": role_infid(target_qudit_kraus(q, 0, 1, eta, 12), q),
        "spectator": role_infid(spectator_qudit_kraus(q, eta, 12), q),
        "canonical": "target",
    }


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30


def main():
    res = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            res = json.load(f)
    res.setdefault("meta", {"n": N, "gate_steps": len(GP),
                            "s_grid": list(S_GRID),
                            "convention": "eta(s)=e^{-s*0.073}, "
                                          "sigma_a2(s)=s*0.047",
                            "axis": "I_q(s) = role dephasing avg gate "
                                    "infidelity at eta(s), r7_t3_axis "
                                    "convention, target tier canonical",
                            "metric": "rate = 1 - F_avg (channel infidelity)"})
    res.setdefault("cells", {})
    for name, q, path in CODES:
        cs, d_, n_, K_ = encoder_forward(path, CONNS)
        assert (d_, n_) == (q, N), (d_, n_)
        print(f"loaded {name}: q={q} K={K_}", flush=True)
        for s in S_GRID:
            key = f"{name}@s{s:.3f}"
            if key in res["cells"]:
                continue
            eta, sa2 = float(np.exp(-s * SIGMA_P2)), s * SIGMA_A2_CAL
            t0 = time.time()
            steps = build_dose_steps(N, q, GP, eta, sigma_a2=sa2)
            inst = build_projective_instrument(cs, q, N, eta, sigma_a2=sa2)
            blocks = evolve_code_blocks(cs, steps, q, N)
            proj = entanglement_fidelity_projective(inst, blocks)
            petz = build_petz_exact_dose(cs, steps, q, N)
            pz = entanglement_fidelity_petz(petz, cs, blocks)
            res["cells"][key] = {
                "code": name, "q": q, "s": s, "eta": eta, "sigma_a2": sa2,
                "projective_rate": proj["rate_1mFavg"],
                "petz_rate": pz["rate_1mFavg"],
                "petz_leak_max":
                    float(max(pz["recovery_leak_per_codeword"])),
                "n_sectors": inst["report"]["n_sectors"],
                "claimed_dim_R": inst["report"]["claimed_dim_R"],
                "hilbert_dim": inst["report"]["hilbert_dim"],
                "I_q_axis": axis_values(q, eta),
                "peak_rss_gb": round(rss_gb(), 2),
                "wall_s": round(time.time() - t0, 1)}
            with open(OUT, "w") as f:
                json.dump(res, f, indent=2)
            print(f"{key}: proj={proj['rate_1mFavg']:.4e} "
                  f"petz={pz['rate_1mFavg']:.4e} "
                  f"rss={res['cells'][key]['peak_rss_gb']}GB "
                  f"({res['cells'][key]['wall_s']}s)", flush=True)
    print("done")


if __name__ == "__main__":
    main()
