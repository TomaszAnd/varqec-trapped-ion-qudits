#!/usr/bin/env python3
"""Gate-study bars, channel v2, s in {0.5, 1.0} (exact calibrated-channel
cells).

All 60 gate-study codes of r5_campaign (ms/zz/ls/csum/cex x seeds
100-111), n=6, exact engine + projective-MAP + Petz. Per-gate median +
range under both instruments in the summary. Writes t2c_results.json
incrementally.
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
OUT = os.path.join(HERE, "t2c_results.json")
CODES_DIR = os.path.join(REPO, "results", "campaigns",
                         "r5_campaign", "t1_codes")

SIGMA_P2, SIGMA_A2_CAL = 0.073, 0.047
D, N = 3, 6
GP = [(i, j, 0, 1) for i in range(N) for j in range(i + 1, N)]
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]
S_GRID = (0.5, 1.0)


def main():
    res = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            res = json.load(f)
    res.setdefault("meta", {"d": D, "n": N, "gate_steps": len(GP),
                            "s_grid": list(S_GRID),
                            "convention": "eta(s)=e^{-s*0.073}, "
                                          "sigma_a2(s)=s*0.047",
                            "metric": "rate = 1 - F_avg (channel infidelity)"})
    res.setdefault("cells", {})
    paths = sorted(glob.glob(os.path.join(CODES_DIR, "*.npz")))
    print(f"{len(paths)} codes", flush=True)
    for s in S_GRID:
        eta, sa2 = float(np.exp(-s * SIGMA_P2)), s * SIGMA_A2_CAL
        steps = build_dose_steps(N, D, GP, eta, sigma_a2=sa2)
        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            key = f"{name}@s{s:.3f}"
            if key in res["cells"]:
                continue
            t0 = time.time()
            cs, *_ = encoder_forward(path, CONNS)
            inst = build_projective_instrument(cs, D, N, eta, sigma_a2=sa2)
            blocks = evolve_code_blocks(cs, steps, D, N)
            proj = entanglement_fidelity_projective(inst, blocks)
            petz = build_petz_exact_dose(cs, steps, D, N)
            pz = entanglement_fidelity_petz(petz, cs, blocks)
            res["cells"][key] = {
                "gate": name.split("_")[0],
                "seed": int(name.split("_s")[1]),
                "s": s, "eta": eta, "sigma_a2": sa2,
                "projective_rate": proj["rate_1mFavg"],
                "petz_rate": pz["rate_1mFavg"],
                "wall_s": round(time.time() - t0, 1)}
            with open(OUT, "w") as f:
                json.dump(res, f, indent=2)
            print(f"{key}: proj={proj['rate_1mFavg']:.4e} "
                  f"petz={pz['rate_1mFavg']:.4e} "
                  f"({res['cells'][key]['wall_s']}s)", flush=True)

    # per-gate median + range, both instruments, per strength
    summary = {}
    for s in S_GRID:
        row = {}
        gates = sorted({c["gate"] for c in res["cells"].values()})
        for g in gates:
            cells = [c for k, c in res["cells"].items()
                     if c["gate"] == g and c["s"] == s]
            row[g] = {}
            for kind in ("projective_rate", "petz_rate"):
                v = np.array([c[kind] for c in cells])
                row[g][kind] = {"median": float(np.median(v)),
                                "min": float(v.min()),
                                "max": float(v.max()),
                                "n_seeds": int(v.size)}
        summary[f"s{s:.3f}"] = row
    res["summary"] = summary
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print("done")


if __name__ == "__main__":
    main()
