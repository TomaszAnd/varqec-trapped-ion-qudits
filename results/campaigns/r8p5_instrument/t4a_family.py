#!/usr/bin/env python3
"""((n,3,3))_3 n-family on the exact density-matrix engine.

Matched minimal-L family (the paper figure's code set), v1 dephasing at
eta in {0.9296 (calibrated), 0.92}: exact engine for n = 5, 6, 7;
validated sampled sector-averaged estimator (MODE A stochastic-phase
unraveling, exact for v1) for n = 8, 9, with a MODE B (Kraus-sampled)
cross-check at n = 8. Adaptive trajectory count: chunks until
SE <= 0.15 x rate or the per-ensemble cap. Writes t4a_results.json
incrementally (resumable per cell).

Memory: no density matrix is ever built at n >= 8; the biggest object is
the n = 9 stacked sector basis (~0.65 GB).
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
    sampled_entanglement_fidelity,
)
from src.simulation import (make_correlated_dephasing_noise_fn,
                            make_meth_mc_noise_fn)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "t4a_results.json")
BR = os.path.join(REPO, "results", "campaigns")

D = 3
ETAS = [0.9296, 0.92]
CODES = {
    5: os.path.join(BR, "campaign", "d3_n5_dist3_16L.npz"),
    6: os.path.join(BR, "r2_campaign", "t4_codes", "n6_L2_s102.npz"),
    7: os.path.join(BR, "r2_campaign", "t1a_codes", "d3_L2_s100.npz"),
    8: os.path.join(BR, "r2_campaign", "t4_codes", "n8_L2_s100.npz"),
    9: os.path.join(BR, "r2_campaign", "t4_codes", "n9_L2_s100.npz"),
}
CHUNK = 1000          # trajectories per ensemble per adaptive round
CAP = 30000           # per-ensemble cap
SE_TARGET_FRAC = 0.15


def sampled_cell(cs, d, n, eta, noise_maker, seed):
    """Adaptive sampled estimate: grow trajectories until the F_e SE is
    below SE_TARGET_FRAC x rate or CAP is reached."""
    gp = [(i, j, 0, 1) for i in range(n) for j in range(i + 1, n)]
    inst = build_projective_instrument(cs, d, n, eta, sigma_a2=0.0)
    nf = noise_maker(n, d, gp, eta)
    n_traj = 0
    est = None
    while True:
        n_traj = CHUNK if n_traj == 0 else min(2 * n_traj, CAP)
        t0 = time.time()
        est = sampled_entanglement_fidelity(inst, nf, n_traj, seed=seed)
        est["wall_s"] = round(time.time() - t0, 1)
        rate = max(est["rate_1mFavg"], 1e-12)
        se_rate = est["F_avg_se"]
        print(f"    n_traj={n_traj}: rate={rate:.4e} se={se_rate:.2e} "
              f"({est['wall_s']}s)", flush=True)
        if se_rate <= SE_TARGET_FRAC * rate or n_traj >= CAP:
            break
    est["n_sectors"] = inst["report"]["n_sectors"]
    return est


def main():
    res = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            res = json.load(f)
    res.setdefault("meta", {"d": D, "etas": ETAS, "sigma_a2": 0.0,
                            "codes": CODES,
                            "metric": "rate = 1 - F_avg (channel infidelity)",
                            "sampled_primary": "modeA stochastic phases "
                                               "(exact v1 unraveling)",
                            "se_target_frac": SE_TARGET_FRAC, "cap": CAP})
    res.setdefault("cells", {})

    def save():
        with open(OUT, "w") as f:
            json.dump(res, f, indent=2)

    for n, path in CODES.items():
        n = int(n)
        conns = [[i, j] for i in range(n) for j in range(i + 1, n)]
        cs, *_ = encoder_forward(path, conns)
        print(f"n={n}: loaded {os.path.basename(path)}", flush=True)
        for eta in ETAS:
            key = f"n{n}@eta{eta}"
            if key in res["cells"]:
                continue
            t0 = time.time()
            if n <= 7:
                gp = [(i, j, 0, 1) for i in range(n)
                      for j in range(i + 1, n)]
                steps = build_dose_steps(n, D, gp, eta, sigma_a2=0.0)
                inst = build_projective_instrument(cs, D, n, eta,
                                                   sigma_a2=0.0)
                blocks = evolve_code_blocks(cs, steps, D, n)
                proj = entanglement_fidelity_projective(inst, blocks)
                petz = build_petz_exact_dose(cs, steps, D, n)
                pz = entanglement_fidelity_petz(petz, cs, blocks)
                res["cells"][key] = {
                    "n": n, "eta": eta, "kind": "exact",
                    "projective_rate": proj["rate_1mFavg"],
                    "petz_rate": pz["rate_1mFavg"],
                    "wall_s": round(time.time() - t0, 1)}
            else:
                est = sampled_cell(cs, D, n, eta, make_meth_mc_noise_fn,
                                   seed=n * 100 + int(eta * 1e4))
                res["cells"][key] = {
                    "n": n, "eta": eta, "kind": "sampled_modeA",
                    "projective_rate": est["rate_1mFavg"],
                    "projective_rate_se": est["F_avg_se"],
                    "n_traj_per_ensemble": est["n_traj_per_ensemble"],
                    "wall_s": round(time.time() - t0, 1)}
            save()
            print(f"{key}: {json.dumps(res['cells'][key])}", flush=True)

    # MODE B cross-check at n = 8, calibrated eta
    key = "n8@eta0.9296_modeB_xcheck"
    if key not in res["cells"]:
        n, eta = 8, 0.9296
        conns = [[i, j] for i in range(n) for j in range(i + 1, n)]
        cs, *_ = encoder_forward(CODES[8] if 8 in CODES else CODES["8"],
                                 conns)
        t0 = time.time()
        gp = [(i, j, 0, 1) for i in range(n) for j in range(i + 1, n)]
        inst = build_projective_instrument(cs, D, n, eta, sigma_a2=0.0)

        def maker(nq, dq, gpq, etaq):
            return make_correlated_dephasing_noise_fn(nq, dq, gpq, etaq,
                                                      n_max=5)
        nf = maker(n, D, gp, eta)
        est = sampled_entanglement_fidelity(inst, nf, 3000, seed=4242)
        ref = res["cells"].get("n8@eta0.9296")
        z = None
        if ref is not None:
            z = (abs(est["rate_1mFavg"] - ref["projective_rate"])
                 / np.hypot(est["F_avg_se"],
                            ref.get("projective_rate_se", 0.0)))
        res["cells"][key] = {
            "n": n, "eta": eta, "kind": "sampled_modeB_xcheck",
            "projective_rate": est["rate_1mFavg"],
            "projective_rate_se": est["F_avg_se"],
            "z_vs_modeA": (round(float(z), 2) if z is not None else None),
            "wall_s": round(time.time() - t0, 1)}
        save()
        print(f"{key}: {json.dumps(res['cells'][key])}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
