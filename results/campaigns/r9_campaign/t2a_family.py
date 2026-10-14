#!/usr/bin/env python3
"""((n,3,3))_3 family, channel v2, full strength grid (exact
calibrated-channel cells).

The paper-figure code set (r8p5_instrument/t4a lineage): n5 deep-16L,
n6_L2_s102, n7 = d3_L2_s100, n8_L2_s100, n9_L2_s100. Exact engine at
n <= 7 (the n=7 cells are REUSED from t2b_results.json d3 -- same code,
same setting, recorded with provenance); validated sampled
sector-averaged estimator (MODE A, v2) at n >= 8 with SE target <= 0.1 x
the smallest adjacent-n rate gap at that strength (adaptive; cap stated
when hit), and a MODE B cross-check at n=8, s=1.0.

Writes t2a_results.json incrementally (resumable per cell). Feeds the
register-size family figures of the paper.
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
from src.simulation import make_channel_v2_noise_fn, make_meth_mc_noise_fn

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "t2a_results.json")
T2B = os.path.join(HERE, "t2b_results.json")
BR = os.path.join(REPO, "results", "campaigns")

SIGMA_P2, SIGMA_A2_CAL = 0.073, 0.047
D = 3
S_GRID = (0.125, 0.25, 0.5, 1.0, 1.5)
CODES = {
    5: os.path.join(BR, "campaign", "d3_n5_dist3_16L.npz"),
    6: os.path.join(BR, "r2_campaign", "t4_codes", "n6_L2_s102.npz"),
    7: os.path.join(BR, "r2_campaign", "t1a_codes", "d3_L2_s100.npz"),
    8: os.path.join(BR, "r2_campaign", "t4_codes", "n8_L2_s100.npz"),
    9: os.path.join(BR, "r2_campaign", "t4_codes", "n9_L2_s100.npz"),
}
CHUNK = 1000
CAP = 32000


def sampled_cell(cs, n, eta, sa2, noise_maker, seed, se_target):
    gp = [(i, j, 0, 1) for i in range(n) for j in range(i + 1, n)]
    inst = build_projective_instrument(cs, D, n, eta, sigma_a2=sa2)
    nf = noise_maker(n, D, gp, eta, sigma_a2=sa2)
    n_traj = 0
    est = None
    while True:
        n_traj = CHUNK if n_traj == 0 else min(2 * n_traj, CAP)
        t0 = time.time()
        est = sampled_entanglement_fidelity(inst, nf, n_traj, seed=seed)
        est["wall_s"] = round(time.time() - t0, 1)
        print(f"    n_traj={n_traj}: rate={est['rate_1mFavg']:.4e} "
              f"se={est['F_avg_se']:.2e} target={se_target:.2e} "
              f"({est['wall_s']}s)", flush=True)
        if est["F_avg_se"] <= se_target or n_traj >= CAP:
            break
    est["n_sectors"] = inst["report"]["n_sectors"]
    est["cap_hit"] = bool(est["F_avg_se"] > se_target)
    return est


def main():
    res = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            res = json.load(f)
    res.setdefault("meta", {
        "d": D, "s_grid": list(S_GRID), "codes": CODES,
        "convention": "eta(s)=e^{-s*0.073}, sigma_a2(s)=s*0.047",
        "sampled": "MODE A (meth MC trajectory) sector-averaged "
                   "estimator, SE target 0.1 x smallest adjacent-n gap, "
                   "cap %d/ensemble" % CAP,
        "n7_provenance": "reused from t2b_results.json d3 cells "
                         "(same code d3_L2_s100, same settings)",
        "metric": "rate = 1 - F_avg (channel infidelity)"})
    res.setdefault("cells", {})

    def save():
        with open(OUT, "w") as f:
            json.dump(res, f, indent=2)

    # n=7: import the t2b d3 cells
    with open(T2B) as f:
        t2b = json.load(f)
    for s in S_GRID:
        key = f"n7@s{s:.3f}"
        if key not in res["cells"]:
            c = t2b["cells"][f"d3@s{s:.3f}"]
            res["cells"][key] = {
                "n": 7, "s": s, "eta": c["eta"], "sigma_a2": c["sigma_a2"],
                "kind": "exact (reused from t2b d3)",
                "projective_rate": c["projective_rate"],
                "petz_rate": c["petz_rate"]}
    save()

    # exact n = 5, 6
    for n in (5, 6):
        conns = [[i, j] for i in range(n) for j in range(i + 1, n)]
        cs, *_ = encoder_forward(CODES[n], conns)
        for s in S_GRID:
            key = f"n{n}@s{s:.3f}"
            if key in res["cells"]:
                continue
            eta, sa2 = float(np.exp(-s * SIGMA_P2)), s * SIGMA_A2_CAL
            gp = [(i, j, 0, 1) for i in range(n) for j in range(i + 1, n)]
            t0 = time.time()
            steps = build_dose_steps(n, D, gp, eta, sigma_a2=sa2)
            inst = build_projective_instrument(cs, D, n, eta, sigma_a2=sa2)
            blocks = evolve_code_blocks(cs, steps, D, n)
            proj = entanglement_fidelity_projective(inst, blocks)
            petz = build_petz_exact_dose(cs, steps, D, n)
            pz = entanglement_fidelity_petz(petz, cs, blocks)
            res["cells"][key] = {
                "n": n, "s": s, "eta": eta, "sigma_a2": sa2, "kind": "exact",
                "projective_rate": proj["rate_1mFavg"],
                "petz_rate": pz["rate_1mFavg"],
                "wall_s": round(time.time() - t0, 1)}
            save()
            print(f"{key}: {json.dumps(res['cells'][key])}", flush=True)

    # sampled n = 8, 9 (MODE A, v2)
    for n in (8, 9):
        conns = [[i, j] for i in range(n) for j in range(i + 1, n)]
        cs, *_ = encoder_forward(CODES[n], conns)
        print(f"n={n}: loaded", flush=True)
        for s in S_GRID:
            key = f"n{n}@s{s:.3f}"
            if key in res["cells"]:
                continue
            eta, sa2 = float(np.exp(-s * SIGMA_P2)), s * SIGMA_A2_CAL
            below = res["cells"][f"n{n-1}@s{s:.3f}"]["projective_rate"]
            # provisional gap: |rate_n - rate_{n-1}|, refined adaptively.
            # first estimate with a loose target, then tighten.
            t0 = time.time()
            se_target = 1e-3
            est = sampled_cell(cs, n, eta, sa2, make_meth_mc_noise_fn,
                               seed=n * 1000 + int(s * 1000), se_target=1e9)
            gap = abs(est["rate_1mFavg"] - below)
            se_target = 0.1 * gap
            n_traj = est["n_traj_per_ensemble"]
            while est["F_avg_se"] > se_target and n_traj < CAP:
                n_traj = min(2 * n_traj, CAP)
                est = sampled_entanglement_fidelity(
                    build_projective_instrument(cs, D, n, eta, sigma_a2=sa2),
                    make_meth_mc_noise_fn(
                        n, D, [(i, j, 0, 1) for i in range(n)
                               for j in range(i + 1, n)], eta,
                        sigma_a2=sa2),
                    n_traj, seed=n * 1000 + int(s * 1000))
                gap = abs(est["rate_1mFavg"] - below)
                se_target = 0.1 * gap
                print(f"    tighten n_traj={n_traj}: "
                      f"rate={est['rate_1mFavg']:.4e} "
                      f"se={est['F_avg_se']:.2e} target={se_target:.2e}",
                      flush=True)
            res["cells"][key] = {
                "n": n, "s": s, "eta": eta, "sigma_a2": sa2,
                "kind": "sampled_modeA_v2",
                "projective_rate": est["rate_1mFavg"],
                "projective_rate_se": est["F_avg_se"],
                "n_traj_per_ensemble": est["n_traj_per_ensemble"],
                "adjacent_gap": gap, "se_target": se_target,
                "cap_hit": bool(est["F_avg_se"] > se_target),
                "wall_s": round(time.time() - t0, 1)}
            save()
            print(f"{key}: {json.dumps(res['cells'][key])}", flush=True)

    # MODE B cross-check, n=8, s=1.0
    key = "n8@s1.000_modeB_xcheck"
    if key not in res["cells"]:
        n, s = 8, 1.0
        eta, sa2 = float(np.exp(-s * SIGMA_P2)), s * SIGMA_A2_CAL
        conns = [[i, j] for i in range(n) for j in range(i + 1, n)]
        cs, *_ = encoder_forward(CODES[n], conns)
        gp = [(i, j, 0, 1) for i in range(n) for j in range(i + 1, n)]
        t0 = time.time()
        inst = build_projective_instrument(cs, D, n, eta, sigma_a2=sa2)
        nf = make_channel_v2_noise_fn(n, D, gp, eta, sigma_a2=sa2)
        est = sampled_entanglement_fidelity(inst, nf, 3000, seed=4242)
        ref = res["cells"]["n8@s1.000"]
        z = (abs(est["rate_1mFavg"] - ref["projective_rate"])
             / np.hypot(est["F_avg_se"], ref["projective_rate_se"]))
        res["cells"][key] = {
            "n": n, "s": s, "kind": "sampled_modeB_xcheck",
            "projective_rate": est["rate_1mFavg"],
            "projective_rate_se": est["F_avg_se"],
            "z_vs_modeA": round(float(z), 2),
            "wall_s": round(time.time() - t0, 1)}
        save()
        print(f"{key}: {json.dumps(res['cells'][key])}", flush=True)

    # monotone check
    mono = {}
    for s in S_GRID:
        rates = [res["cells"][f"n{n}@s{s:.3f}"]["projective_rate"]
                 for n in (5, 6, 7, 8, 9)]
        mono[f"s{s:.3f}"] = {
            "projective_rates_n5_to_n9": rates,
            "monotone_rise_from_n5": bool(all(rates[i] < rates[i + 1]
                                              for i in range(4)))}
    res["summary"] = mono
    save()
    print("done")


if __name__ == "__main__":
    main()
