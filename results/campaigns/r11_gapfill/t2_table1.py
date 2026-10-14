#!/usr/bin/env python3
"""Table I remeasure + exact physical-channel companions (Fig. 12 paired
cells).

(a) Re-measure every i.i.d. p-sweep cell of Table I under the current
locked protocol: same three codes (central code A_L2_s100; unweighted
full basis L=16 = campaign/d3_n5_dist3_16L.npz, the 'c_n5' row;
[[5,1,3]]_Z3), Table-I p grid, lookup decoder, shots 3e5 doubling to
>= 30 errors, cap 4.8e6, Wilson 95%, below-detection = Wilson UB. Fresh
seeds (hash-based, disjoint from the published fig6_panels stream -- the
diff is a protocol-reproducibility check, not a same-stream replay;
stated). Old cells (fig6_panels/measurements.json) are stored alongside
and the diff flags any cell whose new point estimate leaves the old
Wilson interval or vice versa.

(b) Exact physical-channel companion columns: for each p, the matched
strength s_match(p) solves I(eta(s)) = c_3 p on the unified per-gate
infidelity axis (r7_campaign/r7_t3_axis.py target-tier convention,
c_3 = 2/3, I(eta) = (3/4)[1 - (3 + 4 sqrt(eta) + 2 eta^2)/9], validated
against the 0.03459 anchor); exact channel-v2 projective-MAP + full-dose
Petz rates are computed at s_match for all three codes, so the table can
be printed either way (iid columns, physical columns, or both).

Writes t2_table1_results.json (resumable per cell). Feeds Table I and
Fig. 12.
Launch: cd results/campaigns/r11_gapfill
        nohup nice -n 19 python3 t2_table1.py >> t2_table1.log 2>&1 &
"""
import json
import math
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
from src.catalog import five_qudit_code_states
from src.decoders._common import encoder_forward
from src.decoders.exact_channel import build_dose_steps, evolve_code_blocks
from src.decoders.projective_decode import (
    build_petz_exact_dose, build_projective_instrument,
    entanglement_fidelity_petz, entanglement_fidelity_projective,
)
from src.errors import make_hardware_noise_fn, qudit_hardware_error_basis
from src.simulation import simulate_ler_with_correction_factored

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "t2_table1_results.json")
BR = os.path.join(REPO, "results", "campaigns")
MEAS = os.path.join(BR, "fig6_panels", "measurements.json")

D, N = 3, 5
SIGMA_P2, SIGMA_A2_CAL = 0.073, 0.047
P_GRID = (0.02, 0.03, 0.05, 0.08, 0.12, 0.15)
MIN_ERRORS, SHOTS_START, SHOTS_CAP = 30, 300_000, 4_800_000
C3 = 2.0 / 3.0
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]
GP = [(i, j, 0, 1) for i in range(N) for j in range(i + 1, N)]

CODES = [
    ("t1head", os.path.join(BR, "r1_campaign", "t1_codes", "A_L2_s100.npz")),
    ("c_n5", os.path.join(BR, "campaign", "d3_n5_dist3_16L.npz")),
    ("s513", None),
]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    dn = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / dn
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / dn
    return (max(0.0, c - h), c + h)


def I_eta_q3(eta):
    return 0.75 * (1.0 - (3.0 + 4.0 * math.sqrt(eta) + 2.0 * eta ** 2) / 9.0)


def I_of_s(s):
    return I_eta_q3(math.exp(-s * SIGMA_P2))


def s_match(p):
    """Solve I_of_s(s) = C3 * p by bisection (I monotone increasing in s)."""
    target = C3 * p
    lo, hi = 1e-4, 80.0
    assert I_of_s(lo) < target < I_of_s(hi), (p, target)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if I_of_s(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def iid_cell(cs, S, p, seed):
    n_shots = SHOTS_START
    while True:
        r = simulate_ler_with_correction_factored(
            cs, make_hardware_noise_fn(D, N, p), S, N, D,
            n_shots=n_shots, seed=seed)
        ler = float(r["logical_error_rate"])
        k = int(round(ler * n_shots))
        if k >= MIN_ERRORS or n_shots >= SHOTS_CAP:
            lo, hi = wilson(k, n_shots)
            return {"p": p, "ler": ler, "k": k, "n_shots": n_shots,
                    "wilson95": [float(lo), float(hi)],
                    "resolved": bool(k >= MIN_ERRORS)}
        n_shots = min(2 * n_shots, SHOTS_CAP)


def main():
    res = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            res = json.load(f)
    res.setdefault("meta", {
        "d": D, "n": N, "p_grid": list(P_GRID),
        "iid_protocol": "lookup decoder, shots 3e5 doubling to >=30 errors, "
                        "cap 4.8e6, Wilson95, fresh hash seeds (stated: "
                        "independent stream, protocol-reproducibility diff)",
        "companion": "exact channel-v2 proj-MAP + full-dose Petz at "
                     "s_match(p): I(eta(s)) = (2/3) p, r7_t3_axis target-tier "
                     "axis (anchor I(1)=0.03459)",
        "old_source": "fig6_panels/measurements.json (t1head/c_n5/s513)"})
    res.setdefault("iid_cells", {})
    res.setdefault("companion_cells", {})

    S = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]
    states = {}
    for name, path in CODES:
        if path is None:
            states[name] = np.asarray(five_qudit_code_states(D))
        else:
            cs, d_, n_, K_ = encoder_forward(path, CONNS)
            assert (d_, n_) == (D, N)
            states[name] = np.asarray(cs)
        print(f"loaded {name}", flush=True)

    # (a) iid remeasure
    for name, path in CODES:
        for p in P_GRID:
            key = f"{name}@p{p:.2f}"
            if key in res["iid_cells"]:
                continue
            t0 = time.time()
            cell = iid_cell(states[name], S, p,
                            seed=hash(("r11t2", name, p)) % (2 ** 31))
            cell.update({"code": name,
                         "wall_s": round(time.time() - t0, 1)})
            res["iid_cells"][key] = cell
            json.dump(res, open(OUT, "w"), indent=2)
            print(f"{key}: ler={cell['ler']:.3e} k={cell['k']} "
                  f"n={cell['n_shots']} ({cell['wall_s']}s)", flush=True)

    # (b) exact companions at matched per-gate infidelity
    for p in P_GRID:
        sm = s_match(p)
        eta, sa2 = float(np.exp(-sm * SIGMA_P2)), sm * SIGMA_A2_CAL
        steps = None
        for name, path in CODES:
            key = f"{name}@p{p:.2f}"
            if key in res["companion_cells"]:
                continue
            if steps is None:
                steps = build_dose_steps(N, D, GP, eta, sigma_a2=sa2)
            cs = states[name]
            t0 = time.time()
            inst = build_projective_instrument(cs, D, N, eta, sigma_a2=sa2)
            blocks = evolve_code_blocks(cs, steps, D, N)
            proj = entanglement_fidelity_projective(inst, blocks)
            petz = build_petz_exact_dose(cs, steps, D, N)
            pz = entanglement_fidelity_petz(petz, cs, blocks)
            res["companion_cells"][key] = {
                "code": name, "p": p, "s_match": sm, "eta": eta,
                "sigma_a2": sa2, "I_axis": I_of_s(sm),
                "projective_rate": proj["rate_1mFavg"],
                "petz_rate": pz["rate_1mFavg"],
                "wall_s": round(time.time() - t0, 1)}
            json.dump(res, open(OUT, "w"), indent=2)
            print(f"companion {key}: s={sm:.3f} "
                  f"proj={proj['rate_1mFavg']:.4e} "
                  f"petz={pz['rate_1mFavg']:.4e}", flush=True)

    # (c) diff vs the published cells
    with open(MEAS) as f:
        meas = json.load(f)
    diff = []
    for name, path in CODES:
        pts = {round(pt["p"], 4): pt for pt in meas["codes"][name]["points"]}
        for p in P_GRID:
            old = pts.get(round(p, 4))
            new = res["iid_cells"][f"{name}@p{p:.2f}"]
            if old is None:
                continue
            old_lo, old_hi = old["wilson_lo"], old["wilson_hi"]
            new_lo, new_hi = new["wilson95"]
            flag = bool(new["ler"] < old_lo or new["ler"] > old_hi or
                        old["ler"] < new_lo or old["ler"] > new_hi)
            diff.append({
                "code": name, "p": p,
                "old_ler": old["ler"], "old_k": old.get("k"),
                "old_n": old.get("shots"),
                "old_wilson95": [old_lo, old_hi],
                "new_ler": new["ler"], "new_k": new["k"],
                "new_n": new["n_shots"], "new_wilson95": [new_lo, new_hi],
                "moved_beyond_interval": flag})
    res["diff_vs_published"] = diff
    res["diff_summary"] = {
        "n_cells": len(diff),
        "n_flagged": sum(1 for d in diff if d["moved_beyond_interval"]),
        "flagged": [(d["code"], d["p"]) for d in diff
                    if d["moved_beyond_interval"]]}
    json.dump(res, open(OUT, "w"), indent=2)
    print("diff summary:", json.dumps(res["diff_summary"]), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
