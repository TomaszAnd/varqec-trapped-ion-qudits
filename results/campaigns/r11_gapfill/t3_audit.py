#!/usr/bin/env python3
"""Petz crossing audit (register family on the channel) and the register
family at L = 2 (Fig. 3).

Observation under audit: on the corr family panel the PETZ curves cross --
at weak noise the ordering is n6 < n7 < n5 (the n=5 member ABOVE both),
at strong noise n5 < n6 < n7. The family's n=5 member on the channel is
the deep-trained L=16 code (campaign/d3_n5_dist3_16L.npz); n6/n7 are L=2.
The projective curves are monotone in n everywhere (pinned booleans).

Audit steps:
(a) recompute the suspect exact cells independently in this process
    (deep-16L n5 at s in {0.125, 0.25}; central code L=2 n5 at s = 0.125;
    d3 n7 at s = 0.25) and compare to the committed r9_campaign values;
(b) Petz regularization sensitivity: the pseudo-inverse cutoff
    (build_petz_exact_dose rcond, default 1e-12 relative) swept one order
    each way (1e-11 / 1e-13) at two crossing cells -- deep-16L n5@s0.25
    and d3 n7@s0.25 (the n5 x n7 crossing sits between s=0.125 and 0.25);
    eigendecomposition computed once per cell, only the kept-eigenvalue
    set varies;
(c) mechanism + significance numbers: per-n Petz penalty of the deep n5
    member vs the L=2 central code across the s grid (from committed
    r9_campaign t1 + t2a), per-n gate-step dose counts, and the n>=8
    sampled-SE margins (projective only -- no Petz exists at n >= 8;
    stated).

Also writes the companion record t1a_family_L2.json: the family panel
with the n=5 member replaced by the L=2 central code (no load-bearing
L=16), cells copied from committed r9_campaign JSONs with provenance,
and the monotone-in-n booleans recomputed for BOTH instruments on the
full s grid. Feeds Fig. 3.

Writes t3_audit_results.json and t1a_family_L2.json.
Run (foreground-ish, ~15 min dominated by the n=7 cell):
  cd results/campaigns/r11_gapfill
  nohup nice -n 19 python3 t3_audit.py >> t3_audit.log 2>&1 &
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
from src.decoders.exact_channel import (apply_dose, build_dose_steps,
                                        evolve_code_blocks)
from src.decoders.projective_decode import (
    build_petz_exact_dose, build_projective_instrument,
    entanglement_fidelity_petz, entanglement_fidelity_projective,
    favg_from_fe,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "t3_audit_results.json")
BR = os.path.join(REPO, "results", "campaigns")
R9 = os.path.join(BR, "r9_campaign")

SIGMA_P2, SIGMA_A2_CAL = 0.073, 0.047
D = 3

DEEP16 = os.path.join(BR, "campaign", "d3_n5_dist3_16L.npz")
HEADLINE = os.path.join(BR, "r1_campaign", "t1_codes", "A_L2_s100.npz")
D3N7 = os.path.join(BR, "r3_campaign", "t2c_codes", "d3_L2_s100.npz")


def load(path, n):
    conns = [[i, j] for i in range(n) for j in range(i + 1, n)]
    cs, d_, n_, K_ = encoder_forward(path, conns)
    assert (d_, n_) == (D, n)
    return np.asarray(cs)


def cell(cs, n, s):
    eta, sa2 = float(np.exp(-s * SIGMA_P2)), s * SIGMA_A2_CAL
    gp = [(i, j, 0, 1) for i in range(n) for j in range(i + 1, n)]
    steps = build_dose_steps(n, D, gp, eta, sigma_a2=sa2)
    inst = build_projective_instrument(cs, D, n, eta, sigma_a2=sa2)
    blocks = evolve_code_blocks(cs, steps, D, n)
    proj = entanglement_fidelity_projective(inst, blocks)
    petz = build_petz_exact_dose(cs, steps, D, n)
    pz = entanglement_fidelity_petz(petz, cs, blocks)
    return {"projective_rate": proj["rate_1mFavg"],
            "petz_rate": pz["rate_1mFavg"],
            "petz_support_rank": pz["support_rank"],
            "petz_leak_max": float(max(pz["recovery_leak_per_codeword"]))}


def petz_rcond_sweep(cs, n, s, rconds=(1e-11, 1e-12, 1e-13)):
    """One eigendecomposition of N(P_C); vary only the kept set."""
    eta, sa2 = float(np.exp(-s * SIGMA_P2)), s * SIGMA_A2_CAL
    gp = [(i, j, 0, 1) for i in range(n) for j in range(i + 1, n)]
    steps = build_dose_steps(n, D, gp, eta, sigma_a2=sa2)
    blocks = evolve_code_blocks(cs, steps, D, n)
    K, dim = cs.shape
    P_C = cs.T @ cs.conj()
    NPC = apply_dose(P_C, steps, D, n)
    evals, W = np.linalg.eigh(NPC)
    out = {}
    for rc in rconds:
        keep = evals > rc * evals.max()
        lam, Wk = evals[keep], W[:, keep]
        inv_sqrt = (Wk * (1.0 / np.sqrt(lam))[None, :]) @ Wk.conj().T
        f = np.zeros((K, K), dtype=complex)
        for i in range(K):
            for j in range(i, K):
                Y = inv_sqrt @ blocks[(i, j)] @ inv_sqrt
                Z = apply_dose(Y, steps, D, n, adjoint=True)
                f[i, j] = cs[i].conj() @ (Z @ cs[j])
                if j > i:
                    f[j, i] = np.conj(f[i, j])
        F_e = float(np.real(f.sum())) / K ** 2
        F_avg = favg_from_fe(F_e, K)
        out[f"rcond{rc:.0e}"] = {
            "rcond": rc, "support_rank": int(keep.sum()),
            "petz_rate": 1.0 - F_avg,
            "eig_min_kept": float(lam.min()), "eig_max": float(evals.max())}
    return out


def main():
    res = {"meta": {
        "purpose": "Petz crossing audit of the register family on the channel",
        "convention": "eta(s)=e^{-s*0.073}, sigma_a2(s)=s*0.047, channel v2",
        "petz_default_rcond": 1e-12}}
    t2a = json.load(open(os.path.join(R9, "t2a_results.json")))
    t1 = json.load(open(os.path.join(R9, "t1_results.json")))

    # ---------------- (a) independent recompute of suspect cells
    print("(a) recompute suspect cells", flush=True)
    recompute = {}
    deep = load(DEEP16, 5)
    head = load(HEADLINE, 5)
    checks = [
        ("deep16L_n5", deep, 5, 0.125, t2a["cells"]["n5@s0.125"]),
        ("deep16L_n5", deep, 5, 0.25, t2a["cells"]["n5@s0.250"]),
        ("headline_L2_n5", head, 5, 0.125,
         {"projective_rate":
          t1["cells"]["headline_A_L2_s100@v2_s0.125"]["projective_rate"],
          "petz_rate":
          t1["cells"]["headline_A_L2_s100@v2_s0.125"]["petz_rate"]}),
    ]
    for name, cs, n, s, ref in checks:
        t0 = time.time()
        c = cell(cs, n, s)
        recompute[f"{name}@s{s}"] = {
            "new": c,
            "committed": {"projective_rate": ref["projective_rate"],
                          "petz_rate": ref["petz_rate"]},
            "dev_proj": abs(c["projective_rate"] - ref["projective_rate"]),
            "dev_petz": abs(c["petz_rate"] - ref["petz_rate"]),
            "wall_s": round(time.time() - t0, 1)}
        print(f"  {name}@s{s}: dev_proj="
              f"{recompute[f'{name}@s{s}']['dev_proj']:.2e} dev_petz="
              f"{recompute[f'{name}@s{s}']['dev_petz']:.2e}", flush=True)
    res["recompute"] = recompute
    json.dump(res, open(OUT, "w"), indent=2)

    # ---------------- (b) rcond sweep at two crossing cells
    print("(b) rcond sweep", flush=True)
    sweep = {}
    t0 = time.time()
    sweep["deep16L_n5@s0.25"] = petz_rcond_sweep(deep, 5, 0.25)
    print(f"  n5 sweep done ({time.time()-t0:.0f}s)", flush=True)
    d3 = load(D3N7, 7)
    t0 = time.time()
    sweep["d3_n7@s0.25"] = petz_rcond_sweep(d3, 7, 0.25)
    sweep["d3_n7@s0.25"]["committed_petz_rate"] = \
        t2a["cells"]["n7@s0.250"]["petz_rate"]
    print(f"  n7 sweep done ({time.time()-t0:.0f}s)", flush=True)
    res["rcond_sweep"] = sweep
    json.dump(res, open(OUT, "w"), indent=2)

    # ---------------- (c) mechanism + significance numbers
    S_GRID = (0.125, 0.25, 0.5, 1.0, 1.5)
    mech = {"gate_steps_per_n": {n: n * (n - 1) // 2 for n in (5, 6, 7, 8, 9)},
            "per_s": {}}
    for s in S_GRID:
        sk = f"s{s:.3f}"
        deep_pz = t2a["cells"][f"n5@{sk}"]["petz_rate"]
        head_pz = t1["cells"][f"headline_A_L2_s100@v2_{sk}"]["petz_rate"]
        n6_pz = t2a["cells"][f"n6@{sk}"]["petz_rate"]
        n7_pz = t2a["cells"][f"n7@{sk}"]["petz_rate"]
        mech["per_s"][sk] = {
            "deep16L_n5_petz": deep_pz,
            "headline_L2_n5_petz": head_pz,
            "depth_penalty_deep_over_L2": deep_pz / head_pz,
            "n6_petz": n6_pz, "n7_petz": n7_pz,
            "deep_n5_above_n6": bool(deep_pz > n6_pz),
            "deep_n5_above_n7": bool(deep_pz > n7_pz),
            "L2_n5_below_n6": bool(head_pz < n6_pz)}
    # n >= 8: no Petz (sampled projective only); crossing significance is
    # a Petz-only question. Projective adjacent-n gaps vs stored SE:
    n8 = {k: v for k, v in t2a["cells"].items() if k.startswith("n8")}
    n9 = {k: v for k, v in t2a["cells"].items() if k.startswith("n9")}
    proj_margin = {}
    for s in S_GRID:
        sk = f"s{s:.3f}"
        c7 = t2a["cells"][f"n7@{sk}"]["projective_rate"]
        c8 = t2a["cells"][f"n8@{sk}"]
        c9 = t2a["cells"][f"n9@{sk}"]
        se8 = c8.get("projective_rate_se")
        se9 = c9.get("projective_rate_se")
        proj_margin[sk] = {
            "gap_n7_n8": c8["projective_rate"] - c7,
            "gap_n8_n9": c9["projective_rate"] - c8["projective_rate"],
            "se_n8": se8, "se_n9": se9}
    mech["n8plus_projective_margins"] = proj_margin
    res["mechanism"] = mech
    json.dump(res, open(OUT, "w"), indent=2)

    # ---------------- companion record: family panel with L=2 n5 member
    fam = {"meta": {
        "purpose": "F-corr-family with the n=5 member = central code A_L2_s100 "
                   "(no load-bearing L=16); cells "
                   "copied from committed r9 JSONs, provenance stated",
        "n5": "r9_campaign/t1_results.json headline_A_L2_s100@v2_*",
        "n6..n9": "r9_campaign/t2a_results.json"},
        "cells": {}}
    for s in S_GRID:
        sk = f"s{s:.3f}"
        h = t1["cells"][f"headline_A_L2_s100@v2_{sk}"]
        fam["cells"][f"n5@{sk}"] = {
            "n": 5, "s": s, "projective_rate": h["projective_rate"],
            "petz_rate": h["petz_rate"],
            "provenance": "r9_campaign/t1_results.json headline_A_L2_s100"}
        for n in (6, 7, 8, 9):
            c = dict(t2a["cells"][f"n{n}@{sk}"])
            c["provenance"] = "r9 t2a"
            fam["cells"][f"n{n}@{sk}"] = c
    booleans = {}
    for s in S_GRID:
        sk = f"s{s:.3f}"
        pj = [fam["cells"][f"n{n}@{sk}"]["projective_rate"]
              for n in (5, 6, 7, 8, 9)]
        pz = [fam["cells"][f"n{n}@{sk}"].get("petz_rate")
              for n in (5, 6, 7)]
        booleans[sk] = {
            "proj_monotone_rising_from_n5":
                bool(all(pj[i] < pj[i + 1] for i in range(4))),
            "petz_monotone_rising_from_n5_(n<=7)":
                bool(all(pz[i] < pz[i + 1] for i in range(2)))}
    fam["monotone_booleans"] = booleans
    json.dump(fam, open(os.path.join(HERE, "t1a_family_L2.json"), "w"),
              indent=2)
    res["family_L2_booleans"] = booleans
    json.dump(res, open(OUT, "w"), indent=2)
    print("booleans:", json.dumps(booleans), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
