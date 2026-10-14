#!/usr/bin/env python3
"""Gate-study calibrated-channel panel: gate-study data on the full
strength grid.

Exact engine at n=6, channel v2. Per-gate probe-selected code (standard
rule: min k_w2, tiebreak k_w3, then seed — derived here from the
r5_campaign/t1_results.json probe records and stated in the output) for
all five gates, at the missing strengths s in {0.125, 0.25, 1.5}; the
s in {0.5, 1.0} cells are imported from committed
r9_campaign/t2c_results.json with provenance. The 60-code s=0.5 per-gate
median + range block also comes from r9_campaign/t2c_results.json
(already complete).

Writes t5_results.json, shaped for a rate-vs-s figure panel: per gate
one curve over the full s grid, projective + Petz. Feeds the gate-study
channel panel of the paper.
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
OUT = os.path.join(HERE, "t5_results.json")
BR = os.path.join(REPO, "results", "campaigns")
R5_T1 = os.path.join(BR, "r5_campaign", "t1_results.json")
R9_T2C = os.path.join(BR, "r9_campaign", "t2c_results.json")

SIGMA_P2, SIGMA_A2_CAL = 0.073, 0.047
D, N = 3, 6
GP = [(i, j, 0, 1) for i in range(N) for j in range(i + 1, N)]
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]
S_NEW = (0.125, 0.25, 1.5)
S_IMPORT = (0.5, 1.0)
GATES = ("ms", "zz", "ls", "csum", "cex")


def select_per_gate(units):
    """Standard probe selection: min k_w2, tiebreak k_w3, then seed."""
    sel = {}
    for g in GATES:
        cands = []
        for name, u in units.items():
            if u["gate"] != g:
                continue
            cands.append((u["ler_w2"]["k"], u["ler_w3"]["k"], u["seed"],
                          name, u["npz"]))
        cands.sort()
        k2, k3, seed, name, npz = cands[0]
        sel[g] = {"unit": name, "seed": seed, "npz": npz,
                  "k_w2": k2, "k_w3": k3,
                  "rule": "min k_w2, tiebreak k_w3, then seed"}
    return sel


def main():
    res = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            res = json.load(f)
    res.setdefault("meta", {"d": D, "n": N, "gate_steps": len(GP),
                            "s_grid_full": sorted(S_NEW + S_IMPORT),
                            "s_new": list(S_NEW),
                            "s_imported": list(S_IMPORT),
                            "imported_from": "r9_campaign/t2c_results.json",
                            "convention": "eta(s)=e^{-s*0.073}, "
                                          "sigma_a2(s)=s*0.047",
                            "metric": "rate = 1 - F_avg (channel infidelity)"})
    res.setdefault("cells", {})

    with open(R5_T1) as f:
        r5 = json.load(f)
    sel = select_per_gate(r5["units"])
    res["selection"] = sel
    for g in GATES:
        print(f"selected {g}: {sel[g]['unit']} (k_w2={sel[g]['k_w2']}, "
              f"k_w3={sel[g]['k_w3']})", flush=True)

    for s in S_NEW:
        eta, sa2 = float(np.exp(-s * SIGMA_P2)), s * SIGMA_A2_CAL
        steps = build_dose_steps(N, D, GP, eta, sigma_a2=sa2)
        for g in GATES:
            key = f"{sel[g]['unit']}@s{s:.3f}"
            if key in res["cells"]:
                continue
            t0 = time.time()
            cs, d_, n_, K_ = encoder_forward(sel[g]["npz"], CONNS)
            assert (d_, n_) == (D, N)
            inst = build_projective_instrument(cs, D, N, eta, sigma_a2=sa2)
            blocks = evolve_code_blocks(cs, steps, D, N)
            proj = entanglement_fidelity_projective(inst, blocks)
            petz = build_petz_exact_dose(cs, steps, D, N)
            pz = entanglement_fidelity_petz(petz, cs, blocks)
            res["cells"][key] = {
                "gate": g, "seed": sel[g]["seed"], "s": s, "eta": eta,
                "sigma_a2": sa2,
                "projective_rate": proj["rate_1mFavg"],
                "petz_rate": pz["rate_1mFavg"],
                "petz_leak_max": float(max(pz["recovery_leak_per_codeword"])),
                "source": "this run",
                "wall_s": round(time.time() - t0, 1),
            }
            with open(OUT, "w") as f:
                json.dump(res, f, indent=2)
            print(f"{key}: proj={proj['rate_1mFavg']:.4e} "
                  f"petz={pz['rate_1mFavg']:.4e} "
                  f"({res['cells'][key]['wall_s']}s)", flush=True)

    # import s in {0.5, 1.0} for the selected codes from r9 t2c
    with open(R9_T2C) as f:
        t2c = json.load(f)
    for s in S_IMPORT:
        for g in GATES:
            key = f"{sel[g]['unit']}@s{s:.3f}"
            if key in res["cells"]:
                continue
            c = t2c["cells"][key]
            res["cells"][key] = dict(c)
            res["cells"][key]["source"] = \
                "imported r9_campaign/t2c_results.json"

    # figure-shaped panel: one curve per gate over the full grid
    s_all = sorted(S_NEW + S_IMPORT)
    panel = {"s_grid": s_all, "curves": {}}
    for g in GATES:
        cur = {"unit": sel[g]["unit"],
               "projective_rate": [], "petz_rate": []}
        for s in s_all:
            c = res["cells"][f"{sel[g]['unit']}@s{s:.3f}"]
            cur["projective_rate"].append(c["projective_rate"])
            cur["petz_rate"].append(c["petz_rate"])
        panel["curves"][g] = cur
    res["figure_panel"] = panel

    # 60-code per-gate median + range at s=0.5 (from committed t2c)
    med = {}
    for g in GATES:
        for lab in ("projective_rate", "petz_rate"):
            vals = [c[lab] for c in t2c["cells"].values()
                    if c["gate"] == g and abs(c["s"] - 0.5) < 1e-9]
            assert len(vals) == 12, (g, lab, len(vals))
            med.setdefault(g, {})[lab] = {
                "median": float(np.median(vals)),
                "min": float(min(vals)), "max": float(max(vals)),
                "n_seeds": len(vals),
                "source": "r9_campaign/t2c_results.json"}
    res["all60_s0.5_per_gate"] = med
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print("done")


if __name__ == "__main__":
    main()
