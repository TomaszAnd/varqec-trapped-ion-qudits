#!/usr/bin/env python3
"""Selector comparison: seed-selection ground-truth study on the L=2 pool.

Pool: the 16 t1b_queue.py codes (hw/full unweighted arms x seeds 100..107
at L=2, r11_gapfill/t1b_codes). Two ground truths per code:
  (a) measured iid LER at p = 0.05 (escalated, >= 30 errors);
  (b) exact physical-channel projective rate at s = 0.5 and s = 1.0
      (channel v2, projective-MAP; the Petz rate is carried alongside
      as a robustness column, not a third ground truth).

Selectors (each picks one code from a seed subset):
  - train_loss:     lowest final training loss (the published negative
                    control);
  - plateau_loss:   lowest early-stopping loss -- the trace value at the
                    plateau of r1_campaign/t1_queue.py Part B (first three
                    consecutive 100-step windows improving < 1%; seeds
                    that never plateau use their final loss, stated per
                    seed);
  - full_residual:  lowest full-closure KL residual (cross-evaluated);
  - hw_residual:    lowest hardware-closure KL residual;
  - w2_probe:       standard selection (k_w2, k_w3, seed) lexicographic.

Scoring: exact expected ground-truth value of the selected code over all
C(8,N) seed subsets per arm, N = 1..8 (the sampling_budget_v2/c1_bestofn.py
protocol), plus the oracle (select by the ground truth itself) and the
pooled-16 variant at N in {4, 8, 16}. Questions: does the probe selector
track both ground truths; which selectors flip ranking between metrics.

Run AFTER both t1b shards complete:
  python3 t6_selectors.py        (fast -- pure analysis, no simulation)
Writes t6_selector_results.json. Feeds the selector comparison of the paper.
"""
import itertools
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "t6_selector_results.json")

SEEDS = list(range(100, 108))
GT_KEYS = ("iid_p05", "proj_s0.5", "proj_s1.0")


def plateau_loss(trace):
    """Plateau rule of r1_campaign/t1_queue.py Part B: window k = steps [100k, 100(k+1)); L_k = window
    mean; plateau at the first k with three consecutive improvements
    < 1%. Returns (loss_at_plateau, fired, plateau_step)."""
    tr = np.asarray(trace, float)
    nw = len(tr) // 100
    if nw < 4:
        return float(tr[-1]), False, None
    L = np.array([tr[100 * k:100 * (k + 1)].mean() for k in range(nw)])
    consec = 0
    for k in range(1, nw):
        impr = (L[k - 1] - L[k]) / max(L[k - 1], 1e-300)
        consec = consec + 1 if impr < 0.01 else 0
        if consec >= 3:
            return float(L[k]), True, int(100 * (k + 1))
    return float(tr[-1]), False, None


def load_pool():
    pool = {}
    for arm in ("hw", "full"):
        res = json.load(open(os.path.join(HERE, f"t1b_results_{arm}.json")))
        for key, u in res["units"].items():
            trace = np.load(u["npz"], allow_pickle=True)
            losses = np.asarray(trace["losses"], float) \
                if "losses" in trace.files else np.array([u["train_loss"]])
            pl, fired, pstep = plateau_loss(losses)
            pool[key] = {
                "arm": arm, "seed": u["seed"],
                "train_loss": u["train_loss"],
                "plateau_loss": pl, "plateau_fired": fired,
                "plateau_step": pstep,
                "full_residual": u["full_basis_loss"],
                "hw_residual": u["hw_basis_loss"],
                "k_w2": u["ler_w2"]["k"], "k_w3": u["ler_w3"]["k"],
                "gt": {
                    "iid_p05": u["ler_iid_p05"]["ler"],
                    "iid_p05_k": u["ler_iid_p05"]["k"],
                    "iid_p05_n": u["ler_iid_p05"]["n_shots"],
                    "proj_s0.5": u["exact_v2"]["s0.500"]["projective_rate"],
                    "proj_s1.0": u["exact_v2"]["s1.000"]["projective_rate"],
                    "petz_s0.5": u["exact_v2"]["s0.500"]["petz_rate"],
                    "petz_s1.0": u["exact_v2"]["s1.000"]["petz_rate"]}}
    return pool


SELECTORS = {
    "train_loss": lambda c: (c["train_loss"],),
    "plateau_loss": lambda c: (c["plateau_loss"],),
    "full_residual": lambda c: (c["full_residual"],),
    "hw_residual": lambda c: (c["hw_residual"],),
    "w2_probe": lambda c: (c["k_w2"], c["k_w3"], c["seed"]),
}


def expected_pick(codes, selector_key, gt_key, N):
    """Mean ground-truth value of the selector's pick over all C(len,N)
    subsets."""
    vals = []
    sel = SELECTORS.get(selector_key)
    for sub in itertools.combinations(codes, N):
        if selector_key.startswith("oracle:"):
            g = selector_key.split(":", 1)[1]
            pick = min(sub, key=lambda c: c["gt"][g])
        else:
            pick = min(sub, key=sel)
        vals.append(pick["gt"][gt_key])
    return float(np.mean(vals))


def main():
    pool = load_pool()
    out = {"meta": {
        "pool": "t1b_codes (hw/full unweighted arms at L=2, 8 seeds each)",
        "ground_truths": "iid LER @ p=0.05 (measured); exact channel-v2 "
                         "projective rate @ s in {0.5, 1.0}; Petz carried "
                         "as robustness column",
        "plateau_rule": "window mean over 100 steps; three consecutive "
                        "improvements < 1%; never-fires -> final loss "
                        "(stated per seed)",
        "protocol": "exact C(8,N) subset enumeration per arm (SBV2 C1); "
                    "pooled-16 variant at N in {4, 8, 16}"},
        "pool": {k: {kk: vv for kk, vv in v.items()}
                 for k, v in pool.items()}}

    selector_keys = list(SELECTORS) + [f"oracle:{g}" for g in GT_KEYS]
    tables = {}
    for arm in ("hw", "full"):
        codes = [c for c in pool.values() if c["arm"] == arm]
        tab = {}
        for sk in selector_keys:
            tab[sk] = {}
            for gt in GT_KEYS:
                tab[sk][gt] = {f"N{N}": expected_pick(codes, sk, gt, N)
                               for N in range(1, 9)}
        tables[arm] = tab
    pooled = {}
    allc = list(pool.values())
    for sk in selector_keys:
        pooled[sk] = {}
        for gt in GT_KEYS:
            pooled[sk][gt] = {f"N{N}": expected_pick(allc, sk, gt, N)
                              for N in (4, 8, 16)}
    out["per_arm"] = tables
    out["pooled16"] = pooled

    # ranking-flip report: order selectors by expected GT at N=4, per GT
    flips = {}
    for arm in ("hw", "full"):
        r = {}
        for gt in GT_KEYS:
            order = sorted(SELECTORS,
                           key=lambda sk: tables[arm][sk][gt]["N4"])
            r[gt] = order
        flips[arm] = {
            "order_by_gt": r,
            "flip_iid_vs_s0.5": r["iid_p05"] != r["proj_s0.5"],
            "flip_iid_vs_s1.0": r["iid_p05"] != r["proj_s1.0"]}
    out["ranking_flips_N4"] = flips

    json.dump(out, open(OUT, "w"), indent=2, default=str)
    print(json.dumps(flips, indent=1))
    print("done")


if __name__ == "__main__":
    main()
