#!/usr/bin/env python3
"""Selector matrix: the full selector-vs-ground-truth matrix, recomputed
from the r11_gapfill/t6_selectors.py pool (no new cells needed — every
selector x GT pair exists).

Pool: the 16 t1b L=2 codes (hw/full arms, 8 seeds each). Selectors:
train loss, plateau loss, full-closure residual, hw-closure residual,
w2 probe (k_w2, tiebreak k_w3, then seed). Ground truths: measured iid
LER @ p=0.05, exact channel-v2 projective AND Petz rates at s in
{0.5, 1.0}. Exact C(8,N) enumeration per arm + pooled-16 variant.
Cross-validated against r11_gapfill/t6_selector_results.json per_arm
rows (iid / proj) — max relative deviation reported (must be ~0).

Writes t4_matrix.json. Feeds the selector matrix of the paper.
"""
import itertools
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BR = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(BR, "r11_gapfill", "t6_selector_results.json")

r11 = json.load(open(SRC))
pool = r11["pool"]

SELECTORS = {
    "train_loss": lambda c: (c["train_loss"],),
    "plateau_loss": lambda c: (c["plateau_loss"],),
    "full_residual": lambda c: (c["full_residual"],),
    "hw_residual": lambda c: (c["hw_residual"],),
    "w2_probe": lambda c: (c["k_w2"], c["k_w3"], c["seed"]),
}
GTS = ["iid_p05", "proj_s0.5", "proj_s1.0", "petz_s0.5", "petz_s1.0"]


def wilson(k, n, z=1.96):
    ph = k / n
    dn = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / dn
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / dn
    return (max(0.0, c - h), c + h)


def enumerate_arm(names):
    """selector -> gt -> {N: E[gt of selected]}, plus oracle/anti rows."""
    out = {}
    codes = {nm: pool[nm] for nm in names}
    for sel, keyfn in SELECTORS.items():
        out[sel] = {gt: {} for gt in GTS}
        for N in range(1, len(names) + 1):
            tot = {gt: 0.0 for gt in GTS}
            cnt = 0
            for sub in itertools.combinations(names, N):
                chosen = min(sub, key=lambda nm: keyfn(codes[nm]))
                for gt in GTS:
                    tot[gt] += codes[chosen]["gt"][gt]
                cnt += 1
            for gt in GTS:
                out[sel][gt][f"N{N}"] = tot[gt] / cnt
    for gt in GTS:
        vals = sorted(codes[nm]["gt"][gt] for nm in names)
        out.setdefault("oracle", {})[gt] = vals[0]
        out.setdefault("worst", {})[gt] = vals[-1]
        out.setdefault("spread_max_over_min", {})[gt] = vals[-1] / vals[0]
    # the code each selector picks at N = all (deterministic pick)
    out["pick_at_Nfull"] = {
        sel: min(names, key=lambda nm: keyfn(codes[nm]))
        for sel, keyfn in SELECTORS.items()}
    out["oracle_pick"] = {
        gt: min(names, key=lambda nm: codes[nm]["gt"][gt]) for gt in GTS}
    return out


arms = {"hw": [n for n in pool if n.startswith("hw_")],
        "full": [n for n in pool if n.startswith("full_")],
        "pooled16": list(pool)}
res = {"meta": {"source": "r11_gapfill/t6_selector_results.json pool",
                "selectors": list(SELECTORS), "ground_truths": GTS,
                "note": "expected GT of the selector-chosen code, exact "
                        "C(M,N) enumeration; oracle = best code per GT"},
       "arms": {}}
for arm, names in arms.items():
    res["arms"][arm] = enumerate_arm(names)

# iid Wilson intervals of each pick at N=full, for the intervals column
for arm, names in arms.items():
    picks = res["arms"][arm]["pick_at_Nfull"]
    res["arms"][arm]["iid_wilson_of_picks"] = {
        sel: {"code": nm,
              "ler": pool[nm]["gt"]["iid_p05"],
              "wilson95": wilson(pool[nm]["gt"]["iid_p05_k"],
                                 pool[nm]["gt"]["iid_p05_n"])}
        for sel, nm in picks.items()}

# ---- cross-validation against the committed r11 per_arm rows ----
dev = 0.0
for arm in ("hw", "full"):
    for sel in SELECTORS:
        for gt in ("iid_p05", "proj_s0.5", "proj_s1.0"):
            for N in range(1, 9):
                a = r11["per_arm"][arm][sel][gt][f"N{N}"]
                b = res["arms"][arm][sel][gt][f"N{N}"]
                dev = max(dev, abs(a - b) / abs(a))
res["crossval_max_rel_dev_vs_r11"] = dev

# ---- rankings per ground truth (N=4 and N=8, per arm) ----
rank = {}
for arm in ("hw", "full"):
    for gt in GTS:
        for N in ("N4", "N8"):
            rows = sorted(
                ((res["arms"][arm][sel][gt][N], sel) for sel in SELECTORS))
            rank[f"{arm}:{gt}:{N}"] = [
                {"selector": s, "E": v,
                 "vs_oracle": v / res["arms"][arm]["oracle"][gt]}
                for v, s in rows]
res["rankings"] = rank

with open(os.path.join(HERE, "t4_matrix.json"), "w") as f:
    json.dump(res, f, indent=2)

print(f"crossval max rel dev vs r11: {dev:.2e}")
for arm in ("hw", "full"):
    print(f"== {arm} arm (N=8 picks) ==")
    for gt in GTS:
        rows = rank[f"{arm}:{gt}:N8"]
        line = "  ".join(f"{r['selector']}={r['vs_oracle']:.3f}x"
                         for r in rows)
        print(f" {gt:10s}: {line}")
    print("  spread:", {gt: round(
        res['arms'][arm]['spread_max_over_min'][gt], 3) for gt in GTS})
