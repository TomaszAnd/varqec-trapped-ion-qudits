#!/usr/bin/env python3
"""Sampler frontier assembly (Figs. 10-11) from the pilot/corner data.

Merges t1_results_shard{0,1,2}.json, builds the frontier table (one row
per (strategy, f) cell + the existing preset-1500 references), runs the
pre-registered early-exit arithmetic, and writes t3_analysis.json.
Pure analysis: no simulation, no training.
"""
import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EST_SE = None  # filled from data (mean per-cell F_avg_se)


def wilson(k, n, z=1.96):
    ph = k / n
    dn = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / dn
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / dn
    return (max(0.0, c - h), c + h)


def main():
    units = {}
    for f in sorted(glob.glob(os.path.join(HERE, "t1_results_shard*.json"))):
        units.update(json.load(open(f))["units"])

    cells = {}
    for key, u in units.items():
        cell = ("existing_" + u["name"]) if u["kind"] == "existing" else \
            f"{u['sampler']}@{u['frac']}"
        cells.setdefault(cell, []).append(u)

    frontier = {}
    for cell, rows in sorted(cells.items()):
        ks = [r["ler_iid_p05"]["k"] for r in rows]
        ns = [r["ler_iid_p05"]["n"] for r in rows]
        cr = [r["corr_v2_s05_proj"]["rate_1mFavg"] for r in rows]
        se = [r["corr_v2_s05_proj"]["F_avg_se"] for r in rows]
        kp, np_ = int(sum(ks)), int(sum(ns))
        row = {
            "n_seeds": len(rows),
            "certified": sum(1 for r in rows if r.get("certified")),
            "stop_steps": sorted(r["stop_step"] for r in rows
                                 if "stop_step" in r),
            "budget_op_evs": sorted(r["budget_op_evs"] for r in rows
                                    if "budget_op_evs" in r),
            "train_op_evs": sorted(r["train_op_evs"] for r in rows
                                   if "train_op_evs" in r),
            "confirm_op_evs": sorted(r["confirm_op_evs"] for r in rows
                                     if "confirm_op_evs" in r),
            "n_confirms": sorted(r["n_confirms"] for r in rows
                                 if "n_confirms" in r),
            "n_premature": sorted(r["n_premature"] for r in rows
                                  if "n_premature" in r),
            "full_loss": sorted(r["full_loss"] for r in rows
                                if "full_loss" in r),
            "w1_failures": [r["ler_w1"]["k"] for r in rows],
            "w2_failures": [r["ler_w2"]["k"] for r in rows],
            "iid_k_per_seed": ks,
            "iid_pooled": {"k": kp, "n": np_, "ler": kp / np_,
                           "wilson95": wilson(kp, np_)},
            "corr_rates": cr,
            "corr_mean": float(np.mean(cr)),
            "corr_seed_sd": float(np.std(cr, ddof=1)) if len(cr) > 1 else None,
            "corr_est_se": float(np.mean(se)),
        }
        frontier[cell] = row

    # ---- early-exit / knee arithmetic over the PILOT cells ----
    pilot = {c: r for c, r in frontier.items() if not c.startswith("existing")}
    verdict = {}
    if pilot:
        means = {c: r["corr_mean"] for c, r in pilot.items()}
        cmax = max(means, key=means.get)
        cmin = min(means, key=means.get)
        d_corr = means[cmax] - means[cmin]
        sd_pool = float(np.sqrt(np.mean(
            [r["corr_seed_sd"] ** 2 for r in pilot.values()
             if r["corr_seed_sd"] is not None])))
        se_est = float(np.mean([r["corr_est_se"] for r in pilot.values()]))
        sigma = math.sqrt(sd_pool ** 2 + se_est ** 2)
        resolvable_n8 = 2 * sigma * math.sqrt(2 / 8)
        # iid, primary test (1 dof, pre-registered direction): a knee means
        # the CHEAPEST-f cells are worse, so pool the f=0.05 cells against
        # the pooled reference (full batch + f=0.4 cells) and test the
        # conditional-binomial split of the pooled error count.
        cheap = [c for c in pilot if c.endswith("@0.05")]
        ref = [c for c in pilot if c not in cheap]
        kc = sum(pilot[c]["iid_pooled"]["k"] for c in cheap)
        nc = sum(pilot[c]["iid_pooled"]["n"] for c in cheap)
        kr = sum(pilot[c]["iid_pooled"]["k"] for c in ref)
        nr = sum(pilot[c]["iid_pooled"]["n"] for c in ref)
        ktot, frac = kc + kr, nc / (nc + nr)
        z_dir = ((kc - ktot * frac)
                 / math.sqrt(ktot * frac * (1 - frac))) if ktot else 0.0
        # secondary, for the record: max pairwise z with the number of
        # comparisons stated (Bonferroni context, NOT a selection input)
        z_max, z_pair, n_pairs = 0.0, None, 0
        cl = list(pilot)
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                a, b = pilot[cl[i]]["iid_pooled"], pilot[cl[j]]["iid_pooled"]
                if a["k"] + b["k"] == 0:
                    continue
                n_pairs += 1
                z = abs(a["k"] - b["k"]) / math.sqrt(a["k"] + b["k"])
                if z > z_max:
                    z_max, z_pair = z, (cl[i], cl[j])
        all_cert = all(r["certified"] == r["n_seeds"] for r in pilot.values())
        verdict = {
            "all_pilot_cells_fully_certified": all_cert,
            "corr_effect_max": {"pair": [cmin, cmax], "delta": d_corr},
            "corr_sigma": {"seed_sd_pooled": sd_pool, "est_se": se_est,
                           "combined": sigma},
            "corr_resolvable_at_N8": resolvable_n8,
            "corr_effect_resolvable": bool(d_corr >= resolvable_n8),
            "iid_directed_contrast": {
                "cheap_cells": cheap, "k_cheap": kc, "n_cheap": nc,
                "ref_cells": ref, "k_ref": kr, "n_ref": nr, "z": z_dir},
            "iid_effect_resolvable_at_N8": bool(abs(z_dir) * math.sqrt(2) >= 2),
            "iid_max_pairwise_z_record": {"z": z_max, "pair": z_pair,
                                          "n_comparisons": n_pairs},
            "early_exit": bool(all_cert
                               and d_corr < resolvable_n8
                               and abs(z_dir) * math.sqrt(2) < 2),
        }

    out = {"frontier": frontier, "verdict": verdict,
           "n_units": len(units)}
    json.dump(out, open(os.path.join(HERE, "t3_analysis.json"), "w"),
              indent=2)

    print(f"{len(units)} units, {len(frontier)} cells\n")
    hdr = (f"{'cell':26s} {'N':>2} {'cert':>4} {'budget med':>11} "
           f"{'iid pooled LER (Wilson95)':>34} {'corr mean±seedSD':>22}")
    print(hdr)
    for cell, r in frontier.items():
        b = r["budget_op_evs"]
        bmed = f"{int(np.median(b))}" if b else "-"
        ip = r["iid_pooled"]
        w = ip["wilson95"]
        sd = f"±{r['corr_seed_sd']:.1e}" if r["corr_seed_sd"] else ""
        print(f"{cell:26s} {r['n_seeds']:2d} {r['certified']:4d} {bmed:>11} "
              f"{ip['ler']:.2e} [{w[0]:.1e},{w[1]:.1e}] (k={ip['k']:3d})"
              f" {r['corr_mean']:.5f}{sd}")
    print("\nverdict:", json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
