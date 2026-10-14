#!/usr/bin/env python3
"""Figure renders of the w8 manuscript pass (2026-08-17).

Reads committed result data under results/campaigns/ (read-only),
writes into results/figures/ only. Figures rebuilt in that pass
(figure edits D1-D3 and the former Table IV as a figure):

  fig_dist_w8.png      D1: repeated-round distance figure, projective
                       decoder only (Petz curves removed); from
                       r11_gapfill/t4_analysis.json.
  fig_sampler_frontier_w8.png
                       D2: sampler cost-quality comparison without the
                       preset-protocol open squares; axes re-tightened
                       to the certified points; from r10_sampler/
                       t3_analysis.json.
  fig_sampler_traces_w8.png
                       D3: convergence redesign; per strategy, the
                       MEDIAN full-batch monitor loss over all four
                       certified seeds vs cumulative training op-EVs
                       with a min-max band (full-batch strategy uses
                       its per-step losses, which ARE full-batch
                       evaluations); median certified stop starred;
                       tau line kept; from r10_sampler/t1_codes/*.npz.
  fig_headline_pairs_w8.png
                       C167: the paired two-benchmark comparison
                       (former Table IV) as a figure; from r11_gapfill/
                       t2_table1_results.json.

Colors for the sampler figures match across Fig 9/Fig 10 (w5 scheme).
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BP = os.path.join(REPO, "results", "campaigns")
R10 = os.path.join(BP, "r10_sampler")
R11 = os.path.join(BP, "r11_gapfill")
FIG = os.path.join(REPO, "results", "figures")
os.makedirs(FIG, exist_ok=True)

S_GRID = (0.125, 0.25, 0.5, 1.0, 1.5)
# Paul Tol MUTED qualitative scheme (scripts/tol_colors.py); the sand
# accent is darkened (#DDCC77 -> #B8A14C) for lines/markers on white.
N_COL5 = "#B8A14C"

plt.rcParams.update({"font.size": 10, "axes.labelsize": 11,
                     "legend.fontsize": 8, "figure.dpi": 200})


def jload(*parts):
    with open(os.path.join(*parts)) as fh:
        return json.load(fh)


# ------------------------------------------------------- D1: fig_dist
def fig_dist():
    t4 = jload(R11, "t4_analysis.json")
    proj = t4["lambda_vs_s_projective"]
    skeys = ["s0.125", "s0.25", "s0.5", "s1.0", "s1.5"]
    svals = S_GRID
    rows = [("d2", "$((7,3,2))_3$", "#CC6677"),
            ("d3", "$((7,3,3))_3$", "#882255"),
            ("d4", "$((7,3,4))_3$", "#AA4499"),
            ("headline", "$((5,3,3))_3$ central code", N_COL5),
            ("five_one_three_Z3", "$[[5,1,3]]_{Z_3}$", "0.25")]
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    for key, lab, col in rows:
        ys = [proj[key][sk]["per_round_infid"] for sk in skeys]
        mk = "s" if key == "five_one_three_Z3" else "o"
        ax.plot(svals, ys, "-", marker=mk, ms=4.5, color=col, lw=1.4,
                label=lab)
        if key in ("d2", "d3", "d4"):
            # s = 1.5 cell saturation-dominated (F_e at the 1/9 floor
            # within one round): open marker
            ax.plot([svals[-1]], [ys[-1]], marker=mk, ms=4.5,
                    mfc="white", mec=col, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(svals, [str(s) for s in svals])
    ax.minorticks_off()
    ax.set_xlabel("channel strength $s$   (calibrated $s = 1$)")
    ax.set_ylabel(r"per-round infidelity $1 - \lambda$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_dist_w8.png"))
    plt.close(fig)
    print("F-dist d2/d4 per-round margins:",
          [round(t4["trio_ordering_by_s"][sk]
                 ["per_round_ratio_d2_over_d4"], 4) for sk in skeys])
    return True


# --------------------------------------------------- D2: fig_frontier
CELL_LAB = {"full_batch@1.0": "full batch (certified)",
            "importance@0.4": "importance $f{=}0.4$",
            "importance@0.05": "importance $f{=}0.05$",
            "full_basis_is@0.4": "fb-importance $f{=}0.4$",
            "full_basis_is@0.05": "fb-importance $f{=}0.05$"}
CELL_COL = {"full_batch@1.0": "#B8A14C", "importance@0.4": "#332288",
            "importance@0.05": "#88CCEE",
            "full_basis_is@0.4": "#882255",
            "full_basis_is@0.05": "#CC6677"}


def fig_frontier():
    fr = jload(R10, "t3_analysis.json")["frontier"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.4, 7.2),
                                   sharex=True)
    corr_vals = []
    for cell, lab in CELL_LAB.items():
        c = fr[cell]
        x = float(np.median(c["budget_op_evs"]))
        xlo = x - min(c["budget_op_evs"])
        xhi = max(c["budget_op_evs"]) - x
        ler = c["iid_pooled"]["ler"]
        w = c["iid_pooled"]["wilson95"]
        ax1.errorbar([x], [ler], xerr=[[xlo], [xhi]],
                     yerr=[[ler - w[0]], [w[1] - ler]], marker="o",
                     ms=6.5, color=CELL_COL[cell], capsize=2.5, lw=1.2,
                     label=lab, zorder=4)
        ax2.errorbar([x], [c["corr_mean"]], xerr=[[xlo], [xhi]],
                     yerr=[[c["corr_est_se"]], [c["corr_est_se"]]],
                     marker="o", ms=6.5, color=CELL_COL[cell],
                     capsize=2.5, lw=1.2, zorder=4)
        corr_vals += [c["corr_mean"] - c["corr_est_se"],
                      c["corr_mean"] + c["corr_est_se"]]
        for r in c["corr_rates"]:
            ax2.plot([x], [r], marker=".", ms=3.5,
                     color=CELL_COL[cell], zorder=5)
            corr_vals.append(r)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylabel("i.i.d. LER at $p{=}0.05$\n(4-seed pool, Wilson 95%)")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01),
               ncol=2, fontsize=7.2, frameon=False)
    ax2.set_ylabel("calibrated-channel projective\ninfidelity at "
                   "$s{=}0.5$ (mean $\\pm$ SE)")
    ax2.set_xlabel("certified training budget (train + confirmation "
                   "op-EVs)")
    pad = 0.15 * (max(corr_vals) - min(corr_vals))
    ax2.set_ylim(min(corr_vals) - pad, max(corr_vals) + pad)
    ax2.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_sampler_frontier_w8.png"))
    plt.close(fig)
    lo = min(min(fr[c]["budget_op_evs"]) for c in CELL_LAB)
    hi = max(max(fr[c]["budget_op_evs"]) for c in CELL_LAB)
    print(f"frontier certified budgets span {lo}-{hi} op-EVs")
    return True


# ----------------------------------------------------- D3: fig_traces
STRAT = [("full_batch", "full batch (certified)", "#B8A14C"),
         ("importance_0.4", "importance $f{=}0.4$", "#332288"),
         ("importance_0.05", "importance $f{=}0.05$", "#88CCEE"),
         ("full_basis_is_0.4", "fb-importance $f{=}0.4$", "#882255"),
         ("full_basis_is_0.05", "fb-importance $f{=}0.05$", "#CC6677")]


def fig_traces():
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    tau = None
    all_stops, all_fl = [], []
    for prefix, lab, col in STRAT:
        seeds = sorted(glob.glob(
            os.path.join(R10, "t1_codes", f"{prefix}_s1??.npz")))
        assert len(seeds) == 4, (prefix, seeds)
        series, stop_x, fl = [], [], []
        for p in seeds:
            z = np.load(p, allow_pickle=True)
            tau = float(z["tau"])
            cum = np.asarray(z["cum_train_evs"], float)
            stop = int(z["stop_step"])
            all_stops.append(stop)
            fl.append(float(z["full_loss"]))
            all_fl.append(float(z["full_loss"]))
            stop_x.append(cum[min(stop, len(cum) - 1)])
            if prefix == "full_batch":
                # every step of the full-batch strategy IS a full-batch
                # evaluation; verified equal to the monitor log below
                ms = np.asarray(z["monitor_steps"], int)
                ml = np.asarray(z["monitor_losses"], float)
                sl = np.asarray(z["sampled_losses"], float)
                assert np.allclose(sl[np.clip(ms, 0, len(sl) - 1)], ml)
                xs, ys = cum, sl
            else:
                ms = np.asarray(z["monitor_steps"], int)
                ml = np.asarray(z["monitor_losses"], float)
                xs = np.interp(ms, np.arange(len(cum)), cum)
                ys = ml
            # append the certified full-batch value at the stop so every
            # seed's series ends at its confirmed residual
            xs = np.append(np.asarray(xs, float), stop_x[-1])
            ys = np.append(np.asarray(ys, float), fl[-1])
            series.append((xs, ys))
        # common log-x grid across the four seeds (clamped interpolation:
        # a seed that stopped earlier holds its confirmed value)
        x0 = max(min(a[a > 0]) for a, _ in series)
        x1 = max(a.max() for a, _ in series)
        grid = np.geomspace(max(x0, 1.0), x1, 80)
        vals = np.stack([np.interp(np.log(grid),
                                   np.log(np.maximum(a, 1.0)), b)
                         for a, b in series])
        med = np.median(vals, axis=0)
        ax.plot(grid, med, color=col, lw=1.5, label=lab, zorder=4)
        ax.fill_between(grid, vals.min(axis=0), vals.max(axis=0),
                        color=col, alpha=0.16, lw=0, zorder=2)
        ax.plot([float(np.median(stop_x))], [float(np.median(fl))],
                marker="*", ms=12, color=col, mec="k", mew=0.4,
                zorder=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(tau, color="0.4", lw=0.9, ls=":")
    ax.text(0.02, 0.045, r"$\tau = 10^{-4}$", transform=ax.transAxes,
            fontsize=8, color="0.3", va="bottom")
    ax.set_xlabel("cumulative training op-EVs")
    ax.set_ylabel("full-batch channel-weighted loss")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.2, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_sampler_traces_w8.png"))
    plt.close(fig)
    print(f"traces: {len(all_stops)} runs, certified stops "
          f"{min(all_stops)}-{max(all_stops)}, confirmed residuals "
          f"{min(all_fl):.2e}-{max(all_fl):.2e}")
    return True


# --------------------------------------------- C167: fig_headline_pairs
HP_CODES = [("t1head", "$((5,3,3))_3$ central code (MS 40)", "#B8A14C"),
            ("c_n5", "$((5,3,3))_3$ unweighted $L{=}16$ (MS 320)",
             "#332288"),
            ("s513", "$[[5,1,3]]_{Z_3}$ (MS 36)", "0.25")]
P_GRID = (0.02, 0.03, 0.05, 0.08, 0.12, 0.15)


def fig_headline_pairs():
    t2 = jload(R11, "t2_table1_results.json")
    cc, ic = t2["companion_cells"], t2["iid_cells"]
    # spot-check against the retired Table IV
    assert abs(cc["t1head@p0.02"]["petz_rate"] - 3.84e-2) < 5e-4
    assert abs(cc["s513@p0.05"]["projective_rate"] - 0.362) < 5e-3
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.4, 7.2),
                                   sharex=True)
    for code, lab, col in HP_CODES:
        petz = [cc[f"{code}@p{p}"]["petz_rate"] for p in P_GRID]
        proj = [cc[f"{code}@p{p}"]["projective_rate"] for p in P_GRID]
        ax1.plot(P_GRID, petz, "-", marker="o", ms=5, color=col, lw=1.4,
                 label=lab)
        ax1.plot(P_GRID, proj, "--", marker="o", ms=5, mfc="white",
                 color=col, lw=1.1)
        ler = [ic[f"{code}@p{p}"]["ler"] for p in P_GRID]
        lo = [ic[f"{code}@p{p}"]["wilson95"][0] for p in P_GRID]
        hi = [ic[f"{code}@p{p}"]["wilson95"][1] for p in P_GRID]
        ax2.errorbar(P_GRID, ler,
                     yerr=[np.subtract(ler, lo), np.subtract(hi, ler)],
                     marker="o", ms=5, color=col, lw=1.4, capsize=2.5,
                     label=lab)
    ax1.plot([], [], "-", marker="o", ms=5, color="0.4",
             label="Petz recovery (filled)")
    ax1.plot([], [], "--", marker="o", ms=5, mfc="white", color="0.4",
             label="projective decoder (open)")
    ax1.set_yscale("log")
    ax1.set_ylabel("calibrated-channel infidelity\nat matched "
                   "strength $s(p)$")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize=7.0, loc="lower right")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xticks(P_GRID, [str(p) for p in P_GRID])
    ax2.minorticks_off()
    smap = [cc[f"t1head@p{p}"]["s_match"] for p in P_GRID]
    top = ax1.twiny()
    top.set_xscale("log")
    top.set_xlim(ax1.get_xlim())
    top.set_xticks(list(P_GRID))
    top.set_xticklabels([f"{s:.2f}" for s in smap])
    top.minorticks_off()
    top.set_xlabel("matched strength $s(p)$", fontsize=9)
    ax2.set_xlabel("benchmark noise value $p$")
    ax2.set_ylabel("i.i.d. LER (lookup decoder,\nWilson 95%)")
    ax2.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_headline_pairs_w8.png"))
    plt.close(fig)
    print("headline-pairs max joint z:",
          t2.get("diff_summary", {}).get("max_abs_z", "n/a"))
    return True


if __name__ == "__main__":
    ok = {}
    for name, fn in [("D1-dist", fig_dist),
                     ("D2-frontier", fig_frontier),
                     ("D3-traces", fig_traces),
                     ("C167-pairs", fig_headline_pairs)]:
        try:
            ok[name] = fn()
        except Exception as e:
            ok[name] = f"FAIL: {type(e).__name__}: {e}"
    print("render status:", ok)
