#!/usr/bin/env python3
"""Figure renders of the w7 manuscript pass (2026-08-13).

Reads committed result data under results/campaigns/ (read-only),
writes into results/figures/ only. Figures:

  fig_gates_w7.png    gate study: monitor traces (top, unchanged from w5)
                      + iid p-sweep (bottom left, unchanged) + NEW
                      calibrated-channel per-gate panel (bottom right):
                      exact rate vs strength s for each gate's
                      probe-selected code, projective solid + Petz dashed
                      (r12_scrutiny/t5_results.json figure_panel).
                      Replaces fig_gates_w5.png (cost bars move out).
  fig_gatecost_w7.png single-column encoder-cost bars (ansatz counts at
                      n = 6, L = 2), formerly the fig_gates_w5 bottom-right
                      panel; now its own figure near the resource text.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BP = os.path.join(REPO, "results", "campaigns")
R4 = os.path.join(BP, "r4_campaign")
R5 = os.path.join(BP, "r5_campaign")
R6 = os.path.join(BP, "r6_campaign")
R12 = os.path.join(BP, "r12_scrutiny")
FIG = os.path.join(REPO, "results", "figures")
os.makedirs(FIG, exist_ok=True)

# Paul Tol MUTED qualitative scheme (scripts/tol_colors.py); MS carries
# the darkened sand accent (#DDCC77 -> #B8A14C for lines on white).
GATE_C = {"ms": "#B8A14C", "zz": "#332288", "ls": "#44AA99",
          "csum": "#CC6677", "cex": "#AA4499"}
GATE_LAB = {"ms": "MS", "zz": "subspace-ZZ", "ls": "light-shift",
            "csum": "CSUM", "cex": "CEX"}
SEEDS12 = list(range(100, 112))

plt.rcParams.update({"font.size": 10, "axes.labelsize": 11,
                     "legend.fontsize": 8, "figure.dpi": 200})


def jload(*parts):
    with open(os.path.join(*parts)) as fh:
        return json.load(fh)


def eb(ax, xs, ys, los, his, resolved, label, color, marker="o", ls="-"):
    yerr = ([y - lo for y, lo in zip(ys, los)],
            [hi - y for y, hi in zip(ys, his)])
    ax.errorbar(xs, ys, yerr=yerr, color=color, marker=marker, ms=4.5,
                lw=1.3, ls=ls, capsize=2, label=label,
                markerfacecolor=color, zorder=3)
    for x, y, r in zip(xs, ys, resolved):
        if not r:
            ax.plot([x], [y], marker=marker, ms=4.5, mfc="white",
                    mec=color, zorder=4)


def fig_gates_w7():
    t1 = jload(R5, "t1_results.json")
    a1 = jload(R4, "a1_results.json")
    t5 = jload(R12, "t5_results.json")
    units = t1["units"]
    escp = os.path.join(R6, "esc_psweep.json")
    esc = jload(escp)["cells"] if os.path.exists(escp) else {}
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.4))
    ax1, ax2, ax3, ax4 = axes.flat
    for g, color in GATE_C.items():
        traces, opev_scale = [], []
        for s in SEEDS12:
            z = np.load(os.path.join(R5, "t1_codes", f"{g}_s{s}.npz"),
                        allow_pickle=True)
            ms_, ml_ = (np.asarray(z["monitor_steps"]),
                        np.asarray(z["monitor_losses"]))
            traces.append((ms_, ml_))
            opev_scale.append(float(np.asarray(z["race_op_evs"]))
                              / (ms_[-1] + 1))
        grid = traces[0][0]
        vals = np.stack([np.interp(grid, a, b) for a, b in traces])
        med = np.median(vals, axis=0)
        q1, q3 = (np.percentile(vals, 25, axis=0),
                  np.percentile(vals, 75, axis=0))
        sc = float(np.median(opev_scale))
        for ax, x in ((ax1, (grid + 1) * sc), (ax2, grid)):
            ax.plot(x, med, color=color, lw=1.5, label=GATE_LAB[g])
            ax.fill_between(x, q1, q3, color=color, alpha=0.18, lw=0)
    for ax in (ax1, ax2):
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_ylabel("channel-weighted loss (full batch)")
        ax.legend(fontsize=8)
    ax1.set_xscale("log")
    ax1.set_xlabel("cumulative op-EVs")
    ax2.set_xlabel("step")
    # bottom left: iid LER vs p (unchanged from w5)
    for g in GATE_C:
        cells = dict(a1["units"][f"psweep_{g}"]["cells"])
        if g in esc and esc[g].get("combined"):
            cb = esc[g]["combined"]
            cells[f"p{esc[g]['p']}"] = {
                "p": esc[g]["p"], "k": cb["k"], "n": cb["n"],
                "ler": cb["ler"], "wilson95": cb["wilson95"],
                "resolved": cb["resolved"]}
        cc = sorted(cells.values(), key=lambda c: c["p"])
        eb(ax3, [c["p"] for c in cc], [c["ler"] for c in cc],
           [c["wilson95"][0] for c in cc],
           [c["wilson95"][1] for c in cc],
           [c["resolved"] for c in cc], GATE_LAB[g], GATE_C[g])
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xticks([0.02, 0.03, 0.05, 0.08, 0.12, 0.15],
                   labels=["0.02", "0.03", "0.05", "0.08", "0.12",
                           "0.15"])
    ax3.minorticks_off()
    ax3.set_xlabel("physical error probability $p$")
    ax3.set_ylabel("i.i.d. logical error rate")
    ax3.grid(True, which="both", alpha=0.25)
    ax3.legend(fontsize=8, loc="lower right")
    # bottom right: exact calibrated per-gate panel
    fp = t5["figure_panel"]
    sgrid = fp["s_grid"]
    for g in GATE_C:
        cv = fp["curves"][g]
        ax4.plot(sgrid, cv["projective_rate"], "-", marker="o", ms=4.5,
                 color=GATE_C[g], lw=1.4, label=GATE_LAB[g])
        ax4.plot(sgrid, cv["petz_rate"], "--", marker="o", ms=4.5,
                 mfc="white", color=GATE_C[g], lw=1.2)
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xticks(sgrid, [str(s) for s in sgrid])
    ax4.minorticks_off()
    ax4.set_xlabel("channel strength $s$   (calibrated $s = 1$)")
    ax4.set_ylabel(r"exact channel infidelity $1 - F_\mathrm{avg}$")
    ax4.grid(True, which="both", alpha=0.25)
    ax4.legend(fontsize=8, loc="lower right", title="proj. solid / Petz dashed",
               title_fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_gates_w7.png"))
    plt.close(fig)
    # echo caption facts
    tally = {g: sum(1 for s in SEEDS12
                    if units[f"{g}_s{s}"]["ler_w2"]["k"] == 0)
             for g in GATE_C}
    print("gates: w2-perfect tallies:", tally)
    ms_petz = fp["curves"]["ms"]["petz_rate"]
    cex_petz = fp["curves"]["cex"]["petz_rate"]
    print("CEX/MS petz ratios per s:",
          [round(c / m, 2) for c, m in zip(cex_petz, ms_petz)])
    return True


def fig_gatecost_w7():
    gates2q = {"ms": 60, "zz": 60, "ls": 30, "csum": 30, "cex": 30}
    params = {"ms": 120, "zz": 60, "ls": 30, "csum": 0, "cex": 0}
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    xs = np.arange(len(GATE_C))
    b1 = ax.bar(xs - 0.19, [gates2q[g] for g in GATE_C], width=0.38,
                color=[GATE_C[g] for g in GATE_C], alpha=0.9,
                label="two-qudit gates")
    b2 = ax.bar(xs + 0.19, [params[g] for g in GATE_C], width=0.38,
                color=[GATE_C[g] for g in GATE_C], alpha=0.45,
                hatch="//", label="adjustable entangler params")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 2, str(int(h)),
                ha="center", fontsize=7.5)
    ax.set_xticks(xs, [GATE_LAB[g] for g in GATE_C], fontsize=9)
    ax.set_ylim(0, 138)
    ax.set_ylabel("count per encoder ($L = 2$, $|E| = 15$)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_gatecost_w7.png"))
    plt.close(fig)
    return True


if __name__ == "__main__":
    ok = fig_gates_w7() and fig_gatecost_w7()
    print("done:", ok)
