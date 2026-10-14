#!/usr/bin/env python3
"""Figure renders of the w6 manuscript pass (2026-08-11).

Reads committed result data under results/campaigns/ (read-only),
writes into results/figures/ only. Figures rebuilt in that pass:

  fig_n_w6.png     F-n: register family; iid panel (unchanged data) +
                   exact calibrated-channel panel from
                   r11_gapfill/t1a_family_L2.json (n5 member = the L=2
                   flagship code; deep-16L member retired).
  fig_dist_w6.png  F-dist (D4): per-round infidelity 1-lambda vs s,
                   log-log, one curve per code (n=7 trio + n=5 flagship
                   + [[5,1,3]]), projective solid; Petz dashed for the
                   two n=5 references only; from r11_gapfill/
                   t4_analysis.json. n=7 cells at s=1.5 are saturation-
                   dominated (F_e at the 1/9 floor in round 1) and drawn
                   open.
  fig_depth_w6.png depth study (C11): iid LER vs p per L + sampled loss
                   traces + physical rate-vs-s per L (full s grid) from
                   r11_gapfill/t5_sgrid_results.json.
  fig_prune_w6.png pruning ladder (C11): iid LER vs p per gate count +
                   physical rate-vs-s per gate count (full s grid) from
                   r11_gapfill/t5_sgrid_results.json.

Axis convention: per-gate per-qudit average infidelity;
iid side x = c_q p, channel side x = I(eta(s)) of the target-role tier.
"""
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BP = os.path.join(REPO, "results", "campaigns")
R2 = os.path.join(BP, "r2_campaign")
R9 = os.path.join(BP, "r9_campaign")
R11 = os.path.join(BP, "r11_gapfill")
FIG = os.path.join(REPO, "results", "figures")
os.makedirs(FIG, exist_ok=True)

SIGMA_P2 = 0.073
S_GRID = (0.125, 0.25, 0.5, 1.0, 1.5)
CQ = {3: 2.0 / 3.0}
# Paul Tol MUTED qualitative scheme (scripts/tol_colors.py); the sand
# accent is darkened (#DDCC77 -> #B8A14C) for lines/markers on white.
N_COL = {5: "#B8A14C", 6: "#332288", 7: "#44AA99", 8: "#CC6677",
         9: "#882255"}

plt.rcParams.update({"font.size": 10, "axes.labelsize": 11,
                     "legend.fontsize": 8, "figure.dpi": 200})


def jload(*parts):
    with open(os.path.join(*parts)) as fh:
        return json.load(fh)


def I_eta_q3(eta):
    return 0.75 * (1.0 - (3.0 + 4.0 * math.sqrt(eta) + 2.0 * eta ** 2)
                   / 9.0)


def I_of_s(s):
    return I_eta_q3(math.exp(-s * SIGMA_P2))


assert abs(I_of_s(1.0) - 0.03459) < 5e-5, I_of_s(1.0)


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


# ---------------------------------------------------------------- F-n
def fig_n():
    t4 = jload(R2, "t4_results.json")["iid"]
    m = jload(BP, "fig6_panels", "measurements.json")["codes"]
    fam = jload(R11, "t1a_family_L2.json")["cells"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.4, 8.2))

    # (a) iid, unified axis x = c3 p (data unchanged from the W5 render)
    pts = m["t1head"]["points"]
    live = [c for c in pts if not c.get("below_detection", False)]
    eb(ax1, [CQ[3] * c["p"] for c in live], [c["ler"] for c in live],
       [c["wilson_lo"] for c in live], [c["wilson_hi"] for c in live],
       [c["k"] >= 30 for c in live], "$n{=}5$ (central code, MS 40)",
       N_COL[5])
    for n, ms in ((6, 60), (7, 84), (8, 112), (9, 144)):
        cells = sorted((v for k, v in t4.items()
                        if k.startswith(f"iid_n{n}@")),
                       key=lambda c: c["p"])
        cells = [c for c in cells if not c.get("below_detection", False)]
        eb(ax1, [CQ[3] * c["p"] for c in cells],
           [c["ler"] for c in cells],
           [c["wilson95"][0] for c in cells],
           [c["wilson95"][1] for c in cells],
           [c["resolved"] for c in cells],
           f"$n{{=}}{n}$ (MS {ms})", N_COL[n])
    sp = [c for c in m["s513"]["points"]
          if not c.get("below_detection", False)]
    eb(ax1, [CQ[3] * c["p"] for c in sp], [c["ler"] for c in sp],
       [c["wilson_lo"] for c in sp], [c["wilson_hi"] for c in sp],
       [c["k"] >= 30 for c in sp], "$[[5,1,3]]_{Z_3}$ (MS 36)",
       "0.25", marker="s", ls="--")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"per-qudit per-gate infidelity $c_q\,p$  (i.i.d.)")
    ax1.set_ylabel("i.i.d. logical error rate")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(loc="lower right", fontsize=7.2)

    # (b) exact calibrated channel; n5 member = the central code, L=2
    xs = [I_of_s(s) for s in S_GRID]
    for n in (5, 6, 7, 8, 9):
        pr = [fam[f"n{n}@s{s:.3f}"]["projective_rate"] for s in S_GRID]
        ax2.plot(xs, pr, "-", marker="o", ms=4.5, color=N_COL[n], lw=1.4,
                 label=f"$n{{=}}{n}$ projective")
        if n <= 7:
            pz = [fam[f"n{n}@s{s:.3f}"]["petz_rate"] for s in S_GRID]
            ax2.plot(xs, pz, "--", marker="o", ms=4.5, mfc="white",
                     color=N_COL[n], lw=1.1)
    ax2.plot([], [], "--", marker="o", ms=4.5, mfc="white", color="0.3",
             label="Petz bound ($n \\leq 7$, exact)")
    ax2.axvline(I_of_s(1.0), color="0.6", lw=0.8, ls=":")
    ax2.text(I_of_s(1.0) * 1.05, 4.5e-3, "calibrated\n$s{=}1$",
             fontsize=7, color="0.35")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"per-qudit per-gate infidelity $I(\eta(s))$"
                   "  (calibrated channel)")
    ax2.set_ylabel(r"channel infidelity $1 - F_\mathrm{avg}$")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(loc="lower right", fontsize=7.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_n_w6.png"))
    plt.close(fig)
    mono = jload(R11, "t1a_family_L2.json")["monotone_booleans"]
    print("F-n booleans:", {k: all(v.values()) for k, v in mono.items()})
    return True


# ------------------------------------------------------------- F-dist
def fig_dist():
    t4 = jload(R11, "t4_analysis.json")
    proj = t4["lambda_vs_s_projective"]
    petz = t4["lambda_vs_s_petz"]
    skeys = ["s0.125", "s0.25", "s0.5", "s1.0", "s1.5"]
    svals = S_GRID
    rows = [("d2", "$((7,3,2))_3$", "#CC6677"),
            ("d3", "$((7,3,3))_3$", "#882255"),
            ("d4", "$((7,3,4))_3$", "#AA4499"),
            ("headline", "$((5,3,3))_3$ headline", N_COL[5]),
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
    for key, lab, col in rows:
        if key not in ("headline", "five_one_three_Z3"):
            continue
        pk = petz[key]
        ss = sorted(pk.keys(), key=lambda k: float(k[1:]))
        xs = [float(k[1:]) for k in ss]
        ys = [pk[k]["per_round_infid"] for k in ss]
        mk = "s" if key == "five_one_three_Z3" else "o"
        ax.plot(xs, ys, "--", marker=mk, ms=4.5, mfc="white", color=col,
                lw=1.1)
    ax.plot([], [], "--", marker="o", ms=4.5, mfc="white", color="0.3",
            label="Petz bound ($n{=}5$ references)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(svals, [str(s) for s in svals])
    ax.minorticks_off()
    ax.set_xlabel("channel strength $s$   (calibrated $s = 1$)")
    ax.set_ylabel(r"per-round infidelity $1 - \lambda$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=7.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_dist_w6.png"))
    plt.close(fig)
    print("F-dist d2/d4 per-round margins:",
          [round(t4["trio_ordering_by_s"][sk]
                 ["per_round_ratio_d2_over_d4"], 4) for sk in
           ("s0.125", "s0.25", "s0.5", "s1.0", "s1.5")])
    return True


# -------------------------------------------------------------- depth
def _runmed(x, w=51):
    x = np.asarray(x, float)
    out = np.empty_like(x)
    h = w // 2
    for i in range(len(x)):
        out[i] = np.median(x[max(0, i - h):i + h + 1])
    return out


def fig_depth():
    t5 = jload(R9, "t5_results.json")["cells"]
    tr = np.load(os.path.join(R9, "t5_loss_traces.npz"))
    sg = jload(R11, "t5_sgrid_results.json")["cells"]
    LS = (2, 4, 8, 16)
    LCOL = {2: "#B8A14C", 4: "#332288", 8: "#44AA99", 16: "#CC6677"}
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5.4, 9.2))

    pgrid = (0.02, 0.03, 0.05, 0.08, 0.12, 0.15)
    for L in LS:
        med, lo, hi = [], [], []
        for p in pgrid:
            v = [t5[f"depth_L{L}_s{s}@p{p:.2f}"]["ler"]
                 for s in (100, 101, 102, 103)]
            med.append(np.median(v))
            lo.append(min(v))
            hi.append(max(v))
        ax1.plot(pgrid, med, "-", marker="o", ms=4.5, color=LCOL[L],
                 lw=1.4, label=f"$L{{=}}{L}$ (MS {20 * L})")
        ax1.fill_between(pgrid, lo, hi, color=LCOL[L], alpha=0.15, lw=0)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xticks(pgrid, [str(p) for p in pgrid])
    ax1.minorticks_off()
    ax1.set_xlabel("physical error probability $p$")
    ax1.set_ylabel("i.i.d. logical error rate")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(loc="upper left", fontsize=7.5)

    for L in LS:
        traces = [np.asarray(tr[f"depth_L{L}_s{s}"], float)
                  for s in (100, 101, 102, 103)]
        nmin = min(len(t) for t in traces)
        med = np.median(np.stack([_runmed(t[:nmin]) for t in traces]),
                        axis=0)
        ax2.plot(np.arange(nmin), med, color=LCOL[L], lw=1.3,
                 label=f"$L{{=}}{L}$")
    ax2.set_yscale("log")
    ax2.set_ylim(2.0e-4, 2.4e-2)
    ax2.set_xlabel("training step")
    ax2.set_ylabel("sampled loss (median of 4 seeds,\nwindow-51 median)")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(loc="upper right", fontsize=7.5, ncol=2)

    # (c) physical rate vs s per L, full grid
    for L in LS:
        for inst, ls_, mfc in (("projective_rate", "-", None),
                               ("petz_rate", "--", "white")):
            med, lo, hi = [], [], []
            for s in S_GRID:
                v = [sg[f"depth_L{L}_s{sd}@s{s:.3f}"][inst]
                     for sd in (100, 101, 102, 103)]
                med.append(np.median(v))
                lo.append(min(v))
                hi.append(max(v))
            ax3.errorbar(S_GRID, med,
                         yerr=[np.subtract(med, lo),
                               np.subtract(hi, med)],
                         marker="o", ms=4, ls=ls_, lw=1.2, capsize=2,
                         color=LCOL[L], mfc=mfc,
                         label=(f"$L{{=}}{L}$" if inst ==
                                "projective_rate" else None))
    ax3.plot([], [], "--", marker="o", ms=4, mfc="white", color="0.3",
             label="Petz bound (dashed)")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xticks(S_GRID, [str(s) for s in S_GRID])
    ax3.minorticks_off()
    ax3.set_xlabel("channel strength $s$   (calibrated $s = 1$)")
    ax3.set_ylabel(r"channel infidelity $1 - F_\mathrm{avg}$"
                   "\n(4-seed median + range)")
    ax3.grid(True, which="both", alpha=0.25)
    ax3.legend(loc="lower right", fontsize=6.8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_depth_w6.png"))
    plt.close(fig)
    return True


# -------------------------------------------------------------- prune
def fig_prune():
    t5 = jload(R9, "t5_results.json")["cells"]
    sg = jload(R11, "t5_sgrid_results.json")["cells"]
    GC = {40: "#B8A14C", 16: "#332288", 12: "#44AA99", 8: "#CC6677"}
    pgrid = (0.02, 0.03, 0.05, 0.08, 0.12, 0.15)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.4, 7.6))
    for g in (40, 16, 12, 8):
        cells = [t5[f"prune_gates{g}@p{p:.2f}"] for p in pgrid]
        eb(ax1, [c["p"] for c in cells], [c["ler"] for c in cells],
           [c["wilson95"][0] for c in cells],
           [c["wilson95"][1] for c in cells],
           [c["resolved"] for c in cells],
           f"{g} gates" + (" (unpruned)" if g == 40 else ""), GC[g])
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xticks(pgrid, [str(p) for p in pgrid])
    ax1.minorticks_off()
    ax1.set_xlabel("physical error probability $p$")
    ax1.set_ylabel("i.i.d. logical error rate")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(loc="upper left", fontsize=7.5)

    # (b) physical rate vs s per gate count, full grid
    for g in (40, 16, 12, 8):
        pj = [sg[f"prune_gates{g}@s{s:.3f}"]["projective_rate"]
              for s in S_GRID]
        pz = [sg[f"prune_gates{g}@s{s:.3f}"]["petz_rate"]
              for s in S_GRID]
        ax2.plot(S_GRID, pj, "-", marker="o", ms=4.5, color=GC[g],
                 lw=1.4, label=f"{g} gates")
        ax2.plot(S_GRID, pz, "--", marker="o", ms=4.5, mfc="white",
                 color=GC[g], lw=1.1)
    ax2.plot([], [], "--", marker="o", ms=4.5, mfc="white", color="0.3",
             label="Petz bound (dashed)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xticks(S_GRID, [str(s) for s in S_GRID])
    ax2.minorticks_off()
    ax2.set_xlabel("channel strength $s$   (calibrated $s = 1$)")
    ax2.set_ylabel(r"channel infidelity $1 - F_\mathrm{avg}$")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(loc="lower right", fontsize=7.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_prune_w6.png"))
    plt.close(fig)
    return True


if __name__ == "__main__":
    ok = {"fig_n": fig_n(), "fig_dist": fig_dist(),
          "fig_depth": fig_depth(), "fig_prune": fig_prune()}
    print("rendered:", ok)
