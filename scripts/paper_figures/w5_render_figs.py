#!/usr/bin/env python3
"""Figure renders of the w5 manuscript pass (2026-08-08).

Reads committed result data under results/campaigns/ (read-only),
writes into results/figures/ only. Figure list:

  fig_n_w5.png        F-n: register family; iid panel + correlated exact
                      panel (r2 t4 iid + fig6 measurements; r9 t2a).
  fig_q_w5.png        F-q: qudit dimension; iid reconstruction panel
                      (r5 t23, r6 presentation) + exact correlated cells
                      (r9 t2d) on the I_q axis.
  fig_dist_w5.png     F-dist: repeated-round per-round infidelity
                      (r9 t3_analysis) + single-round margin vs s (r9 t2b).
  fig_depth_w5.png    depth study: iid LER vs p per L + sampled loss
                      traces + physical-instrument rates (r9 t5, w5 t0).
  fig_gates_w5.png    gate study: monitor traces + iid p-sweep +
                      gate-cost bars (r5 t1_codes, r4 a1, r6 esc).
  fig_prune_w5.png    pruning ladder: iid LER vs p per gate count +
                      physical-instrument rates (r9 t5, w5 t0).
  fig_sampler_frontier_w5.png  R10 Fig-9 replacement (r10 t3_analysis,
                      shard JSONs; preset refs at committed totals from
                      sampling_budget_v2 a2_decomposition).
  fig_sampler_traces_w5.png    R10 Fig-10 replacement (r10 t1_codes npz).

Axis convention: per-gate per-qudit average infidelity;
iid side x = c_q p, channel side x = I(eta(s)) of the target-role
dephasing tier (q = 3 closed form validated against the 0.03459 anchor;
q = 4, 5 cells carry their stored I_q_axis).
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
R4 = os.path.join(BP, "r4_campaign")
R5 = os.path.join(BP, "r5_campaign")
R6 = os.path.join(BP, "r6_campaign")
R9 = os.path.join(BP, "r9_campaign")
R10 = os.path.join(BP, "r10_sampler")
T0 = os.path.join(BP, "w5_t0")
SBV2 = os.path.join(BP, "sampling_budget_v2")
FIG = os.path.join(REPO, "results", "figures")
os.makedirs(FIG, exist_ok=True)

SIGMA_P2 = 0.073
S_GRID = (0.125, 0.25, 0.5, 1.0, 1.5)
CQ = {3: 2.0 / 3.0, 4: 0.6, 5: 8.0 / 15.0}
# Paul Tol MUTED qualitative scheme (scripts/tol_colors.py); the sand
# accent is darkened (#DDCC77 -> #B8A14C) for lines/markers on white.
C = {"q3": "#B8A14C", "q4": "#117733", "q5": "#AA4499"}
N_COL = {5: "#B8A14C", 6: "#332288", 7: "#44AA99", 8: "#CC6677",
         9: "#882255"}
GATE_C = {"ms": "#B8A14C", "zz": "#332288", "ls": "#44AA99",
          "csum": "#CC6677", "cex": "#AA4499"}
GATE_LAB = {"ms": "MS", "zz": "subspace-ZZ", "ls": "light-shift",
            "csum": "CSUM", "cex": "CEX"}
SEEDS12 = list(range(100, 112))
NQ = 5

plt.rcParams.update({"font.size": 10, "axes.labelsize": 11,
                     "legend.fontsize": 8, "figure.dpi": 200})


def jload(*parts):
    with open(os.path.join(*parts)) as fh:
        return json.load(fh)


def I_eta_q3(eta):
    """Target-role dephasing average gate infidelity, q = 3:
    I(eta) = (3/4)[1 - (3 + 4 sqrt(eta) + 2 eta^2)/9]."""
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
    t2a = jload(R9, "t2a_results.json")["cells"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.4, 8.2))

    # (a) iid, unified axis x = c3 p
    pts = m["t1head"]["points"]
    live = [c for c in pts if not c.get("below_detection", False)]
    eb(ax1, [CQ[3] * c["p"] for c in live], [c["ler"] for c in live],
       [c["wilson_lo"] for c in live], [c["wilson_hi"] for c in live],
       [c["k"] >= 30 for c in live], "$n{=}5$ (headline, MS 40)",
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

    # (b) correlated exact, x = I(eta(s))
    xs = [I_of_s(s) for s in S_GRID]
    for n in (5, 6, 7, 8, 9):
        pr = [t2a[f"n{n}@s{s:.3f}"]["projective_rate"] for s in S_GRID]
        ax2.plot(xs, pr, "-", marker="o", ms=4.5, color=N_COL[n], lw=1.4,
                 label=f"$n{{=}}{n}$ projective")
        if n <= 7:
            pz = [t2a[f"n{n}@s{s:.3f}"]["petz_rate"] for s in S_GRID]
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
                   "  (channel v2)")
    ax2.set_ylabel(r"channel infidelity $1 - F_\mathrm{avg}$")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(loc="lower right", fontsize=7.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_n_w5.png"))
    plt.close(fig)
    # provenance echoes
    mono = jload(R9, "t2a_results.json")["summary"]
    print("F-n: monotone booleans:", {k: v for k, v in mono.items()
                                      if "monotone" in str(k).lower()}
          if isinstance(mono, dict) else mono)
    return True


# ---------------------------------------------------------------- F-q
def _pw(p, w):
    return math.comb(NQ, w) * p ** w * (1 - p) ** (NQ - w)


def _reconstruct(qp, q, ps):
    lo, pt, hi = [], [], []
    for p in ps:
        s_lo = s_pt = s_hi = 0.0
        for w in range(1, NQ + 1):
            c = qp[f"q{q}_w{w}"]
            s_lo += _pw(p, w) * c["wilson95"][0]
            s_pt += _pw(p, w) * c["ler"]
            s_hi += _pw(p, w) * c["wilson95"][1]
        lo.append(s_lo)
        pt.append(s_pt)
        hi.append(s_hi)
    return lo, pt, hi


def fig_q():
    m = jload(BP, "fig6_panels", "measurements.json")["codes"]
    qp = jload(R5, "t23_results.json")["t2_qpanel"]
    t2d = jload(R9, "t2d_results.json")["cells"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.4, 8.2))

    # (a) iid with reconstruction (r6 presentation, unified axis)
    pts = m["t1head"]["points"]
    live = [c for c in pts if not c.get("below_detection", False)]
    eb(ax1, [CQ[3] * c["p"] for c in live], [c["ler"] for c in live],
       [c["wilson_lo"] for c in live], [c["wilson_hi"] for c in live],
       [c["k"] >= 30 for c in live],
       "$((5,3,3))_3$ (MS 40), direct MC", C["q3"])
    ps = sorted(float(p) for p in qp["q4_reconstruction"])
    lo4, pt4, hi4 = _reconstruct(qp, 4, ps)
    x4 = [CQ[4] * p for p in ps]
    ax1.plot(x4, pt4, "--", marker="D", mfc="white", ms=4.5,
             color=C["q4"], lw=1.2,
             label="$((5,3,3))_4$ (MS 60) reconstruction\n"
                   "(band: sector Wilson 95%)")
    ax1.fill_between(x4, [max(v, 1e-12) for v in lo4], hi4,
                     color=C["q4"], alpha=0.18, lw=0)
    anch = sorted((v for k, v in qp.items()
                   if k.startswith("q4_anchor")), key=lambda v: v["p"])
    eb(ax1, [CQ[4] * v["p"] for v in anch], [v["ler"] for v in anch],
       [v["wilson95"][0] for v in anch], [v["wilson95"][1] for v in anch],
       [v["k"] >= 30 for v in anch],
       "$((5,3,3))_4$ direct anchors ($10^7$)", C["q4"], marker="o")
    lo5, pt5, hi5 = _reconstruct(qp, 5, ps)
    x5 = [CQ[5] * p for p in ps]
    floor = 2e-10
    ax1.plot(x5, hi5, "-", marker="v", mfc="white", ms=5, color=C["q5"],
             lw=1.2,
             label="$((5,3,3))_5$ (MS 80) bounded band\n"
                   "(all sectors 0 errors; 0 up to Wilson UB)")
    ax1.fill_between(x5, [floor] * len(x5), hi5, color=C["q5"],
                     alpha=0.15, lw=0, hatch="//", edgecolor=C["q5"])
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylim(floor, None)
    ax1.set_xlabel(r"per-qudit per-gate infidelity $c_q\,p$  (i.i.d.)")
    ax1.set_ylabel("i.i.d. logical error rate")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01),
               ncol=2, fontsize=6.8, frameon=False)

    # (b) exact correlated cells on the I_q axis
    codes = {"q3": ("headline", 3, C["q3"], "$q{=}3$ (central code, MS 40)"),
             "q4": ("q4_L2_s100", 4, C["q4"], "$q{=}4$ (MS 60)"),
             "q5": ("q5_L2_s100", 5, C["q5"], "$q{=}5$ (MS 80)")}
    keymap = {}
    for key in t2d:
        keymap.setdefault(key.split("@")[0], []).append(key)
    name_by_q = {}
    for base, keys in keymap.items():
        qv = t2d[keys[0]]["q"]
        name_by_q[qv] = base
    for tag, (_, qv, color, lab) in codes.items():
        base = name_by_q[qv]
        cells = sorted((t2d[f"{base}@s{s:.3f}"] for s in S_GRID),
                       key=lambda c: c["s"])
        xs = [c["I_q_axis"]["target"] for c in cells]
        ax2.plot(xs, [c["projective_rate"] for c in cells], "-",
                 marker="o", ms=4.5, color=color, lw=1.4,
                 label=lab + ", projective")
        ax2.plot(xs, [c["petz_rate"] for c in cells], "--", marker="o",
                 ms=4.5, mfc="white", color=color, lw=1.1)
    ax2.plot([], [], "--", marker="o", ms=4.5, mfc="white", color="0.3",
             label="Petz bound (exact)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"per-qudit per-gate infidelity $I_q(\eta(s))$"
                   "  (channel v2)")
    ax2.set_ylabel(r"channel infidelity $1 - F_\mathrm{avg}$")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(loc="lower right", fontsize=7.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q_w5.png"))
    plt.close(fig)
    print("F-q: q codes:", name_by_q)
    return True


# ------------------------------------------------------------- F-dist
def fig_dist():
    t3 = jload(R9, "t3_analysis.json")["cells"]
    t2b = jload(R9, "t2b_results.json")["cells"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.4, 7.6))

    # (a) per-round infidelity 1 - lambda at s = 0.25
    rows = [("d2", "$((7,3,2))_3$", "#c44e52"),
            ("d3", "$((7,3,3))_3$", "#dd8452"),
            ("d4", "$((7,3,4))_3$", "#937860"),
            ("headline", "headline $n{=}5$", N_COL[5]),
            ("five_one_three_Z3", "$[[5,1,3]]_{Z_3}$", "0.25")]
    vals, labs, cols = [], [], []
    for key, lab, col in rows:
        c = t3[f"{key}@s0.25"]
        vals.append(c["per_round_infid_1mlambda"])
        labs.append(lab)
        cols.append(col)
    xs = np.array([0, 1, 2, 3.4, 4.4])
    bars = ax1.bar(xs, vals, width=0.8, color=cols, alpha=0.85)
    for x, v in zip(xs, vals):
        ax1.text(x, v + 0.004, f"{v:.4f}", ha="center", fontsize=7.5)
    ax1.axvline(2.85, color="0.7", lw=0.8, ls=":")
    ax1.text(1.0, 0.315, "$n{=}7$ trio (21 gate steps/round)",
             fontsize=7.5, ha="center", color="0.3")
    ax1.text(3.9, 0.315, "$n{=}5$ (10 steps)", fontsize=7.5,
             ha="center", color="0.3")
    ax1.set_xticks(xs, labs, fontsize=7.5, rotation=12)
    ax1.set_ylim(0, 0.34)
    ax1.set_ylabel(r"per-round infidelity $1 - \lambda$")
    ax1.grid(True, axis="y", alpha=0.25)

    # (b) single-round exact separation vs s
    for inst, lw, alpha, suffix in (("petz_rate", 1.5, 1.0, "Petz"),
                                    ("projective_rate", 1.1, 0.55,
                                     "projective")):
        r_d2 = np.array([t2b[f"d2@s{s:.3f}"][inst] for s in S_GRID])
        r_d3 = np.array([t2b[f"d3@s{s:.3f}"][inst] for s in S_GRID])
        r_d4 = np.array([t2b[f"d4@s{s:.3f}"][inst] for s in S_GRID])
        ax2.plot(S_GRID, r_d2 / r_d4, "-", marker="o", ms=4.5,
                 color="#c44e52", lw=lw, alpha=alpha,
                 label=f"$d{{=}}2$ / $d{{=}}4$, {suffix}")
        ax2.plot(S_GRID, r_d3 / r_d4, "-", marker="s", ms=4,
                 color="#dd8452", lw=lw, alpha=alpha,
                 label=f"$d{{=}}3$ / $d{{=}}4$, {suffix}")
    ax2.axhline(1.0, color="0.5", lw=0.8)
    ax2.set_xscale("log")
    ax2.set_xticks(S_GRID, [str(s) for s in S_GRID])
    ax2.minorticks_off()
    ax2.set_xlabel("channel strength $s$   (calibrated $s = 1$)")
    ax2.set_ylabel("single-round rate ratio vs $d{=}4$")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(loc="upper right", fontsize=7.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_dist_w5.png"))
    plt.close(fig)
    print("F-dist: petz d2/d4 margins:",
          [round(t2b[f'd2@s{s:.3f}']['petz_rate']
                 / t2b[f'd4@s{s:.3f}']['petz_rate'], 4) for s in S_GRID])
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
    t0 = jload(T0, "t0_results.json")["summary"]["depth"]
    LS = (2, 4, 8, 16)
    LCOL = {2: "#12355b", 4: "#1f77b4", 8: "#e0a010", 16: "#d62728"}
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5.4, 9.2))

    # (a) iid LER vs p, seed median + range per L
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

    # (b) sampled training loss vs step (median over 4 seeds, smoothed)
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

    # (c) physical instrument (w5 t0): rate vs L, both s, both instruments
    for s, mk in ((0.5, "o"), (1.0, "s")):
        sk = f"s{s:.3f}"
        for inst, ls_, lab in (("petz", "--", "Petz"),
                               ("proj", "-", "projective")):
            med = [t0[sk]["per_L"][f"L{L}"][f"{inst}_median"] for L in LS]
            lo = [t0[sk]["per_L"][f"L{L}"][f"{inst}_range"][0] for L in LS]
            hi = [t0[sk]["per_L"][f"L{L}"][f"{inst}_range"][1] for L in LS]
            ax3.errorbar(LS, med,
                         yerr=[np.subtract(med, lo), np.subtract(hi, med)],
                         marker=mk, ms=5, ls=ls_, lw=1.3, capsize=3,
                         color="#6a51a3" if s == 0.5 else "#238b45",
                         mfc=("white" if inst == "petz" else None),
                         label=f"$s{{=}}{s}$, {lab}")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xticks(LS, [str(L) for L in LS])
    ax3.minorticks_off()
    ax3.set_ylim(0.03, 0.55)
    ax3.set_xlabel("ansatz depth $L$")
    ax3.set_ylabel(r"channel infidelity $1 - F_\mathrm{avg}$"
                   "\n(4-seed median + range)")
    ax3.grid(True, which="both", alpha=0.25)
    ax3.legend(loc="lower center", fontsize=6.8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_depth_w5.png"))
    plt.close(fig)
    return True


# -------------------------------------------------------------- gates
def fig_gates():
    t1 = jload(R5, "t1_results.json")
    a1 = jload(R4, "a1_results.json")
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
    # bottom left: iid LER vs p
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
    # bottom right: encoder-cost bars (ansatz counts at n = 6, L = 2)
    gates2q = {"ms": 60, "zz": 60, "ls": 30, "csum": 30, "cex": 30}
    params = {"ms": 120, "zz": 60, "ls": 30, "csum": 0, "cex": 0}
    xs = np.arange(len(GATE_C))
    b1 = ax4.bar(xs - 0.19, [gates2q[g] for g in GATE_C], width=0.38,
                 color=[GATE_C[g] for g in GATE_C], alpha=0.9,
                 label="two-qudit gates")
    b2 = ax4.bar(xs + 0.19, [params[g] for g in GATE_C], width=0.38,
                 color=[GATE_C[g] for g in GATE_C], alpha=0.45,
                 hatch="//", label="trainable entangler params")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, h + 2, str(int(h)),
                 ha="center", fontsize=7.5)
    ax4.set_xticks(xs, [GATE_LAB[g] for g in GATE_C], fontsize=9)
    ax4.set_ylim(0, 138)
    ax4.set_ylabel("count per encoder ($L = 2$, $|E| = 15$)")
    ax4.grid(True, axis="y", alpha=0.25)
    ax4.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_gates_w5.png"))
    plt.close(fig)
    # caption tallies: w2-perfect seeds per gate
    tally = {}
    for g in GATE_C:
        tally[g] = sum(1 for s in SEEDS12
                       if units[f"{g}_s{s}"]["ler_w2"]["k"] == 0)
    print("gates: w2-perfect tallies:", tally)
    return True


# ------------------------------------------------------------ pruning
def fig_prune():
    t5 = jload(R9, "t5_results.json")["cells"]
    t0 = jload(T0, "t0_results.json")["summary"]["prune"]
    GC = {40: "#12355b", 16: "#1f77b4", 12: "#e0a010", 8: "#d62728"}
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
    # (b) physical instrument
    gates = (40, 16, 12, 8)
    for s, col in ((0.5, "#6a51a3"), (1.0, "#238b45")):
        sk = f"s{s:.3f}"
        pz = [t0[sk][f"gates{g}"]["petz_rate"] for g in gates]
        pj = [t0[sk][f"gates{g}"]["projective_rate"] for g in gates]
        ax2.plot(gates, pj, "-", marker="o", ms=5, color=col, lw=1.4,
                 label=f"$s{{=}}{s}$, projective")
        ax2.plot(gates, pz, "--", marker="o", ms=5, mfc="white",
                 color=col, lw=1.2, label=f"$s{{=}}{s}$, Petz")
    ax2.set_yscale("log")
    ax2.set_ylim(4e-2, 8e-1)
    ax2.set_xticks(gates, [str(g) for g in gates])
    ax2.invert_xaxis()
    ax2.set_xlabel("surviving entangling gates")
    ax2.set_ylabel(r"channel infidelity $1 - F_\mathrm{avg}$")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(loc="lower left", bbox_to_anchor=(0.03, 0.20),
               fontsize=6.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_prune_w5.png"))
    plt.close(fig)
    return True


# --------------------------------------------------------- frontier
CELL_LAB = {"full_batch@1.0": "full batch (certified)",
            "importance@0.4": "importance $f{=}0.4$",
            "importance@0.05": "importance $f{=}0.05$",
            "full_basis_is@0.4": "fb-importance $f{=}0.4$",
            "full_basis_is@0.05": "fb-importance $f{=}0.05$"}
PRESET_MAP = {"existing_full_batch": "full_batch",
              "existing_stratified_0.2": "stratified_0.2",
              "existing_importance_0.2": "importance_0.2",
              "existing_importance_0.35": "importance_0.35",
              "existing_full_basis_is_0.1": "full_basis_is_0.1"}
CELL_COL = {"full_batch@1.0": "#12355b", "importance@0.4": "#1f77b4",
            "importance@0.05": "#2ca02c",
            "full_basis_is@0.4": "#e0a010",
            "full_basis_is@0.05": "#d62728"}


def fig_frontier():
    fr = jload(R10, "t3_analysis.json")["frontier"]
    a2 = jload(SBV2, "a2_decomposition.json")["samplers"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.4, 7.8),
                                   sharex=True)
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
        for r in c["corr_rates"]:
            ax2.plot([x], [r], marker=".", ms=3.5,
                     color=CELL_COL[cell], zorder=5)
    for ex, preset in PRESET_MAP.items():
        c = fr[ex]
        x = a2[preset]["committed_total_op_evs"]
        ler = c["iid_pooled"]["ler"]
        w = c["iid_pooled"]["wilson95"]
        ax1.errorbar([x], [ler], yerr=[[ler - w[0]], [w[1] - ler]],
                     marker="s", ms=6, mfc="white", color="0.4",
                     capsize=2.5, lw=1.0, zorder=3)
        ax2.errorbar([x], [c["corr_mean"]],
                     yerr=[[c["corr_est_se"]], [c["corr_est_se"]]],
                     marker="s", ms=6, mfc="white", color="0.4",
                     capsize=2.5, lw=1.0, zorder=3)
    ax1.errorbar([], [], marker="s", ms=6, mfc="white", color="0.4",
                 label="preset-1500 published protocol\n(1 seed, "
                       "totals incl. monitor)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylabel("i.i.d. LER at $p{=}0.05$\n(4-seed pool, Wilson 95%)")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01),
               ncol=2, fontsize=6.6, frameon=False)
    ax2.set_ylabel("channel-v2 projective rate, $s{=}0.5$\n"
                   "(mean $\\pm$ estimator SE; dots: seeds)")
    ax2.set_xlabel("certified training budget (train + confirmation "
                   "op-EVs)")
    ax2.set_ylim(0.6535, 0.658)
    ax2.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_sampler_frontier_w5.png"))
    plt.close(fig)
    v = jload(R10, "t3_analysis.json")["verdict"]
    print("frontier verdict keys:", list(v.keys())
          if isinstance(v, dict) else str(v)[:200])
    return True


def fig_traces():
    order = [("full_batch_s100", "full batch", "#12355b"),
             ("importance_0.4_s100", "importance $f{=}0.4$", "#1f77b4"),
             ("importance_0.05_s100", "importance $f{=}0.05$",
              "#2ca02c"),
             ("full_basis_is_0.4_s100", "fb-importance $f{=}0.4$",
              "#e0a010"),
             ("full_basis_is_0.05_s100", "fb-importance $f{=}0.05$",
              "#d62728")]
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    tau = None
    for fname, lab, col in order:
        p = os.path.join(R10, "t1_codes", f"{fname}.npz")
        if not os.path.exists(p):
            # importance cells may be named without the frac dot
            alt = fname.replace("0.4", "04").replace("0.05", "005")
            p = os.path.join(R10, "t1_codes", f"{alt}.npz")
        z = np.load(p, allow_pickle=True)
        tau = float(z["tau"])
        cum = np.asarray(z["cum_train_evs"], float)
        sl = np.asarray(z["sampled_losses"], float)
        ax.plot(cum, sl, color=col, lw=0.9, alpha=0.5)
        msteps = np.asarray(z["monitor_steps"], int)
        mloss = np.asarray(z["monitor_losses"], float)
        mx = np.interp(msteps, np.arange(len(cum)), cum)
        ax.plot(mx, mloss, marker="o", ms=5, ls="none", color=col,
                label=lab, zorder=4)
        stop = int(z["stop_step"])
        ax.plot([cum[min(stop, len(cum) - 1)]],
                [sl[min(stop, len(sl) - 1)]], marker="*", ms=11,
                color=col, mec="k", mew=0.4, zorder=5)
    ax.axhline(tau, color="0.4", lw=0.9, ls=":")
    ax.text(52, tau * 1.03, r"$\tau = 10^{-4}$", fontsize=8,
            color="0.3", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cumulative training op-EVs")
    ax.set_ylabel("channel-weighted loss")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_sampler_traces_w5.png"))
    plt.close(fig)
    return True


if __name__ == "__main__":
    ok = {}
    for name, fn in [("F-n", fig_n), ("F-q", fig_q),
                     ("F-dist", fig_dist), ("depth", fig_depth),
                     ("gates", fig_gates), ("prune", fig_prune),
                     ("frontier", fig_frontier), ("traces", fig_traces)]:
        try:
            ok[name] = fn()
        except Exception as e:
            ok[name] = f"FAIL: {type(e).__name__}: {e}"
    print("render status:", ok)
