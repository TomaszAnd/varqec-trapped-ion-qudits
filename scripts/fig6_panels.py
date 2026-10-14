#!/usr/bin/env python3
"""Four one-parameter i.i.d. panels (register size, K, q, distance); the measurements behind Figs. 3 and 4.

Each panel varies EXACTLY ONE of n, K, q, dist and holds the others fixed:
  panel_n    ((n,3,3))_3  n=5..9   campaign codes (L=16 a2a MS, uniform protocol)
  panel_K    ((5,K,3))_3  K=2,3    (K=3 slot: campaign n5; singleton bound caps K<=3)
  panel_q    ((5,3,3))_q  q=3,4,5  (K=3 held fixed -- fig6_panel_train codes)
  panel_dist ((5,3,d))_3  d=2,3
The published ((5,3,3))_3 (L=10 ring, MS budget 100 -- the P2-selected headline) and
the analytic [[5,1,3]]_Z3 (= ((5,3,3))_3) are valid points of every panel's fixed
slice and are drawn as references wherever comparable (handoff: [[5,1,3]] only where
its (n,K,q,d) is comparable).

Measurement: i.i.d. per-qudit hardware error + batched factored lookup decoder
(batched lookup decoder), shots >= 300k with escalation 300k -> 1M -> 3M until a cell
has >= 30 logical errors (n=9 capped at 1M -- 17 min/1M on this box). Cells that
stay at 0 errors are flagged below_detection and carry the Wilson upper bound --
NEVER reported as zero. Every code carries its MS budget
L*|E|*(q-1); every point states p. Wilson 95% CIs.

Measurements are cached per-code in fig6_panels/measurements.json (codes are shared
across panels); the four PNGs render from that JSON only -- scripts/make_figures.py
imports render_all_panels() so a data-complete run can never be figure-empty.

Usage:
  fig6_panels.py            # measure whatever code sources exist, then render
  fig6_panels.py --render   # render-only from committed JSON (no recompute)
"""
from __future__ import annotations
import os, sys, json, math, time, argparse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BPR = os.path.join(REPO, "results", "campaigns")
OUTDIR = os.path.join(BPR, "fig6_panels")
MEAS = os.path.join(OUTDIR, "measurements.json")

PS = [0.02, 0.03, 0.05, 0.08, 0.12, 0.15]
LADDER = [300_000, 1_000_000, 3_000_000]
MIN_ERRORS = 30  # escalate until this many logical errors (or ladder cap)

# ---- code registry ---------------------------------------------------------
# ms = MS budget L*|E|*(q-1); None for analytic states (encoder not simulated).
def _a2a_ms(L, n, q):
    return L * (n * (n - 1) // 2) * (q - 1)

CODES = {
    # campaign qutrit family ((n,3,3))_3, L=16 a2a MS, uniform protocol
    **{f"c_n{n}": dict(label=f"(({n},3,3))$_3$ L16 a2a", n=n, d=3,
                       ms=_a2a_ms(16, n, 3), seed0=2000 + 10 * n,
                       src=("npz", os.path.join(BPR, "campaign",
                                                f"d3_n{n}_dist3_16L.npz")))
      for n in (5, 6, 7, 8, 9)},
    "K2":    dict(label="((5,2,3))$_3$ L16 a2a", n=5, d=3, ms=320, seed0=2210,
                  src=("npz", os.path.join(OUTDIR, "K2_d3_n5_dist3_16L.npz"))),
    "dist2": dict(label="((5,3,2))$_3$ L16 a2a", n=5, d=3, ms=320, seed0=2220,
                  src=("npz", os.path.join(OUTDIR, "d3_n5_dist2_16L.npz"))),
    "d4K3":  dict(label="((5,3,3))$_4$ L16 a2a", n=5, d=4, ms=480, seed0=2230,
                  src=("npz", os.path.join(OUTDIR, "d4K3_n5_dist3_16L.npz"))),
    "d5K3":  dict(label="((5,3,3))$_5$ L16 a2a", n=5, d=5, ms=640, seed0=2240,
                  src=("npz", os.path.join(OUTDIR, "d5K3_n5_dist3_16L.npz"))),
    "s513":  dict(label="[[5,1,3]]$_{Z_3}$ (analytic)", n=5, d=3, ms=None,
                  seed0=2260, src=("catalog", "five_qudit_d3")),
    # the central code: the ((5,3,3))_3 reference on every panel
    "t1head": dict(label="((5,3,3))$_3$ central code L2 a2a", n=5, d=3,
                   ms=40, seed0=2270,
                   src=("npz", os.path.join(BPR, "r1_campaign", "t1_codes",
                                            "A_L2_s100.npz"))),
}

PANELS = {
    "n":    dict(title="vary $n$:  $((n,3,3))_3$,  K=3, q=3, dist=3 fixed",
                 codes=["c_n5", "c_n6", "c_n7", "c_n8", "c_n9", "t1head",
                        "s513"],
                 refs=["t1head", "s513"]),
    "K":    dict(title="vary $K$:  $((5,K,3))_3$,  n=5, q=3, dist=3 fixed"
                       "  (singleton bound: $K \\leq 3$)",
                 codes=["K2", "c_n5", "t1head", "s513"],
                 refs=["t1head", "s513"]),
    "q":    dict(title="vary $q$:  $((5,3,3))_q$,  n=5, K=3, dist=3 fixed",
                 codes=["c_n5", "d4K3", "d5K3", "t1head", "s513"],
                 refs=["t1head", "s513"]),
    "dist": dict(title="vary dist:  $((5,3,\\mathrm{dist}))_3$,  n=5, K=3, q=3 fixed",
                 codes=["dist2", "c_n5", "t1head", "s513"],
                 refs=["t1head", "s513"]),
}

# The retired ring code ("pub") no longer appears on any panel; its cached
# measurements stay in measurements.json for provenance only.
REF_STYLE = {"t1head": dict(fmt="*--", color="goldenrod", ms=9),
             "s513": dict(fmt="s--", color="black", ms=5)}


def wilson(k, n, z=1.96):
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), c + h)


def load_states(key):
    """Return code_states (K, d^n) for a registry key, or None if source missing."""
    kind, ref = CODES[key]["src"]
    if kind == "npz":
        if not os.path.exists(ref):
            return None
        n = CODES[key]["n"]
        conns = [[i, j] for i in range(n) for j in range(i + 1, n)]
        from src.decoders._common import encoder_forward
        cs, *_ = encoder_forward(ref, conns)
        return cs
    from src.catalog import load_code
    return np.asarray(load_code(ref)["code_states"])


def measure_code(key, cs):
    """LER curve over PS with shot escalation; returns list of point dicts."""
    from src.errors import qudit_hardware_error_basis, make_hardware_noise_fn
    from src.simulation import simulate_ler_with_correction_factored
    n, d, seed0 = CODES[key]["n"], CODES[key]["d"], CODES[key]["seed0"]
    single = [np.asarray(E, complex) for E in qudit_hardware_error_basis(d)]
    ladder = [s for s in LADDER if not (n >= 9 and s > 1_000_000)]
    pts = []
    for p in PS:
        k = shots = None
        for shots in ladder:
            r = simulate_ler_with_correction_factored(
                cs, make_hardware_noise_fn(d, n, p), single, n, d,
                n_shots=shots, seed=seed0 + int(p * 1000))
            k = int(round(float(r["logical_error_rate"]) * shots))
            if k >= MIN_ERRORS:
                break
        lo, hi = wilson(k, shots)
        pts.append(dict(p=p, shots=shots, k=k, ler=k / shots,
                        wilson_lo=lo, wilson_hi=hi, below_detection=(k == 0)))
        tag = "BELOW-DETECTION, Wilson UB" if k == 0 else f"k={k}"
        print(f"    p={p}: {k/shots:.3e} [{lo:.2e},{hi:.2e}] "
              f"@{shots//1000}k ({tag})", flush=True)
    return pts


def render_panel(name, meas, outdir=OUTDIR):
    spec = PANELS[name]
    have = [k for k in spec["codes"] if k in meas]
    if not have:
        return None
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    cmap = plt.get_cmap("viridis")
    main_keys = [k for k in have if k not in spec["refs"]]
    for i, key in enumerate(have):
        pts = meas[key]["points"]
        ms_b = meas[key]["ms_budget"]
        lab = meas[key]["label"] + (f"  [MS {ms_b}]" if ms_b else "  [no encoder]")
        res = [q for q in pts if not q["below_detection"]]
        bd = [q for q in pts if q["below_detection"]]
        if key in spec["refs"]:
            st = dict(REF_STYLE[key]); fmt = st.pop("fmt")
            kw = dict(lw=1.3, alpha=0.85, **st)
        else:
            fmt, kw = "o-", dict(color=cmap(main_keys.index(key) /
                                            max(1, len(main_keys) - 1)),
                                 lw=1.6, ms=5)
        if res:
            xs = [q["p"] for q in res]; ys = [q["ler"] for q in res]
            ax.errorbar(xs, ys,
                        yerr=[np.subtract(ys, [q["wilson_lo"] for q in res]).clip(0),
                              np.subtract([q["wilson_hi"] for q in res], ys).clip(0)],
                        fmt=fmt, capsize=2, label=lab, **kw)
        if bd:  # 0-error cells: Wilson UPPER BOUND, open down-triangle
            col = kw.get("color")
            ax.plot([q["p"] for q in bd], [q["wilson_hi"] for q in bd], "v",
                    color=col, mfc="white", ms=6,
                    label=None if res else lab + " (UB)")
    ax.set_yscale("log")
    ax.set_xlabel("physical error rate $p$ (i.i.d. per qudit, stated per point)")
    ax.set_ylabel("logical error rate (factored lookup decoder)")
    ax.set_title(f"Fig 6{name}: {spec['title']}\n"
                 "shots $\\geq$300k escalated to $\\geq$30 errors; "
                 "$\\triangledown$ = 0 errors (Wilson 95% upper bound); "
                 "[MS] = $L\\cdot|E|\\cdot(q-1)$", fontsize=9)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7.5)
    fig.tight_layout()
    out = os.path.join(outdir, f"fig6_panel_{name}.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    return out


def render_panel_q_infid(meas, outdir=OUTDIR):
    """The q panel with x = per-qudit average channel infidelity.

    The i.i.d. discrete hardware channel at injection probability p has
    per-qudit 1-F_avg = c_q * p exactly (linear), with
    c_q = (4q-4)/(q^2+q) = 2/3, 3/5, 8/15 at q = 3, 4, 5 (numerically
    verified in r1_campaign/t5a_remap.json). Rendering LER against c_q*p
    removes the "equal p is not physically identical across q" caveat at
    the channel-strength level; the raw-p panel is kept for the appendix.
    """
    C_Q = {3: 2.0 / 3.0, 4: 0.6, 5: 8.0 / 15.0}
    spec = PANELS["q"]
    have = [k for k in spec["codes"] if k in meas]
    if not have:
        return None
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    cmap = plt.get_cmap("viridis")
    main_keys = [k for k in have if k not in spec["refs"]]
    for key in have:
        q = CODES[key]["d"]
        cq = C_Q[q]
        pts = meas[key]["points"]
        ms_b = meas[key]["ms_budget"]
        lab = (meas[key]["label"] + (f"  [MS {ms_b}]" if ms_b else
                                     "  [no encoder]"))
        res = [x for x in pts if not x["below_detection"]]
        bd = [x for x in pts if x["below_detection"]]
        if key in spec["refs"]:
            st = dict(REF_STYLE[key]); fmt = st.pop("fmt")
            kw = dict(lw=1.3, alpha=0.85, **st)
        else:
            fmt, kw = "o-", dict(color=cmap(main_keys.index(key) /
                                            max(1, len(main_keys) - 1)),
                                 lw=1.6, ms=5)
        if res:
            xs = [cq * x["p"] for x in res]; ys = [x["ler"] for x in res]
            ax.errorbar(xs, ys,
                        yerr=[np.subtract(ys, [x["wilson_lo"] for x in res]).clip(0),
                              np.subtract([x["wilson_hi"] for x in res], ys).clip(0)],
                        fmt=fmt, capsize=2, label=lab, **kw)
        if bd:
            col = kw.get("color")
            ax.plot([cq * x["p"] for x in bd],
                    [x["wilson_hi"] for x in bd], "v",
                    color=col, mfc="white", ms=6,
                    label=None if res else lab + " (UB)")
    ax.set_yscale("log")
    ax.set_xlabel("per-qudit average channel infidelity  $1-F_\\mathrm{avg}"
                  " = c_q\\,p$   ($c_q = \\frac{4q-4}{q^2+q}$)")
    ax.set_ylabel("logical error rate (factored lookup decoder)")
    ax.set_title("Fig 6q (remapped): vary $q$:  $((5,3,3))_q$, n=5, K=3, "
                 "dist=3 fixed\nx-axis = per-qudit channel infidelity "
                 "(removes the cross-$q$ channel-strength caveat); "
                 "$\\triangledown$ = below detection (Wilson 95% UB)",
                 fontsize=9)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7.5)
    fig.tight_layout()
    out = os.path.join(outdir, "fig6_panel_q_infid.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    return out


def render_all_panels():
    """Render every panel PNG from committed measurements.json (no recompute)."""
    if not os.path.exists(MEAS):
        return []
    meas = json.load(open(MEAS))["codes"]
    outs = [p for name in PANELS if (p := render_panel(name, meas))]
    if (p := render_panel_q_infid(meas)):
        outs.append(p)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="render-only from JSON")
    ap.add_argument("--force", action="store_true", help="re-measure cached codes")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    if args.render:
        for p in render_all_panels():
            print(f"wrote {p}")
        return

    doc = json.load(open(MEAS)) if os.path.exists(MEAS) else {
        "p_grid": PS, "min_errors": MIN_ERRORS, "ladder": LADDER,
        "policy": "escalate shots until >=30 errors or ladder cap (n>=9 capped "
                  "at 1M); k=0 cells are below-detection Wilson-UB, never zero",
        "instrument": "batched factored lookup LER",
        "codes": {}}
    needed = sorted({k for s in PANELS.values() for k in s["codes"]})
    for key in needed:
        if key in doc["codes"] and not args.force:
            print(f"[cache] {key} already measured -- skip", flush=True)
            continue
        cs = load_states(key)
        if cs is None:
            print(f"[skip] {key}: source not on disk yet "
                  f"({CODES[key]['src'][1]})", flush=True)
            continue
        print(f"\n=== {key} ({CODES[key]['label']}) ===", flush=True)
        t = time.time()
        pts = measure_code(key, cs)
        doc["codes"][key] = dict(label=CODES[key]["label"],
                                 ms_budget=CODES[key]["ms"],
                                 n=CODES[key]["n"], d=CODES[key]["d"],
                                 seed0=CODES[key]["seed0"], points=pts,
                                 wall_s=round(time.time() - t, 1))
        json.dump(doc, open(MEAS, "w"), indent=2)
    for p in render_all_panels():
        print(f"wrote {p}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
