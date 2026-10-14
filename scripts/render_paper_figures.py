#!/usr/bin/env python3
"""Render every data figure of the paper from the committed campaign records.

Usage (from the repository root):

    PYTHONPATH=. python3 scripts/render_paper_figures.py [figure ...]

Writes PNGs into results/figures/.  With no argument all ten figures are
rendered; otherwise only the named ones (keys of the FIGURES table below).
The per-figure functions live in scripts/paper_figures/ (one module per
revision pass of the manuscript; the module names carry the pass tag the
figure file names end with).  Each function echoes the caption facts it
asserts at render time.  Nothing here recomputes a logical error rate or a
channel infidelity: the data come from results/campaigns/ (see
docs/PROVENANCE.md for the record behind each figure).
"""
import importlib
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "paper_figures"))

# paper figure -> (output file, module, function)
FIGURES = {
    "fig3_n":        ("fig_n_w6.png",                "w6_render_figs", "fig_n"),
    "fig4_q":        ("fig_q_w5.png",                "w5_render_figs", "fig_q"),
    "fig5_dist":     ("fig_dist_w8.png",             "w8_render_figs", "fig_dist"),
    "fig6_depth":    ("fig_depth_w6.png",            "w6_render_figs", "fig_depth"),
    "fig7_prune":    ("fig_prune_w6.png",            "w6_render_figs", "fig_prune"),
    "fig8_gatecost": ("fig_gatecost_w7.png",         "w7_render_figs", "fig_gatecost_w7"),
    "fig9_gates":    ("fig_gates_w7.png",            "w7_render_figs", "fig_gates_w7"),
    "fig10_frontier": ("fig_sampler_frontier_w8.png", "w8_render_figs", "fig_frontier"),
    "fig11_traces":  ("fig_sampler_traces_w8.png",   "w8_render_figs", "fig_traces"),
    "fig12_pairs":   ("fig_headline_pairs_w8.png",   "w8_render_figs", "fig_headline_pairs"),
}


def main(argv):
    wanted = argv or list(FIGURES)
    out_dir = os.path.join(REPO, "results", "figures")
    os.makedirs(out_dir, exist_ok=True)
    status = {}
    for key in wanted:
        fname, modname, fn = FIGURES[key]
        mod = importlib.import_module(modname)
        print(f"== {key}: {modname}.{fn} -> results/figures/{fname}", flush=True)
        getattr(mod, fn)()
        status[key] = os.path.exists(os.path.join(out_dir, fname))
    print("rendered:", status)
    missing = [k for k, ok in status.items() if not ok]
    if missing:
        raise SystemExit(f"missing outputs: {missing}")


if __name__ == "__main__":
    main(sys.argv[1:])
