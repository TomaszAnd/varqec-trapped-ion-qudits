#!/usr/bin/env python3
"""i.i.d. depth/pruning sweeps (Figs. 6-7; instrument untouched: iid draw +
lookup decoder, shot-fraction LER, Table-I p grid).

(a) Pruning sweep: central-code encoder at 40/16/12/8 surviving entangling
    gates (40 = A_L2_s100; 16/12/8 = the magnitude-pruning codes
    abl_prune60/70/80 of r2_campaign/t3_ablation.py), iid LER vs p over
    the Table-I grid, >=30 errors per cell or the shot cap stated
    (Wilson 95% recorded either way).
(b) Depth-study data: channel-weighted ((5,3,3))_3 at
    L in {2,4,8,16} x seeds {100..103} (the r1_campaign/t1 A_L sweep --
    reused, not retrained), same grid; plus per-code training-loss traces
    exported from the checkpoints' 'losses' arrays. GAP (stated): the
    A_L npz predate the full-batch monitor machinery and carry only the
    sampled per-step 'losses' history -- no de-noised monitor arrays.

Writes t5_results.json + t5_loss_traces.npz incrementally.
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
from src.errors import make_hardware_noise_fn, qudit_hardware_error_basis
from src.simulation import simulate_ler_with_correction_factored

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "t5_results.json")
TRACES = os.path.join(HERE, "t5_loss_traces.npz")
BR = os.path.join(REPO, "results", "campaigns")

D, N = 3, 5
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]
P_GRID = (0.02, 0.03, 0.05, 0.08, 0.12, 0.15)
MIN_ERRORS = 30
SHOTS_START = 300_000
SHOTS_CAP = 4_800_000

PRUNE = [
    ("prune_gates40", os.path.join(BR, "r1_campaign", "t1_codes",
                                   "A_L2_s100.npz")),
    ("prune_gates16", os.path.join(BR, "r2_campaign", "t3_codes",
                                   "abl_prune60.npz")),
    ("prune_gates12", os.path.join(BR, "r2_campaign", "t3_codes",
                                   "abl_prune70.npz")),
    ("prune_gates8", os.path.join(BR, "r2_campaign", "t3_codes",
                                  "abl_prune80.npz")),
]
DEPTH = [(f"depth_L{L}_s{seed}",
          os.path.join(BR, "r1_campaign", "t1_codes", f"A_L{L}_s{seed}.npz"),
          L, seed)
         for L in (2, 4, 8, 16) for seed in (100, 101, 102, 103)]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def iid_cell(cs, S, p, seed):
    """Escalate shots (independent reruns, doubling) until >= MIN_ERRORS
    or SHOTS_CAP."""
    n_shots = SHOTS_START
    while True:
        r = simulate_ler_with_correction_factored(
            cs, make_hardware_noise_fn(D, N, p), S, N, D,
            n_shots=n_shots, seed=seed)
        ler = float(r["logical_error_rate"])
        k = int(round(ler * n_shots))
        if k >= MIN_ERRORS or n_shots >= SHOTS_CAP:
            lo, hi = wilson(k, n_shots)
            return {"p": p, "ler": ler, "k": k, "n_shots": n_shots,
                    "wilson95": [float(lo), float(hi)],
                    "resolved": bool(k >= MIN_ERRORS)}
        n_shots = min(2 * n_shots, SHOTS_CAP)


def main():
    res = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            res = json.load(f)
    res.setdefault("meta", {
        "d": D, "n": N, "p_grid": list(P_GRID),
        "instrument": "iid draw + lookup decoder (untouched; "
                      "shot-fraction LER)",
        "min_errors": MIN_ERRORS, "shots_cap": SHOTS_CAP,
        "loss_trace_gap": "A_L depth-sweep npz carry only the sampled "
                          "per-step 'losses' history; no de-noised "
                          "monitor arrays (predate the monitor machinery) "
                          "-- gap stated, no retraining"})
    res.setdefault("cells", {})
    S = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]

    traces = {}
    if os.path.exists(TRACES):
        traces = dict(np.load(TRACES))

    jobs = [(name, path, None, None) for name, path in PRUNE] + \
           [(name, path, L, seed) for name, path, L, seed in DEPTH]
    for name, path, L, seed in jobs:
        cs, d_, n_, K_ = encoder_forward(path, CONNS)
        assert (d_, n_) == (D, N)
        # export loss trace once per code
        if name not in traces:
            data = np.load(path, allow_pickle=True)
            if "losses" in data.files:
                traces[name] = np.asarray(data["losses"], dtype=float)
            np.savez_compressed(TRACES, **traces)
        for p in P_GRID:
            key = f"{name}@p{p:.2f}"
            if key in res["cells"]:
                continue
            t0 = time.time()
            cell = iid_cell(cs, S, p, seed=hash((name, p)) % (2 ** 31))
            cell.update({"code": name, "path": os.path.relpath(path, REPO),
                         "L": L, "seed": seed,
                         "wall_s": round(time.time() - t0, 1)})
            res["cells"][key] = cell
            with open(OUT, "w") as f:
                json.dump(res, f, indent=2)
            print(f"{key}: ler={cell['ler']:.3e} k={cell['k']} "
                  f"n={cell['n_shots']} ({cell['wall_s']}s)", flush=True)
    print("done")


if __name__ == "__main__":
    main()
