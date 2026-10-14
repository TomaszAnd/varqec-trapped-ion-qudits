#!/usr/bin/env python3
"""Residual ladder (EXPLORATORY, timeboxed): fine-tune the deepest existing
code one order lower in residual.

Warm-starts from headline_p05_Lsweep/n5_L28.npz (unweighted full-closure
L=28, full-basis residual 1.95e-4 — the deepest, lowest-residual trained
code in the paper) and continues FULL-BATCH unweighted training toward
target 1.95e-5 (one order lower). Then scores the fine-tuned code
exactly on channel v2 (projective + Petz, full s grid) and on the iid
benchmark at p=0.05.

Question: does "training deeper into an unweighted loss actively hurts
calibrated-channel recoverability" continue for one more order of
residual? Clearly labeled exploratory — not a paper protocol run.

Writes t3_ft_results.json incrementally; code to t3_ft_codes/.
"""
import json
import math
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

import src.jax_backend  # noqa: F401  (x64 first)
import numpy as np
import jax
import jax.numpy as jnp
import optax

from src.seed_race import build_race
from src.decoders._common import encoder_forward
from src.decoders.exact_channel import build_dose_steps, evolve_code_blocks
from src.decoders.projective_decode import (
    build_petz_exact_dose, build_projective_instrument,
    entanglement_fidelity_petz, entanglement_fidelity_projective,
)
from src.errors import qudit_hardware_error_basis, make_hardware_noise_fn
from src.simulation import simulate_ler_with_correction_factored

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "t3_ft_results.json")
CODE_DIR = os.path.join(HERE, "t3_ft_codes")
BR = os.path.join(REPO, "results", "campaigns")
SRC_NPZ = os.path.join(BR, "headline_p05_Lsweep", "n5_L28.npz")
FT_NPZ = os.path.join(CODE_DIR, "n5_L28_ft1e-5.npz")

D, N, L = 3, 5, 28
SIGMA_P2, SIGMA_A2_CAL = 0.073, 0.047
GP = [(i, j, 0, 1) for i in range(N) for j in range(i + 1, N)]
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]
S_GRID = (0.125, 0.25, 0.5, 1.0, 1.5)
TARGET = 1.95e-5
MAX_STEPS = 12000
LR = 0.005


def wilson(k, n, z=1.96):
    ph = k / n
    dn = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / dn
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / dn
    return (max(0.0, c - h), c + h)


def main():
    os.makedirs(CODE_DIR, exist_ok=True)
    res = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            res = json.load(f)
    res.setdefault("meta", {
        "label": "EXPLORATORY fine-tune, not a paper protocol run",
        "source_code": "headline_p05_Lsweep/n5_L28.npz (residual 1.95e-4)",
        "target": TARGET, "max_steps": MAX_STEPS, "lr": LR,
        "objective": "unweighted full-closure full batch (as trained)"})

    if "training" not in res:
        z = np.load(SRC_NPZ)
        theta = jnp.asarray(z["params"])
        val_grad_fn, full_loss_fn, *_rest = build_race(
            D, N, 3, L, noise="full")
        full_weights = _rest[3]
        l0 = float(full_loss_fn(theta))
        print(f"start full-basis loss {l0:.6e}", flush=True)
        opt = optax.adam(LR)
        state = opt.init(theta)

        @jax.jit
        def step(theta, state):
            (val, grad) = val_grad_fn(theta, full_weights)
            updates, state = opt.update(grad, state)
            return optax.apply_updates(theta, updates), state, val

        t0 = time.time()
        hist = []
        best = (l0, theta)
        stop_step = None
        for i in range(1, MAX_STEPS + 1):
            theta, state, val = step(theta, state)
            if i % 25 == 0:
                fl = float(full_loss_fn(theta))
                hist.append((i, fl))
                if fl < best[0]:
                    best = (fl, theta)
                if i % 500 == 0:
                    print(f"step {i}: full loss {fl:.6e} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                if fl <= TARGET:
                    stop_step = i
                    break
        final_loss, final_theta = best
        np.savez(FT_NPZ, params=np.asarray(final_theta),
                 d=D, n_qudit=N, K=3, n_layers=L, distance=3,
                 final_loss=final_loss, noise="full", entangler="ms",
                 connectivity="all-to-all",
                 connections=np.array(CONNS),
                 note="exploratory fine-tune from n5_L28.npz")
        res["training"] = {
            "start_loss": l0, "final_loss": final_loss,
            "target_reached": bool(final_loss <= TARGET),
            "stop_step": stop_step, "wall_s": round(time.time() - t0, 1),
            "history_every25": hist[-40:]}
        with open(OUT, "w") as f:
            json.dump(res, f, indent=2)
        print(f"fine-tune done: {l0:.3e} -> {final_loss:.3e}", flush=True)

    # ---- exact v2 scoring ----
    res.setdefault("cells", {})
    cs, d_, n_, K_ = encoder_forward(FT_NPZ, CONNS)
    assert (d_, n_) == (D, N)
    for s in S_GRID:
        key = f"ft_L28@v2_s{s:g}"
        if key in res["cells"]:
            continue
        eta, sa2 = float(np.exp(-s * SIGMA_P2)), s * SIGMA_A2_CAL
        steps = build_dose_steps(N, D, GP, eta, sigma_a2=sa2)
        inst = build_projective_instrument(cs, D, N, eta, sigma_a2=sa2)
        blocks = evolve_code_blocks(cs, steps, D, N)
        proj = entanglement_fidelity_projective(inst, blocks)
        petz = build_petz_exact_dose(cs, steps, D, N)
        pz = entanglement_fidelity_petz(petz, cs, blocks)
        res["cells"][key] = {"s": s,
                             "projective_rate": proj["rate_1mFavg"],
                             "petz_rate": pz["rate_1mFavg"]}
        with open(OUT, "w") as f:
            json.dump(res, f, indent=2)
        print(f"{key}: proj={proj['rate_1mFavg']:.4e} "
              f"petz={pz['rate_1mFavg']:.4e}", flush=True)

    # ---- iid LER at p=0.05 ----
    if "iid_p005" not in res:
        singles = [np.asarray(E, complex)
                   for E in qudit_hardware_error_basis(D)]
        shots = 300000
        r = simulate_ler_with_correction_factored(
            np.asarray(cs), make_hardware_noise_fn(D, N, 0.05), singles,
            N, D, n_shots=shots, seed=1234)
        ler = float(r["logical_error_rate"])
        k = int(round(ler * shots))
        res["iid_p005"] = {"shots": shots, "k": k, "ler": ler,
                           "wilson95": wilson(k, shots)}
        with open(OUT, "w") as f:
            json.dump(res, f, indent=2)
        print("iid:", res["iid_p005"], flush=True)
    print("done")


if __name__ == "__main__":
    main()
