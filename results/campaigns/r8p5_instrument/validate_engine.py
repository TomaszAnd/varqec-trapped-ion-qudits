#!/usr/bin/env python3
"""Exact-engine validation: full n=5 dose trajectory convergence (both
unravelings) against the exact density-matrix engine + engine timing at
n=5 and n=7.

Writes engine_validation.json next to this script.
"""
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

import numpy as np

import src.jax_backend  # noqa: F401  (x64 first)
from src.decoders.exact_channel import apply_dose, build_dose_steps
from src.simulation import make_channel_v2_noise_fn, make_meth_mc_noise_fn

SIGMA_A2 = 0.047
ETA = float(np.exp(-0.073))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "engine_validation.json")


def a2a(n):
    return [(i, j, 0, 1) for i in range(n) for j in range(i + 1, n)]


def rand_state(dim, seed):
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return psi / np.linalg.norm(psi)


def main():
    res = {"eta": ETA, "sigma_a2": SIGMA_A2}
    d = 3

    # ── timing: one full-dose application per n ──
    for n in (5, 7):
        dim = d ** n
        steps = build_dose_steps(n, d, a2a(n), ETA, sigma_a2=SIGMA_A2)
        psi = rand_state(dim, seed=n)
        rho0 = np.outer(psi, psi.conj())
        t0 = time.time()
        rho = apply_dose(rho0, steps, d, n)
        wall_fwd = time.time() - t0
        t0 = time.time()
        _ = apply_dose(rho, steps, d, n, adjoint=True)
        wall_adj = time.time() - t0
        tr = float(abs(np.trace(rho) - 1.0))
        herm = float(np.max(np.abs(rho - rho.conj().T)))
        min_eig = float(np.linalg.eigvalsh(rho).min()) if n <= 5 else None
        res[f"n{n}_timing"] = {
            "n_gate_steps": len(steps), "wall_s_forward": round(wall_fwd, 3),
            "wall_s_adjoint": round(wall_adj, 3), "trace_dev": tr,
            "herm_dev": herm, "min_eig": min_eig,
            "rho_bytes_mb": round(rho.nbytes / 2 ** 20, 1)}
        print(f"n={n}: dose {len(steps)} steps, fwd {wall_fwd:.3f}s, "
              f"adj {wall_adj:.3f}s, trace dev {tr:.2e}", flush=True)

    # ── n=5 full-dose trajectory convergence, both unravelings ──
    n = 5
    dim = d ** n
    gp = a2a(n)
    psi = rand_state(dim, seed=42)
    steps_kr = build_dose_steps(n, d, gp, ETA, sigma_a2=SIGMA_A2,
                                mode="kraus", n_max_dephasing=5,
                                n_max_jitter=10)
    steps_ex = build_dose_steps(n, d, gp, ETA, sigma_a2=SIGMA_A2)
    rho_kr = apply_dose(np.outer(psi, psi.conj()), steps_kr, d, n)
    rho_ex = apply_dose(np.outer(psi, psi.conj()), steps_ex, d, n)
    res["n5_kraus_vs_exact_maxdev"] = float(np.max(np.abs(rho_kr - rho_ex)))

    N_TRAJ = 4000
    for tag, maker, ref in (("modeB_kraus_sampler",
                             make_channel_v2_noise_fn, rho_kr),
                            ("modeA_trajectory_mc",
                             make_meth_mc_noise_fn, rho_ex)):
        kw = {"n_max": 5} if maker is make_channel_v2_noise_fn else {}
        nf = maker(n, d, gp, ETA, sigma_a2=SIGMA_A2, **kw)
        rng = np.random.default_rng(777)
        acc = np.zeros((dim, dim), dtype=complex)
        t0 = time.time()
        for _ in range(N_TRAJ):
            v = nf(psi, rng)
            acc += np.outer(v, v.conj())
        acc /= N_TRAJ
        wall = time.time() - t0
        dev = float(np.max(np.abs(acc - ref)))
        res[tag] = {"n_traj": N_TRAJ, "max_dev_vs_engine": dev,
                    "stat_scale": float(1.0 / np.sqrt(N_TRAJ)),
                    "wall_s": round(wall, 1),
                    "ms_per_traj": round(1e3 * wall / N_TRAJ, 3)}
        print(f"{tag}: dev {dev:.2e} vs 1/sqrt(N) {1/np.sqrt(N_TRAJ):.2e} "
              f"({wall:.1f}s)", flush=True)

    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
