#!/usr/bin/env python3
"""Repeated-round memory experiment (Fig. 5).

One round = v2 dose at strength s followed by the projective-MAP
instrument applied as a CPTP channel: sector collapse + unitary
correction, summed exactly:

  Lambda(sigma) = sum_k C_k B_k (B_k^dag sigma B_k) B_k^dag C_k^dag
                  + (1-P) sigma (1-P),  P = Ball Ball^dag

computed densely as W blockdiag(Ball^dag sigma Ball) W^dag + rest with
W = [C_k B_k] precomputed once per (code, s) -- exact matrix arithmetic,
a handful of dim^2 x dim gemms per block per round.

Codes: n=7 trio (d2/d3/d4 selected) at s in {0.25, 0.5}; references
central code A_L2_s100 + [[5,1,3]]_Z3 at n=5, same s. One common
instrument construction for all codes (the w<=2-closure prior-ordered
projective MAP at the round's channel setting). R = 12 rounds or
F_e < 0.5, whichever first. Per-round F_e/F_avg/rate recorded; the
exponential-regime fit is done by the analysis step (t3_analyze.py).

Resumable at (code, s) granularity. Writes t3_results.json incrementally.
A one-off consistency check validates the dense channel application
against apply_instrument_dense at n=5.
"""
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

import numpy as np

import src.jax_backend  # noqa: F401  (x64 first)
from src.catalog import five_qudit_code_states
from src.decoders._common import apply_local_np, encoder_forward
from src.decoders.exact_channel import apply_dose, build_dose_steps
from src.decoders.projective_decode import (
    apply_instrument_dense, build_projective_instrument, favg_from_fe,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "t3_results.json")
BR = os.path.join(REPO, "results", "campaigns")

SIGMA_P2, SIGMA_A2_CAL = 0.073, 0.047
D = 3
S_VALUES = (0.25, 0.5)
R_MAX = 12
FE_FLOOR = 0.5

CODES = [
    # (name, n, path or None for [[5,1,3]])
    ("d2", 7, os.path.join(BR, "r3_campaign", "t2c_codes", "d2_L2_s100.npz")),
    ("d3", 7, os.path.join(BR, "r3_campaign", "t2c_codes", "d3_L2_s100.npz")),
    ("d4", 7, os.path.join(BR, "r3_campaign", "t2c_codes", "d4_L2_s102.npz")),
    ("headline", 5, os.path.join(BR, "r1_campaign", "t1_codes",
                                 "A_L2_s100.npz")),
    ("five_one_three_Z3", 5, None),
]


def instrument_channel_ops(inst):
    """(Ball, W, blockmask) for the dense CPTP instrument application."""
    Ball = inst['Ball']
    d, n = inst['d'], inst['n_qudit']
    R = Ball.shape[1]
    W = np.empty_like(Ball)
    mask = np.zeros((R, R))
    for s_ in inst['sectors']:
        sl = slice(s_['offset'], s_['offset'] + s_['dim'])
        for c in range(s_['dim']):
            W[:, s_['offset'] + c] = apply_local_np(
                np.ascontiguousarray(Ball[:, s_['offset'] + c]),
                s_['C_dag'], s_['wires'], s_['inv'], d, n)
        mask[sl, sl] = 1.0
    return Ball, W, mask


def apply_instrument_channel(sigma, Ball, W, mask):
    Y = Ball.conj().T @ sigma                     # (R, dim)
    G = Y @ Ball                                  # (R, R)
    out = W @ ((G * mask) @ W.conj().T)
    SB = sigma @ Ball
    out += sigma - Ball @ Y - SB @ Ball.conj().T + Ball @ (G @ Ball.conj().T)
    return out


def consistency_check():
    """Dense channel application == apply_instrument_dense (n=5, v2 s=0.5)."""
    s = 0.5
    eta, sa2 = float(np.exp(-s * SIGMA_P2)), s * SIGMA_A2_CAL
    cs = five_qudit_code_states(D)
    inst = build_projective_instrument(cs, D, 5, eta, sigma_a2=sa2)
    Ball, W, mask = instrument_channel_ops(inst)
    rng = np.random.default_rng(3)
    A = rng.standard_normal((243, 243)) + 1j * rng.standard_normal((243, 243))
    sigma = A @ A.conj().T
    sigma /= np.trace(sigma).real
    ref = apply_instrument_dense(inst, sigma)
    new = apply_instrument_channel(sigma, Ball, W, mask)
    dev = float(np.max(np.abs(ref - new)))
    print(f"consistency check (dense gemm vs apply_instrument_dense): "
          f"max dev {dev:.2e}", flush=True)
    assert dev < 1e-10, dev
    return dev


def run_unit(name, n, path, s, res):
    key = f"{name}@s{s:.2f}"
    if key in res["cells"]:
        return
    eta, sa2 = float(np.exp(-s * SIGMA_P2)), s * SIGMA_A2_CAL
    conns = [[i, j] for i in range(n) for j in range(i + 1, n)]
    gp = [(i, j, 0, 1) for i in range(n) for j in range(i + 1, n)]
    if path is None:
        cs = five_qudit_code_states(D)
    else:
        cs, d_, n_, K_ = encoder_forward(path, conns)
        assert (d_, n_) == (D, n)
    K = cs.shape[0]
    t0 = time.time()
    steps = build_dose_steps(n, D, gp, eta, sigma_a2=sa2)
    inst = build_projective_instrument(cs, D, n, eta, sigma_a2=sa2)
    Ball, W, mask = instrument_channel_ops(inst)
    t_setup = time.time() - t0
    print(f"{key}: setup {t_setup:.0f}s "
          f"({inst['report']['n_sectors']} sectors, "
          f"R={inst['report']['claimed_dim_R']}/{D**n})", flush=True)

    blocks = {}
    for i in range(K):
        for j in range(i, K):
            blocks[(i, j)] = np.outer(cs[i], cs[j].conj())

    rounds = []
    for r in range(1, R_MAX + 1):
        t0 = time.time()
        f = np.zeros((K, K), dtype=complex)
        for i in range(K):
            for j in range(i, K):
                sig = apply_dose(blocks[(i, j)], steps, D, n)
                sig = apply_instrument_channel(sig, Ball, W, mask)
                blocks[(i, j)] = sig
                f[i, j] = cs[i].conj() @ (sig @ cs[j])
                if j > i:
                    f[j, i] = np.conj(f[i, j])
        F_e = float(np.real(f.sum()) / K ** 2)
        F_avg = favg_from_fe(F_e, K)
        tr = float(np.real(np.trace(blocks[(0, 0)])))
        rounds.append({"round": r, "F_e": F_e, "F_avg": F_avg,
                       "rate_1mFavg": 1.0 - F_avg,
                       "trace_block00": tr,
                       "wall_s": round(time.time() - t0, 1)})
        print(f"  {key} round {r}: F_e={F_e:.6f} rate={1.0-F_avg:.4e} "
              f"({rounds[-1]['wall_s']}s)", flush=True)
        if F_e < FE_FLOOR:
            break
    res["cells"][key] = {
        "code": name, "n": n, "s": s, "eta": eta, "sigma_a2": sa2,
        "n_sectors": inst['report']['n_sectors'],
        "claimed_dim_R": inst['report']['claimed_dim_R'],
        "setup_wall_s": round(t_setup, 1),
        "rounds": rounds,
    }
    with open(OUT, "w") as fjs:
        json.dump(res, fjs, indent=2)


def main():
    res = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            res = json.load(f)
    res.setdefault("meta", {
        "d": D, "s_values": list(S_VALUES), "R_max": R_MAX,
        "fe_floor": FE_FLOOR,
        "round": "rho -> Lambda_projMAP(dose_v2(rho)), exact",
        "instrument": "common construction: w<=2-closure prior-ordered "
                      "projective MAP at the round's channel setting",
        "convention": "eta(s)=e^{-s*0.073}, sigma_a2(s)=s*0.047",
        "metric": "per-round F_e / F_avg / rate=1-F_avg (channel "
                  "infidelity, NOT a shot fraction)"})
    res.setdefault("cells", {})
    res["meta"]["consistency_dev"] = consistency_check()
    # cheap n=5 references first, then the n=7 trio
    for name, n, path in sorted(CODES, key=lambda c: c[1]):
        for s in S_VALUES:
            run_unit(name, n, path, s, res)
    print("done")


if __name__ == "__main__":
    main()
