#!/usr/bin/env python3
"""Optional robustness row for the repeated-rounds experiment: [[5,1,3]]_Z3
under its NATURAL stabilizer-syndrome instrument, one strength (s = 0.5),
same round iteration -- bounds the common-instrument penalty.

Instrument: the same build_syndrome_sectors machinery, but the correction
set is the weight-<=2 WEYL (generalized-Pauli) set X^a Z^b -- the errors
the stabilizer syndrome measurement natively distinguishes; each weight-1
Weyl maps the code space to a definite syndrome space, so the sectors ARE
the stabilizer syndrome spaces with max-prior coset-representative
corrections. Priors are an analytic dephasing-dominance tier ordering
(I >> Z-type w1 >> mixed w1 >> Z(x)Z w2 >> mixed w2 >> XX-type w2);
only the ORDER enters the construction. Single-round rates under this
instrument are also recorded (vs the common instrument's t2b/t1 values).

Writes t3_stab_results.json.
"""
import itertools
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

import numpy as np

import src.jax_backend  # noqa: F401  (x64 first)
from src.catalog import five_qudit_code_states
from src.decoders.exact_channel import (apply_dose, build_dose_steps,
                                        evolve_code_blocks)
from src.decoders.projective_decode import (build_syndrome_sectors,
                                            entanglement_fidelity_projective,
                                            favg_from_fe)
from t3_rounds import apply_instrument_channel, instrument_channel_ops

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "t3_stab_results.json")

SIGMA_P2, SIGMA_A2_CAL = 0.073, 0.047
D, N = 3, 5
S = 0.5
GP = [(i, j, 0, 1) for i in range(N) for j in range(i + 1, N)]
R_MAX = 12
FE_FLOOR = 0.5


def weyl_ops(d):
    """Non-identity single-qudit Weyl ops X^a Z^b with a type tag."""
    w = np.exp(2j * np.pi / d)
    X = np.zeros((d, d), dtype=complex)
    for j in range(d):
        X[(j + 1) % d, j] = 1.0
    Z = np.diag(w ** np.arange(d))
    ops = {}
    for a in range(d):
        for b in range(d):
            if a == 0 and b == 0:
                continue
            typ = "Z" if a == 0 else ("X" if b == 0 else "M")
            ops[(a, b)] = (np.linalg.matrix_power(X, a)
                           @ np.linalg.matrix_power(Z, b), typ)
    return ops


def inverse_perm_for(wires, n):
    new_axes = list(wires) + [ax for ax in range(n) if ax not in wires]
    return tuple(new_axes.index(k) for k in range(n))


def weyl_correction_set(d, n):
    """(wires, C_dag, inv, prior) for the weight-<=2 Weyl set with the
    dephasing-dominance tier prior (order-only)."""
    ops = weyl_ops(d)
    tier_w1 = {"Z": 1e-1, "M": 1e-2, "X": 1e-2}
    tier_w2 = {("Z", "Z"): 1e-3, ("Z", "M"): 1e-4, ("M", "Z"): 1e-4,
               ("Z", "X"): 1e-4, ("X", "Z"): 1e-4, ("M", "M"): 1e-5,
               ("M", "X"): 1e-5, ("X", "M"): 1e-5, ("X", "X"): 1e-5}
    corr = [((), np.array([[1.0 + 0j]]), tuple(range(n)), 1.0)]
    for q in range(n):
        inv = inverse_perm_for((q,), n)
        for (a, b), (M, typ) in ops.items():
            corr.append(((q,), M.conj().T, inv, tier_w1[typ]))
    for q1, q2 in itertools.combinations(range(n), 2):
        inv = inverse_perm_for((q1, q2), n)
        for (a1, b1), (M1, t1) in ops.items():
            for (a2, b2), (M2, t2) in ops.items():
                M = np.kron(M1, M2)
                corr.append(((q1, q2), M.conj().T, inv,
                             tier_w2[(t1, t2)]))
    return corr


def main():
    eta, sa2 = float(np.exp(-S * SIGMA_P2)), S * SIGMA_A2_CAL
    cs = five_qudit_code_states(D)
    corr = weyl_correction_set(D, N)
    print(f"{len(corr)} Weyl corrections (incl. identity)", flush=True)
    t0 = time.time()
    inst = build_syndrome_sectors(cs, corr, D, N)
    print(f"sectors: {inst['report']['n_sectors']}, "
          f"R={inst['report']['claimed_dim_R']}/{D**N}, "
          f"dims={sorted(set(inst['report']['sector_dims']))} "
          f"({time.time()-t0:.1f}s)", flush=True)
    steps = build_dose_steps(N, D, GP, eta, sigma_a2=sa2)

    # single-round rates under the native instrument
    blocks0 = evolve_code_blocks(cs, steps, D, N)
    proj = entanglement_fidelity_projective(inst, blocks0)

    # round iteration
    Ball, W, mask = instrument_channel_ops(inst)
    K = cs.shape[0]
    blocks = {(i, j): np.outer(cs[i], cs[j].conj())
              for i in range(K) for j in range(i, K)}
    rounds = []
    for r in range(1, R_MAX + 1):
        f = np.zeros((K, K), dtype=complex)
        for i in range(K):
            for j in range(i, K):
                sig = apply_dose(blocks[(i, j)], steps, D, N)
                sig = apply_instrument_channel(sig, Ball, W, mask)
                blocks[(i, j)] = sig
                f[i, j] = cs[i].conj() @ (sig @ cs[j])
                if j > i:
                    f[j, i] = np.conj(f[i, j])
        F_e = float(np.real(f.sum()) / K ** 2)
        rounds.append({"round": r, "F_e": F_e,
                       "rate_1mFavg": 1.0 - favg_from_fe(F_e, K)})
        print(f"round {r}: F_e={F_e:.6f}", flush=True)
        if F_e < FE_FLOOR:
            break

    out = {"meta": {"code": "five_one_three_Z3", "d": D, "n": N, "s": S,
                    "eta": eta, "sigma_a2": sa2,
                    "instrument": "native Weyl-syndrome projective MAP "
                                  "(w<=2 Weyl set, dephasing-dominance "
                                  "tier prior, order-only)",
                    "sector_report": inst["report"]},
           "single_round": {"projective_rate": proj["rate_1mFavg"],
                            "F_e": proj["F_e"]},
           "rounds": rounds}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print("done")


if __name__ == "__main__":
    main()
