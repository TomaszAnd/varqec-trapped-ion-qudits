#!/usr/bin/env python3
"""Certified optimal recovery (App. F): exact optimal-recovery SDP for
channel fidelity at n=5.

For each code and strength, the optimal recovery R*: B(H_243) -> B(C^K)
maximizing the entanglement fidelity of R o N_dose is the solution of
the standard channel-fidelity SDP (Fletcher-Shor-Win 2007) over the
Choi matrix X of R:

    maximize   F_e = Tr[A X],  A = (1/K^2) sum_ij M_ij^T (x) |j><i|
    subject to X >= 0 (PSD, hermitian),  Tr_out X = I_243  (TP)

with M_ij = N(|psi_i><psi_j|) the exact engine dose blocks. The output
space is the K-dim logical space (any recovery into the codespace
scored via the codewords is subsumed). Variable: 729 x 729 hermitian.
Support: rank of N(P_C) is reported per cell (243 = full at n=5);
no truncation needed.

Certificates recorded per cell: F_e(SDP) >= F_e(Petz) and
F_e(SDP) >= F_e(projective) (both must hold), plus the Barnum-Knill
two-sided check 1 - F_e(Petz) <= 2 (1 - F_e(opt)).

Codes: central code, [[5,1,3]], deep_L28 (residual 1.95e-4), and
depth_L16_s102 (the best trained code by Petz at weak s, from the
residual-ladder crossover analysis); s in {0.25, 0.5}. Question: does
stabilizer-wins survive at optimal recovery? Writes t2_results.json
incrementally.
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
from src.decoders._common import encoder_forward
from src.decoders.exact_channel import build_dose_steps, evolve_code_blocks, \
    apply_dose
from src.decoders.projective_decode import (
    build_petz_exact_dose, build_projective_instrument,
    entanglement_fidelity_petz, entanglement_fidelity_projective,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "t2_results.json")
BR = os.path.join(REPO, "results", "campaigns")

SIGMA_P2, SIGMA_A2_CAL = 0.073, 0.047
D, N = 3, 5
DIM = D ** N
GP = [(i, j, 0, 1) for i in range(N) for j in range(i + 1, N)]
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]
S_GRID = (0.5, 0.25)

CODES = {
    "five_one_three_Z3": None,
    "headline_A_L2_s100": os.path.join(BR, "r1_campaign", "t1_codes",
                                       "A_L2_s100.npz"),
    "deep_L28": os.path.join(BR, "headline_p05_Lsweep", "n5_L28.npz"),
    "depth_L16_s102": os.path.join(BR, "r1_campaign", "t1_codes",
                                   "A_L16_s102.npz"),
}


def solve_cell(cs, steps):
    import cvxpy as cp
    K = cs.shape[0]
    blocks = evolve_code_blocks(cs, steps, D, N)
    # support rank of N(P_C)
    P_C = cs.T @ cs.conj()
    NPC = apply_dose(P_C, steps, D, N)
    evals = np.linalg.eigvalsh(NPC)
    rank = int((evals > 1e-12 * evals.max()).sum())

    A = np.zeros((DIM * K, DIM * K), dtype=complex)
    for i in range(K):
        for j in range(K):
            M = blocks[(i, j)] if i <= j else blocks[(j, i)].conj().T
            B = np.zeros((K, K))
            B[j, i] = 1.0
            A += np.kron(M.T, B)
    A /= K ** 2
    assert np.linalg.norm(A - A.conj().T) < 1e-12

    X = cp.Variable((DIM * K, DIM * K), hermitian=True)
    cons = [X >> 0,
            cp.partial_trace(X, [DIM, K], 1) == np.eye(DIM)]
    prob = cp.Problem(cp.Maximize(cp.real(cp.trace(A @ X))), cons)
    t0 = time.time()
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
        solver = "CLARABEL"
    except Exception as e:  # noqa: BLE001
        print(f"CLARABEL failed ({e}); falling back to SCS", flush=True)
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-8, max_iters=200000)
        solver = "SCS"
    F_e_opt = float(prob.value)
    return F_e_opt, rank, solver, prob.status, time.time() - t0, blocks


def main():
    res = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            res = json.load(f)
    res.setdefault("meta", {
        "d": D, "n": N, "s_grid": list(S_GRID),
        "convention": "eta(s)=e^{-s*0.073}, sigma_a2(s)=s*0.047",
        "sdp": "channel-fidelity SDP over Choi of R: B(H_243)->B(C^K); "
               "maximize Tr[A X], X PSD, Tr_out X = I",
        "metric": "rate = 1 - F_avg from the OPTIMAL F_e"})
    res.setdefault("cells", {})

    states = {}
    for name, path in CODES.items():
        states[name] = (five_qudit_code_states(D) if path is None
                        else encoder_forward(path, CONNS)[0])

    for s in S_GRID:
        eta, sa2 = float(np.exp(-s * SIGMA_P2)), s * SIGMA_A2_CAL
        steps = build_dose_steps(N, D, GP, eta, sigma_a2=sa2)
        for name, cs in states.items():
            key = f"{name}@v2_s{s:g}"
            if key in res["cells"]:
                continue
            K = cs.shape[0]
            F_e_opt, rank, solver, status, wall, blocks = \
                solve_cell(cs, steps)
            # reference instruments on the same blocks
            inst = build_projective_instrument(cs, D, N, eta, sigma_a2=sa2)
            proj = entanglement_fidelity_projective(inst, blocks)
            petz = build_petz_exact_dose(cs, steps, D, N)
            pz = entanglement_fidelity_petz(petz, cs, blocks)
            F_avg_opt = (K * F_e_opt + 1.0) / (K + 1.0)
            cell = {
                "code": name, "s": s, "eta": eta, "sigma_a2": sa2,
                "support_rank_NPC": rank, "solver": solver,
                "status": status, "wall_s": round(wall, 1),
                "F_e_opt": F_e_opt,
                "optimal_rate_1mFavg": 1.0 - F_avg_opt,
                "petz_rate": pz["rate_1mFavg"],
                "projective_rate": proj["rate_1mFavg"],
                "petz_F_e": pz["F_e"], "proj_F_e": proj["F_e"],
                "cert_opt_ge_petz": bool(F_e_opt >= pz["F_e"] - 1e-7),
                "cert_opt_ge_proj": bool(F_e_opt >= proj["F_e"] - 1e-7),
                "barnum_knill_ok": bool(
                    (1.0 - pz["F_e"]) <= 2.0 * (1.0 - F_e_opt) + 1e-7),
                "petz_over_opt_rate_ratio":
                    pz["rate_1mFavg"] / (1.0 - F_avg_opt)
                    if F_avg_opt < 1.0 else None,
            }
            res["cells"][key] = cell
            with open(OUT, "w") as f:
                json.dump(res, f, indent=2)
            print(f"{key}: opt_rate={cell['optimal_rate_1mFavg']:.6e} "
                  f"petz={cell['petz_rate']:.4e} "
                  f"proj={cell['projective_rate']:.4e} "
                  f"[{solver} {status} {cell['wall_s']}s rank={rank}]",
                  flush=True)

    # ordering summary per s
    summary = {}
    for s in S_GRID:
        row = {}
        ref = res["cells"][f"five_one_three_Z3@v2_s{s:g}"]
        for name in CODES:
            c = res["cells"].get(f"{name}@v2_s{s:g}")
            if c is None:
                continue
            row[name] = {
                "optimal_rate": c["optimal_rate_1mFavg"],
                "ratio_vs_513_optimal":
                    c["optimal_rate_1mFavg"] / ref["optimal_rate_1mFavg"],
            }
        summary[f"s{s:g}"] = row
    res["summary"] = summary
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(summary, indent=1))
    print("done")


if __name__ == "__main__":
    main()
