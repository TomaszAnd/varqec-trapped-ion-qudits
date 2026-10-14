#!/usr/bin/env python3
"""Certified optimal recovery (App. F): exact optimal-recovery SDP at n=5 —
fixed-point iteration with CERTIFIED two-sided bounds.

Primal (channel-fidelity SDP over the Choi X of R: B(H_243)->B(C^K)):
    maximize  F_e = Tr[A X],  A = (1/K^2) sum_ij M_ij^T (x) |j><i|
    s.t.      Tr_out X = I_243,  X >= 0.
Dual:  min Tr[Y]  s.t.  Y (x) I_K >= A.

Generic solvers fail at this size (CLARABEL native crash; SCS setup
> 10 min; a plain ADMM converges too slowly), so we run the standard
fixed-point iteration for linear objectives over CPTP Choi matrices
(Reimpell-Werner / Jezek-Rehacek-Fiurasek family):

    W = A' X A',   Lam = Tr_out W,   X <- (Lam^{-1/2} (x) I) W
    (Lam^{-1/2} (x) I),

with A' = A + c I >= 0 (c = max(0, -lambda_min(A)); Tr X = 243 is
fixed by TP, so the shift changes the objective by the constant 243 c
and not the argmax). Every iterate is EXACTLY TP and PSD, so
lb = Tr[A X] is a rigorous lower bound on F_e_opt at every step; the
KKT candidate Y = Lam^{1/2} - c I, lambda-shifted to dual feasibility
(Y' = Y + |lambda_min(Y (x) I - A)| I), gives a rigorous upper bound
Tr[Y']. The certified gap is reported per cell; no convergence claim
is load-bearing.

Codes: [[5,1,3]], central code, deep_L28, depth_L16_s102;
s in {0.5, 0.25}. Writes t2_results.json incrementally.
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
K = 3
NBIG = DIM * K

CODES = {
    "five_one_three_Z3": None,
    "headline_A_L2_s100": os.path.join(BR, "r1_campaign", "t1_codes",
                                       "A_L2_s100.npz"),
    "deep_L28": os.path.join(BR, "headline_p05_Lsweep", "n5_L28.npz"),
    "depth_L16_s102": os.path.join(BR, "r1_campaign", "t1_codes",
                                   "A_L16_s102.npz"),
}

MAX_IT = 3000
GAP_TARGET = 2e-6          # absolute certified F_e gap
CHECK_EVERY = 10


def tr_out(X):
    T = X.reshape(DIM, K, DIM, K)
    return np.einsum("akbk->ab", T)


def kron_IK_mul(Y, X):
    """(Y (x) I_K) @ X without forming the kron."""
    T = X.reshape(DIM, K * NBIG)
    return (Y @ T).reshape(NBIG, NBIG)


def solve_jrf(A):
    lamA = float(np.linalg.eigvalsh(A)[0])
    c = max(0.0, -lamA) + 1e-12
    Ash = A + c * np.eye(NBIG)
    X = np.eye(NBIG, dtype=complex) / K
    best_lb, best_ub = -np.inf, np.inf
    t0 = time.time()
    it_done = 0
    for it in range(1, MAX_IT + 1):
        W = Ash @ X @ Ash
        Lam = tr_out(W)
        w, V = np.linalg.eigh(Lam)
        w = np.maximum(w, 1e-300)
        Lm = (V * (1.0 / np.sqrt(w))[None, :]) @ V.conj().T
        X = kron_IK_mul(Lm, kron_IK_mul(Lm, W.conj().T).conj().T)
        X = 0.5 * (X + X.conj().T)
        it_done = it
        if it % CHECK_EVERY == 0 or it == MAX_IT:
            lb = float(np.real(np.trace(A @ X)))
            # dual candidate Y = Lam^{1/2} - c I, lambda-shift to feasible
            Yc = (V * np.sqrt(w)[None, :]) @ V.conj().T - c * np.eye(DIM)
            Smat = kron_IK_mul(Yc, np.eye(NBIG, dtype=complex)) - A
            lam = float(np.linalg.eigvalsh(Smat)[0])
            ub = float(np.real(np.trace(Yc))) + DIM * max(0.0, -lam)
            best_lb = max(best_lb, lb)
            best_ub = min(best_ub, ub)
            gap = best_ub - best_lb
            if it % 100 == 0:
                print(f"  it {it}: lb {best_lb:.9f} ub {best_ub:.9f} "
                      f"gap {gap:.2e} ({time.time()-t0:.0f}s)", flush=True)
            if gap < GAP_TARGET:
                break
    return best_lb, best_ub, it_done, time.time() - t0


def main():
    res = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            res = json.load(f)
    res.setdefault("meta", {
        "d": D, "n": N, "s_grid": list(S_GRID),
        "convention": "eta(s)=e^{-s*0.073}, sigma_a2(s)=s*0.047",
        "solver": "Reimpell-Werner/JRF fixed point; certified bounds "
                  "(every iterate exactly TP+PSD -> lb; lambda-shifted "
                  "dual Y = Lam^{1/2} - cI -> ub)",
        "gap_target_F_e": GAP_TARGET,
        "metric": "rate = 1 - F_avg from OPTIMAL F_e (bounds propagated)"})
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
            blocks = evolve_code_blocks(cs, steps, D, N)
            P_C = cs.T @ cs.conj()
            NPC = apply_dose(P_C, steps, D, N)
            evals = np.linalg.eigvalsh(NPC)
            rank = int((evals > 1e-12 * evals.max()).sum())
            A = np.zeros((NBIG, NBIG), dtype=complex)
            for i in range(K):
                for j in range(K):
                    M = blocks[(i, j)] if i <= j else \
                        blocks[(j, i)].conj().T
                    B = np.zeros((K, K))
                    B[j, i] = 1.0
                    A += np.kron(M.T, B)
            A /= K ** 2
            A = 0.5 * (A + A.conj().T)
            print(f"{key}: solving (support rank {rank})", flush=True)
            lb, ub, iters, wall = solve_jrf(A)

            inst = build_projective_instrument(cs, D, N, eta, sigma_a2=sa2)
            proj = entanglement_fidelity_projective(inst, blocks)
            petz = build_petz_exact_dose(cs, steps, D, N)
            pz = entanglement_fidelity_petz(petz, cs, blocks)

            def rate(F_e):
                return 1.0 - (K * F_e + 1.0) / (K + 1.0)

            cell = {
                "code": name, "s": s, "eta": eta, "sigma_a2": sa2,
                "support_rank_NPC": rank,
                "F_e_opt_lb": lb, "F_e_opt_ub": ub,
                "F_e_gap_certified": ub - lb,
                "optimal_rate_lb": rate(ub),
                "optimal_rate_ub": rate(lb),
                "petz_rate": pz["rate_1mFavg"],
                "projective_rate": proj["rate_1mFavg"],
                "petz_F_e": pz["F_e"], "proj_F_e": proj["F_e"],
                "cert_opt_ge_petz": bool(ub >= pz["F_e"] - 1e-9),
                "petz_over_opt_rate_ratio_range": [
                    pz["rate_1mFavg"] / rate(lb) if rate(lb) > 0 else None,
                    pz["rate_1mFavg"] / rate(ub) if rate(ub) > 0 else None],
                "iters": iters, "wall_s": round(wall, 1),
            }
            res["cells"][key] = cell
            with open(OUT, "w") as f:
                json.dump(res, f, indent=2)
            print(f"{key}: opt rate in [{rate(ub):.6e}, {rate(lb):.6e}] "
                  f"petz={pz['rate_1mFavg']:.4e} proj="
                  f"{proj['rate_1mFavg']:.4e} ({iters} it, {wall:.0f}s)",
                  flush=True)

    # certified ordering summary
    summary = {}
    for s in S_GRID:
        ref = res["cells"].get(f"five_one_three_Z3@v2_s{s:g}")
        if ref is None:
            continue
        row = {}
        for name in CODES:
            c = res["cells"].get(f"{name}@v2_s{s:g}")
            if c is None or name == "five_one_three_Z3":
                continue
            row[name] = {
                "rate_interval": [c["optimal_rate_lb"],
                                  c["optimal_rate_ub"]],
                "ratio_vs_513_interval": [
                    c["optimal_rate_lb"] / ref["optimal_rate_ub"],
                    c["optimal_rate_ub"] / ref["optimal_rate_lb"]],
                "certified_513_wins": bool(
                    c["optimal_rate_lb"] > ref["optimal_rate_ub"]),
            }
        summary[f"s{s:g}"] = {
            "ref_513_rate_interval": [ref["optimal_rate_lb"],
                                      ref["optimal_rate_ub"]],
            "trained": row}
    res["summary"] = summary
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(summary, indent=1))
    print("done")


if __name__ == "__main__":
    main()
