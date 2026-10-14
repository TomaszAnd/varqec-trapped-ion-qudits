#!/usr/bin/env python3
"""Certified optimal recovery (App. F): exact optimal-recovery SDP at n=5,
specialized ADMM solver with CERTIFIED two-sided bounds.

Problem (channel-fidelity SDP over the Choi X of R: B(H_243)->B(C^K)):

    maximize  F_e = Tr[A X],   A = (1/K^2) sum_ij M_ij^T (x) |j><i|
    s.t.      Tr_out X = I_243,   X >= 0        (X hermitian 729x729)

Generic solvers fail at this size on this machine (CLARABEL: native
crash; SCS: setup exceeds 10 min), so this script runs a two-block ADMM
(affine projection in closed form; PSD projection via eigh) and
certifies the answer independently of ADMM convergence claims:

  LOWER bound: X_feas = (W^{-1/2} (x) I_K) Z (W^{-1/2} (x) I_K) with
    Z the PSD iterate, W = Tr_out Z  -> exactly TP and PSD, so
    Tr[A X_feas] <= F_e_opt.
  UPPER bound: dual is  min Tr[Y] s.t. Y (x) I_K - A >= 0. Candidate
    Y from the iterates, made feasible by shifting with
    |lambda_min(Y (x) I_K - A)| I  -> Tr[Y'] >= F_e_opt.

Both bounds are exact linear algebra on the final iterates; the gap is
reported per cell. Codes: [[5,1,3]], central code, deep_L28,
depth_L16_s102; s in {0.5, 0.25}. Writes t2_results.json incrementally.
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

MAX_IT = 4000
RHO = 1.0
GAP_TARGET = 5e-5          # absolute F_e gap to certify
CHECK_EVERY = 25


def tr_out(X):
    """Partial trace over the K-dim output factor: (NBIG,NBIG)->(DIM,DIM)."""
    T = X.reshape(DIM, K, DIM, K)
    return np.einsum("akbk->ab", T)


def kron_IK(Y):
    """Y (x) I_K without forming kron explicitly."""
    Z = np.zeros((DIM, K, DIM, K), dtype=complex)
    for k in range(K):
        Z[:, k, :, k] = Y
    return Z.reshape(NBIG, NBIG)


def proj_affine(X):
    """Project onto {Tr_out X = I_DIM} (hermitian-preserving)."""
    T = tr_out(X) - np.eye(DIM)
    return X - kron_IK(T) / K


def proj_psd(X):
    w, V = np.linalg.eigh(X)
    pos = w > 0
    return (V[:, pos] * w[pos][None, :]) @ V[:, pos].conj().T


def lower_bound(Z, A):
    """Rescale the PSD iterate into an exactly TP Choi; Tr[A X_feas]."""
    W = tr_out(Z)
    w, V = np.linalg.eigh(W)
    w = np.maximum(w, 1e-14)
    Wm = (V * (1.0 / np.sqrt(w))[None, :]) @ V.conj().T
    F = kron_IK(Wm)
    Xf = F @ Z @ F.conj().T
    return float(np.real(np.trace(A @ Xf)))


def upper_bound(Y, A):
    """Shift Y to dual feasibility; return certified Tr[Y']."""
    Yh = 0.5 * (Y + Y.conj().T)
    Smat = kron_IK(Yh) - A
    lam = float(np.linalg.eigvalsh(Smat)[0])
    shift = max(0.0, -lam)
    return float(np.real(np.trace(Yh))) + DIM * shift, lam


def solve_admm(A):
    X = np.eye(NBIG, dtype=complex) / K
    Z = X.copy()
    U = np.zeros_like(X)
    best = {"lb": -np.inf, "ub": np.inf}
    t0 = time.time()
    for it in range(1, MAX_IT + 1):
        X = proj_affine(Z - U + A / RHO)
        Z = proj_psd(X + U)
        U += X - Z
        if it % CHECK_EVERY == 0:
            lb = lower_bound(Z, A)
            # dual candidate from the scaled multiplier S = rho*U:
            # S + A ~ Y (x) I_K  ->  Y = (1/K) Tr_out(S + A)
            Y = tr_out(RHO * U + A) / K
            ub, lam = upper_bound(Y, A)
            best["lb"] = max(best["lb"], lb)
            best["ub"] = min(best["ub"], ub)
            gap = best["ub"] - best["lb"]
            if it % 200 == 0:
                print(f"  it {it}: lb {best['lb']:.8f} ub {best['ub']:.8f} "
                      f"gap {gap:.2e} lam_min {lam:.1e} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if gap < GAP_TARGET:
                break
    return best["lb"], best["ub"], it, time.time() - t0


def main():
    res = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            res = json.load(f)
    res.setdefault("meta", {
        "d": D, "n": N, "s_grid": list(S_GRID),
        "convention": "eta(s)=e^{-s*0.073}, sigma_a2(s)=s*0.047",
        "solver": "specialized ADMM, certified primal/dual bounds "
                  "(TP-rescaled feasible Choi; lambda-shifted dual Y)",
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
            print(f"{key}: solving (support rank {rank})", flush=True)
            lb, ub, iters, wall = solve_admm(A)

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
                "optimal_rate_from_ub_Fe": rate(ub),   # lower bd on rate
                "optimal_rate_from_lb_Fe": rate(lb),   # upper bd on rate
                "petz_rate": pz["rate_1mFavg"],
                "projective_rate": proj["rate_1mFavg"],
                "petz_F_e": pz["F_e"], "proj_F_e": proj["F_e"],
                "cert_opt_ge_petz": bool(ub >= pz["F_e"] - 1e-9),
                "petz_over_opt_rate_ratio_range": [
                    pz["rate_1mFavg"] / rate(ub) if rate(ub) > 0 else None,
                    pz["rate_1mFavg"] / rate(lb) if rate(lb) > 0 else None],
                "iters": iters, "wall_s": round(wall, 1),
            }
            res["cells"][key] = cell
            with open(OUT, "w") as f:
                json.dump(res, f, indent=2)
            print(f"{key}: opt rate in [{rate(ub):.6e}, {rate(lb):.6e}] "
                  f"petz={pz['rate_1mFavg']:.4e} proj="
                  f"{proj['rate_1mFavg']:.4e} ({iters} it, {wall:.0f}s)",
                  flush=True)

    # ordering summary with certified intervals
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
            # certified ordering: trained_rate_lb > 513_rate_ub ?
            row[name] = {
                "rate_interval": [c["optimal_rate_from_ub_Fe"],
                                  c["optimal_rate_from_lb_Fe"]],
                "ratio_vs_513_interval": [
                    c["optimal_rate_from_ub_Fe"]
                    / ref["optimal_rate_from_lb_Fe"],
                    c["optimal_rate_from_lb_Fe"]
                    / ref["optimal_rate_from_ub_Fe"]],
                "certified_513_wins": bool(
                    c["optimal_rate_from_ub_Fe"]
                    > ref["optimal_rate_from_lb_Fe"]),
            }
        summary[f"s{s:g}"] = {
            "ref_513_rate_interval": [ref["optimal_rate_from_ub_Fe"],
                                      ref["optimal_rate_from_lb_Fe"]],
            "trained": row}
    res["summary"] = summary
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(summary, indent=1))
    print("done")


if __name__ == "__main__":
    main()
