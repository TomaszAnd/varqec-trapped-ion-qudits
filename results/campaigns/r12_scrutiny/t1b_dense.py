#!/usr/bin/env python3
"""Independent re-derivation of the central cell at s=0.5.

From-scratch evaluation of the v2 dose + both instruments for the
central code and [[5,1,3]]_Z3 at s = 0.5, through a DIFFERENT code
path than src/decoders/exact_channel + projective_decode:

  - Dose: Gauss-Hermite quadrature of the Gaussian phase averages
    (numerical integral over per-qudit phases and the pair jitter
    angle, 32 nodes), applied as dense unitary conjugations of the
    full 243x243 density matrix. No Schur masks, no per-wire
    tensordots. Role couplings re-coded from the paper's channel
    definitions; jitter generator re-built from S_phi = X01 x I +
    I x X01 and U(dtheta) = exp(-i dtheta S^2/4).
  - Projective instrument: sectors re-derived by an independently
    coded prior-ordered Gram-Schmidt on dense images; instrument
    applied as the explicit dense CPTP sum
    Lambda(sigma) = sum_k C_k Pi_k sigma Pi_k C_k^dag + Pi_r sigma Pi_r;
    F_e from the definitional formula
    F_e = (1/K^2) sum_ij <psi_i|Lambda(N(|psi_i><psi_j|))|psi_j>.
  - Petz: P_C -> N(P_C) by quadrature dose, scipy.linalg.eigh inverse
    square root (same rcond convention 1e-12 relative), adjoint dose =
    reversed step order (each Gaussian factor is self-adjoint).

Shared inputs (instrument DEFINITION, not numerics): the codewords,
and the weight-<=2 closure correction set with its channel prior from
build_weighted_correction_set (the instrument is defined by that
ordered set). Dense correction matrices are built column-by-column and
all subsequent arithmetic is dense.

Agreement target: engine values from r9_campaign/t1_results.json.
Writes t1b_results.json with per-cell deviations.
"""
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

import numpy as np
from numpy.polynomial.hermite import hermgauss
import scipy.linalg as sla

import src.jax_backend  # noqa: F401  (x64 first)
from src.catalog import five_qudit_code_states
from src.decoders._common import encoder_forward, apply_local_np
from src.errors import ErrorModel
from src.meth_weights import compute_meth_group_weights_v2
from src.decoders.weighted_map import build_weighted_correction_set

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "t1b_results.json")
BR = os.path.join(REPO, "results", "campaigns")

D, N = 3, 5
DIM = D ** N
S = 0.5
SIGMA_P2 = 0.073 * S
SIGMA_A2 = 0.047 * S
ETA = float(np.exp(-SIGMA_P2))
GP = [(i, j) for i in range(N) for j in range(i + 1, N)]
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]
NODES = 32

# ── dose via Gauss-Hermite quadrature ────────────────────────────────

_x, _w = hermgauss(NODES)


def _gauss_nodes(sigma2):
    """phi values and probability weights for phi ~ N(0, sigma2)."""
    return _x * np.sqrt(2.0 * sigma2), _w / np.sqrt(np.pi)


def role_couplings(role, d=D):
    """Sec III / App C phase-coupling vectors, transitions (0, 1)."""
    if role == "control":            # f_j = j, f[ctrl_level=0] = 0
        f = np.arange(d, dtype=float)
        f[0] = 0.0
    elif role == "target":           # 3 everywhere, 1 on ctrl, 2 on tgt
        f = np.full(d, 3.0)
        f[0], f[1] = 1.0, 2.0
    else:                            # spectator: f_j = j
        f = np.arange(d, dtype=float)
    return f


def qudit_phase_unitary(f, phi, q):
    """Dense diag unitary e^{i phi F_q} on the full register."""
    dvec = np.ones(1, dtype=complex)
    for k in range(N):
        loc = np.exp(1j * phi * f) if k == q else np.ones(D, dtype=complex)
        dvec = np.kron(dvec, loc)
    return dvec                       # diagonal entries


def jitter_unitary_diag_ev(phi_ms=0.0, d=D):
    """Eigen-decomposition of S_phi^2 on the pair space; returns (V, g)
    with U(dtheta) = V diag(e^{-i dtheta g}) V^dag, g = eigs(S^2)/4."""
    X01 = np.zeros((d, d), dtype=complex)
    X01[0, 1] = np.exp(1j * phi_ms)
    X01[1, 0] = np.exp(-1j * phi_ms)
    Sp = np.kron(X01, np.eye(d)) + np.kron(np.eye(d), X01)
    S2 = Sp @ Sp
    g, V = np.linalg.eigh(S2)
    return V, g / 4.0


def embed_pair(op9, qa, qb):
    """Embed a (d^2, d^2) pair operator on wires (qa, qb) into DIM,
    via kron with identity and axis permutation."""
    big = np.kron(op9, np.eye(D ** (N - 2), dtype=complex))
    T = big.reshape([D] * (2 * N))
    wires = [qa, qb] + [q for q in range(N) if q not in (qa, qb)]
    perm = [0] * N
    for pos, w in enumerate(wires):
        perm[w] = pos
    axes = [perm[q] for q in range(N)]
    T = T.transpose(axes + [a + N for a in axes])
    return np.ascontiguousarray(T.reshape(DIM, DIM))


def dose_step(rho, qa, qb, adjoint=False):
    """One gate step: per-qudit role dephasing + pair jitter, each a
    Gauss-Hermite average of unitary conjugations."""
    ops = []
    # per-qudit dephasing (independent phases -> sequential averages)
    for q in range(N):
        role = ("control" if q == qa else
                "target" if q == qb else "spectator")
        f = role_couplings(role)
        phis, ws = _gauss_nodes(SIGMA_P2)
        ops.append(("deph", q, f, phis, ws))
    # pair jitter
    V, g = jitter_unitary_diag_ev()
    dts, wj = _gauss_nodes(SIGMA_A2)
    ops.append(("jit", (qa, qb), (V, g), dts, wj))
    if adjoint:
        ops = ops[::-1]
    for kind, where, dat, nodes, ws in ops:
        acc = np.zeros_like(rho)
        if kind == "deph":
            f = dat
            for phi, w in zip(nodes, ws):
                dv = qudit_phase_unitary(f, -phi if adjoint else phi, where)
                acc += w * (dv[:, None] * rho * dv.conj()[None, :])
        else:
            V, g = dat
            qa_, qb_ = where
            for dt, w in zip(nodes, ws):
                sgn = 1.0 if adjoint else -1.0
                U9 = (V * np.exp(sgn * 1j * dt * g)[None, :]) @ V.conj().T
                U = embed_pair(U9, qa_, qb_)
                acc += w * (U @ rho @ U.conj().T)
        rho = acc
    return rho


def apply_dose_quad(rho, adjoint=False):
    steps = GP[::-1] if adjoint else GP
    for (qa, qb) in steps:
        rho = dose_step(rho, qa, qb, adjoint=adjoint)
    return rho


# ── independent instrument construction ──────────────────────────────


def dense_op(C_dag, wires, inv):
    """Dense (DIM, DIM) matrix of the correction C = ... via column
    application of apply_local_np (op semantics from the shared
    definition; arithmetic afterwards is dense)."""
    M = np.zeros((DIM, DIM), dtype=complex)
    eye = np.eye(DIM, dtype=complex)
    for c in range(DIM):
        M[:, c] = apply_local_np(eye[:, c], C_dag, wires, inv, D, N)
    return M


def build_sectors_dense(code_states, corr, tol=1e-8):
    order = sorted(range(len(corr)), key=lambda i: -corr[i][3])
    Q = np.zeros((DIM, 0), dtype=complex)
    sectors = []
    for oi in order:
        wires, C_dag, inv, w = corr[oi]
        # stored matrix = the correction C = E^dag (source convention:
        # E = C_dag.conj().T in build_syndrome_sectors)
        Cd = dense_op(C_dag, wires, inv)          # dense correction C
        Ek = Cd.conj().T                          # dense error op E
        X = Ek @ code_states.T                    # (DIM, K)
        resid = X - Q @ (Q.conj().T @ X)
        resid = resid - Q @ (Q.conj().T @ resid)
        U, s, _ = np.linalg.svd(resid, full_matrices=False)
        m = int((s > tol).sum())
        if m == 0:
            continue
        B = U[:, :m]
        sectors.append({"B": B, "C": Ek.conj().T, "prior": float(w),
                        "dim": m})
        Q = np.concatenate([Q, B], axis=1)
    return sectors, Q


def instrument_apply_dense(sectors, Q, sigma):
    out = np.zeros_like(sigma)
    for s_ in sectors:
        Pi = s_["B"] @ s_["B"].conj().T
        out += s_["C"] @ Pi @ sigma @ Pi @ s_["C"].conj().T
    Pr = np.eye(DIM) - Q @ Q.conj().T
    out += Pr @ sigma @ Pr
    return out


def main():
    res = {"meta": {"s": S, "eta": ETA, "sigma_a2": SIGMA_A2,
                    "nodes": NODES,
                    "method": "Gauss-Hermite quadrature dose + dense "
                              "instrument arithmetic (independent path)",
                    "reference": "r9_campaign/t1_results.json"},
           "cells": {}}
    r9 = json.load(open(os.path.join(BR, "r9_campaign", "t1_results.json")))

    # shared instrument definition: correction set + channel prior
    E_full = ErrorModel(d=D, n_qudit=N, distance=3,
                        closed=True).build_grouped(verbose=False)
    _, prior = compute_meth_group_weights_v2(D, N, distance=3, eta=ETA,
                                             sigma_a2=SIGMA_A2, phi=0.0,
                                             n_max=2)
    corr = build_weighted_correction_set(E_full, prior, N, D, max_weight=2)
    print(f"correction set: {len(corr)}", flush=True)

    codes = {
        "headline_A_L2_s100": encoder_forward(
            os.path.join(BR, "r1_campaign", "t1_codes", "A_L2_s100.npz"),
            CONNS)[0],
        "five_one_three_Z3": five_qudit_code_states(D),
    }
    for name, cs in codes.items():
        t0 = time.time()
        K = cs.shape[0]
        # dose on all blocks
        blocks = {}
        for i in range(K):
            for j in range(i, K):
                rho = np.outer(cs[i], cs[j].conj())
                blocks[(i, j)] = apply_dose_quad(rho)
        # projective
        sectors, Q = build_sectors_dense(cs, corr)
        f = np.zeros((K, K), dtype=complex)
        for i in range(K):
            for j in range(i, K):
                lam = instrument_apply_dense(sectors, Q, blocks[(i, j)])
                f[i, j] = cs[i].conj() @ (lam @ cs[j])
                f[j, i] = np.conj(f[i, j])
        F_e = float(np.real(f.sum()) / K ** 2)
        proj_rate = 1.0 - (K * F_e + 1.0) / (K + 1.0)
        # Petz
        P_C = cs.T @ cs.conj()
        NPC = apply_dose_quad(P_C)
        evals, W = sla.eigh(NPC)
        keep = evals > 1e-12 * evals.max()
        inv_sqrt = (W[:, keep] * (1.0 / np.sqrt(evals[keep]))[None, :]) \
            @ W[:, keep].conj().T
        fz = np.zeros((K, K), dtype=complex)
        for i in range(K):
            for j in range(i, K):
                Y = inv_sqrt @ blocks[(i, j)] @ inv_sqrt
                Z = apply_dose_quad(Y, adjoint=True)
                fz[i, j] = cs[i].conj() @ (Z @ cs[j])
                fz[j, i] = np.conj(fz[i, j])
        F_ez = float(np.real(fz.sum()) / K ** 2)
        petz_rate = 1.0 - (K * F_ez + 1.0) / (K + 1.0)

        ref = r9["cells"][f"{name}@v2_s0.500"]
        res["cells"][name] = {
            "proj_rate_independent": proj_rate,
            "petz_rate_independent": petz_rate,
            "proj_rate_engine": ref["projective_rate"],
            "petz_rate_engine": ref["petz_rate"],
            "proj_abs_dev": abs(proj_rate - ref["projective_rate"]),
            "petz_abs_dev": abs(petz_rate - ref["petz_rate"]),
            "n_sectors": len(sectors),
            "claimed_dim": int(Q.shape[1]),
            "petz_support_rank": int(keep.sum()),
            "wall_s": round(time.time() - t0, 1),
        }
        with open(OUT, "w") as fo:
            json.dump(res, fo, indent=2)
        print(f"{name}: proj {proj_rate:.10e} (dev "
              f"{res['cells'][name]['proj_abs_dev']:.2e})  petz "
              f"{petz_rate:.10e} (dev "
              f"{res['cells'][name]['petz_abs_dev']:.2e})", flush=True)
    print("done")


if __name__ == "__main__":
    main()
