#!/usr/bin/env python3
"""Recovery maps and figures of merit for the calibrated-channel benchmark.

Everything here is a CPTP map or a functional that is linear in the density
matrix, so that recovery composed with one noise round is itself a quantum
channel and the exact evaluation of paper Sec. V B applies (a shot-by-shot
decoder such as ``weighted_map`` is nonlinear in the state and its LER
depends on how the channel is unraveled; it is not used on this benchmark).

(a) Projective syndrome decoder (paper App. F, "The decoder construction").
    Input: the weight-<=2 correction set of the completed channel basis
    (221 operators at n = 5, q = 3), ordered by DESCENDING prior under the
    calibrated channel.  Walking down the list, the images E_k|psi_i> of
    each correction are orthogonalized against the subspaces already
    claimed by more probable corrections; the residual directions whose
    singular values exceed a tolerance span the syndrome sector of E_k, with
    projector Pi_k (a correction whose images are fully claimed receives no
    sector).  The decoder is the projective measurement {Pi_k} u {Pi_rest},
    Pi_rest the unclaimed remainder of Hilbert space, followed by the
    unitary correction C_k = E_k^dagger on outcome k and the IDENTITY on
    Pi_rest.  The sectors are orthogonal and every correction in the
    closure basis is unitary, so the map is exactly CPTP
    (``tests/test_projective_decode.py::test_instrument_cptp_dense``).
    The per-correction residual / overlap record is returned so the
    construction can be inspected.  At n = 5, q = 3 the central code gives
    81 three-dimensional sectors; [[5,1,3]]_{Z_3} gives 74 / 7 / 7 sectors
    of dimension 3 / 2 / 1 at s = 1 (88 in all) and 92 sectors at s = 0.5;
    in both cases the sectors claim the full 243-dimensional space.

(b) Exact Petz recovery (App. F Eq. (F2)) for the FULL noise round N (not
    a weight-<=2 truncation): with codespace projector P the Petz map is
        R(X) = P N^dagger( N(P)^{-1/2} X N(P)^{-1/2} ) P,
    applied here with N and N^dagger as exact superoperators from
    ``exact_channel`` (no Kraus enumeration needed).  The inverse square
    root lives on supp N(P); output mass outside the support is scored as
    unrecovered (fidelity contribution 0), the standard completion-agnostic
    lower-bound convention -- the leak per codeword is reported.

(c) Figures of merit, all exact and linear in rho: the entanglement
    fidelity of the recovered channel Lambda o N on the K-dimensional
    logical space,
        F_e = (1/K^2) sum_ij <psi_i| Lambda(N(|psi_i><psi_j|)) |psi_j>,
    the Haar-average fidelity via the affine relation
        F_avg = (K F_e + 1) / (K + 1),
    and the channel infidelity 1 - F_avg quoted throughout the paper.
    NOTE: these are channel infidelities, not shot fractions; they are not
    numerically comparable to the i.i.d. LERs of Sec. V A.

(d) Sampled estimator for n >= 8 (Sec. V B: "a Monte Carlo estimator
    constructed so that its expectation equals the exact value").  For a
    trajectory |phi> of either unraveling, the sector-averaged score matrix
        Sc[i,j] = sum_k <psi_i|C_k Pi_k|phi><phi|Pi_k C_k^dagger|psi_j>
                  + <psi_i|Pi_rest|phi><phi|Pi_rest|psi_j>
    is linear in the trajectory projector |phi><phi|, so its expectation
    over trajectories is the exact functional <psi_i|Lambda(N(rho))|psi_j>
    of the input rho = E|phi><phi|, independent of the unraveling.
    Off-diagonal input blocks |psi_i><psi_j| are not physical states, so
    they are reached by the polarization identity over the pure inputs
    (psi_i + psi_j)/sqrt2 and (psi_i + i psi_j)/sqrt2 -- every ensemble the
    noise function propagates is a physical pure state.

Consumers: ``src.central_comparison`` (Table I, the certified optimal
recovery), ``tests/test_projective_decode.py``, and the calibrated-channel
campaign scripts under ``results/campaigns`` (r8p5_instrument, r9_campaign,
r10_sampler, r11_gapfill, r12_scrutiny, w5_t0).
"""
from __future__ import annotations
from typing import Callable, List, Tuple

import numpy as np

from src.decoders._common import apply_local_np


# ── (a) syndrome sectors ─────────────────────────────────────────────


def build_syndrome_sectors(code_states: np.ndarray,
                           correction_set: List[Tuple],
                           d: int, n_qudit: int,
                           tol: float = 1e-8) -> dict:
    """Orthogonalized syndrome sectors from a prior-ordered correction set
    (paper App. F, "The decoder construction").

    Args:
      code_states: (K, d^n) orthonormal codewords.
      correction_set: list of (wires, C_dag, inverse_perm, prior_w) from
        build_weighted_correction_set (any order; sorted here by descending
        prior_w, stable in the original flat order).
      d, n_qudit: local dimension and register size.
      tol: absolute singular-value threshold for a residual direction to
        found a sector dimension (image columns have unit norm for the
        unitary closure corrections, so absolute == relative).  This is the
        "fixed numerical tolerance" of App. F.

    Returns dict:
      'Ball': (d^n, R) stacked orthonormal sector bases (claimed subspace),
          R = sum of sector dimensions; Pi_rest = 1 - Ball Ball^dagger.
      'sectors': list of dicts per formed sector:
          'corr_idx' (index into the SORTED order), 'orig_idx' (input order),
          'wires', 'C_dag', 'inv', 'prior', 'dim', 'offset' (column offset
          in Ball), 'A' (dim_k, K) = B_k^dag E_k psi_i (own-sector image
          coordinates, used by every evaluator),
          'resid_svals' (kept + dropped singular values of the residual),
          'claimed_overlap_fro' (Frobenius norm of the image part already
          claimed by higher-prior sectors).
      'skipped': list of dicts (corr_idx, orig_idx, prior, max_resid_sval)
          for corrections whose images were entirely claimed (no sector).
      'Qpsi': (R, K) Ball^dag psi_i (rest-sector bookkeeping).
      'report': JSON-serializable summary (sector count and dimensions,
          claimed dimension R, rest dimension, tolerance).
      plus 'code_states', 'd', 'n_qudit' so the dict is self-contained.
    """
    K, dim = code_states.shape
    # Descending prior; Python's sort is stable so ties keep the flat
    # basis order.
    order = sorted(range(len(correction_set)),
                   key=lambda i: -correction_set[i][3])
    Q_blocks: List[np.ndarray] = []
    R_total = 0
    sectors = []
    skipped = []

    def _Q():
        return np.concatenate(Q_blocks, axis=1) if Q_blocks else None

    for rank, oi in enumerate(order):
        wires, C_dag, inv, w = correction_set[oi]
        E = C_dag.conj().T          # the error op; correction C = E^dag
        # X = [E psi_0, ..., E psi_{K-1}]: the images of the codewords.
        X = np.stack([apply_local_np(code_states[k], E, wires, inv,
                                     d, n_qudit) for k in range(K)]).T
        Q = _Q()
        if Q is None:
            resid = X
            overlap_fro = 0.0
        else:
            # Block Gram-Schmidt against everything already claimed.
            proj = Q.conj().T @ X
            resid = X - Q @ proj
            # one re-orthogonalization pass for numerical hygiene
            resid = resid - Q @ (Q.conj().T @ resid)
            overlap_fro = float(np.linalg.norm(proj))
        # Residual directions above tol found the sector; its dimension is
        # the numerical rank of the residual (K for a correction whose
        # images are entirely new, less when degenerate with earlier ones).
        U, s, _ = np.linalg.svd(resid, full_matrices=False)
        keep = s > tol
        m = int(keep.sum())
        if m == 0:
            skipped.append({'corr_idx': rank, 'orig_idx': oi,
                            'prior': float(w),
                            'max_resid_sval': float(s.max(initial=0.0))})
            continue
        B = np.ascontiguousarray(U[:, keep])
        A = B.conj().T @ X                       # (m, K)
        sectors.append({'corr_idx': rank, 'orig_idx': oi, 'wires': wires,
                        'C_dag': C_dag, 'inv': inv, 'prior': float(w),
                        'dim': m, 'offset': R_total, 'A': A,
                        'resid_svals': s.tolist(),
                        'claimed_overlap_fro': overlap_fro})
        Q_blocks.append(B)
        R_total += m

    Ball = _Q()
    if Ball is None:
        raise ValueError("no sectors formed -- empty correction set?")
    Qpsi = Ball.conj().T @ code_states.T          # (R, K)
    report = {
        'n_corrections': len(correction_set),
        'n_sectors': len(sectors),
        'n_skipped_fully_claimed': len(skipped),
        'claimed_dim_R': int(R_total),
        'hilbert_dim': int(dim),
        'rest_dim': int(dim - R_total),
        'sector_dims': [s_['dim'] for s_ in sectors],
        'sector_priors': [s_['prior'] for s_ in sectors],
        'sector_claimed_overlap_fro':
            [s_['claimed_overlap_fro'] for s_ in sectors],
        'tol': float(tol),
    }
    return {'Ball': Ball, 'sectors': sectors, 'skipped': skipped,
            'Qpsi': Qpsi, 'report': report, 'code_states': code_states,
            'd': d, 'n_qudit': n_qudit}


def apply_instrument_dense(inst: dict, sigma: np.ndarray) -> np.ndarray:
    """Dense Lambda(sigma) = sum_k C_k Pi_k sigma Pi_k C_k^dag
    + Pi_rest sigma Pi_rest: the decoder applied as a channel, column by
    column.  For tests and small n (materializes d^n x d^n terms); the
    figure-of-merit paths below never call this -- see
    ``instrument_channel_ops`` for the production form.

    Args:
      inst: output of ``build_syndrome_sectors``.
      sigma: (d^n, d^n) matrix.
    """
    d, n = inst['d'], inst['n_qudit']
    Ball = inst['Ball']
    out = np.zeros_like(sigma)
    for s_ in inst['sectors']:
        B = Ball[:, s_['offset']:s_['offset'] + s_['dim']]
        # Pi_k sigma Pi_k, then C_k (.) C_k^dag applied to columns and rows.
        collapsed = B @ (B.conj().T @ sigma @ B) @ B.conj().T
        cols = np.stack([apply_local_np(collapsed[:, c], s_['C_dag'],
                                        s_['wires'], s_['inv'], d, n)
                         for c in range(collapsed.shape[1])]).T
        rows = np.stack([apply_local_np(cols[r, :].conj(), s_['C_dag'],
                                        s_['wires'], s_['inv'], d, n).conj()
                         for r in range(cols.shape[0])])
        out += rows
    # Rest sector, identity correction: (1 - P) sigma (1 - P).
    P_rest = sigma - Ball @ (Ball.conj().T @ sigma)
    P_rest = P_rest - (P_rest @ Ball) @ Ball.conj().T
    out += P_rest
    return out


def instrument_channel_ops(inst: dict):
    """(Ball, W, blockmask) for the dense CPTP application of the projective
    decoder as a recovery CHANNEL (used for the repeated-round memory
    experiment of App. F): W stacks the corrected sector bases W_k = C_k B_k
    column-block-wise, so

      Lambda(sigma) = W blockdiag(Ball^dag sigma Ball) W^dag
                      + (1-P) sigma (1-P),   P = Ball Ball^dag,

    a handful of dim^2 x R gemms instead of the per-column
    apply_instrument_dense loop.  Pinned equal to apply_instrument_dense
    by tests/test_projective_decode.py::test_channel_ops_match_dense.

    Returns:
      Ball: (d^n, R) sector bases; W: (d^n, R) corrected bases;
      mask: (R, R) 0/1 block-diagonal mask selecting the within-sector
      blocks of Ball^dag sigma Ball.
    """
    from src.decoders._common import apply_local_np
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


def apply_instrument_channel(sigma: np.ndarray, Ball: np.ndarray,
                             W: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Lambda(sigma) via the precomputed (Ball, W, blockmask) of
    instrument_channel_ops. Works on any matrix (codeword blocks
    included), exact matrix arithmetic."""
    Y = Ball.conj().T @ sigma
    G = Y @ Ball
    out = W @ ((G * mask) @ W.conj().T)
    # (1-P) sigma (1-P) expanded: sigma - P sigma - sigma P + P sigma P.
    SB = sigma @ Ball
    out += sigma - Ball @ Y - SB @ Ball.conj().T + Ball @ (G @ Ball.conj().T)
    return out


def sector_outcome_probs(inst: dict, rho: np.ndarray) -> np.ndarray:
    """Outcome probabilities [p_1..p_S, p_rest] of the syndrome measurement
    on an exact density matrix rho: p_k = Tr(Pi_k rho), p_rest = Tr(rho) -
    sum_k p_k (linear, exact)."""
    Ball = inst['Ball']
    G = Ball.conj().T @ rho @ Ball
    diag = np.real(np.diag(G))
    probs = []
    for s_ in inst['sectors']:
        probs.append(diag[s_['offset']:s_['offset'] + s_['dim']].sum())
    probs.append(float(np.real(np.trace(rho)) - diag.sum()))
    return np.array(probs)


# ── (c) exact figures of merit ───────────────────────────────────────


def _fe_from_score_matrix(f: np.ndarray) -> float:
    """F_e = (1/K^2) sum_ij f[i,j] with f[i,j] = <psi_i|Lambda(rho_ij)|psi_j>,
    rho_ij = N(|psi_i><psi_j|): the entanglement fidelity of Lambda o N with
    respect to the maximally entangled state on the logical space."""
    K = f.shape[0]
    return float(np.real(f.sum()) / K ** 2)


def favg_from_fe(F_e: float, K: int) -> float:
    """Haar-average fidelity from entanglement fidelity,
    F_avg = (K F_e + 1) / (K + 1) (standard affine relation for a K-dim
    logical space; Horodecki et al. 1999, Nielsen 2002)."""
    return (K * F_e + 1.0) / (K + 1.0)


def projective_map_score_matrix(inst: dict, blocks: dict) -> np.ndarray:
    """f[i,j] = <psi_i| Lambda(blocks[i,j]) |psi_j> for the projective
    syndrome decoder, exact matrix arithmetic.

    Args:
      inst: output of ``build_syndrome_sectors``.
      blocks: ``blocks[(i, j)] = N(|psi_i><psi_j|)`` for i <= j (from
        exact_channel.evolve_code_blocks); i > j filled by Hermiticity.

    Returns:
      (K, K) complex Hermitian score matrix.

    Sector k contributes <psi_i|C_k Pi_k sig Pi_k C_k^dag|psi_j>
    = A_k[:, i]^dag G_kk A_k[:, j] with A_k = B_k^dag E_k psi (own-sector
    image coordinates, since C_k^dag psi_i = E_k psi_i) and G_kk the
    sector's diagonal block of Ball^dag sig Ball.  The rest sector with
    identity correction contributes <psi_i|(1-P) sig (1-P)|psi_j>, expanded
    into the four terms sij - t1 - t2 + t3 below using Qpsi = Ball^dag psi.
    """
    code_states = inst['code_states']
    K = code_states.shape[0]
    Ball, Qpsi = inst['Ball'], inst['Qpsi']
    f = np.zeros((K, K), dtype=complex)
    for i in range(K):
        for j in range(i, K):
            sig = blocks[(i, j)]
            Y = sig @ Ball
            G = Ball.conj().T @ Y                # (R, R)
            val = 0.0 + 0.0j
            for s_ in inst['sectors']:
                sl = slice(s_['offset'], s_['offset'] + s_['dim'])
                A = s_['A']                       # (m, K)
                val += A[:, i].conj() @ G[sl, sl] @ A[:, j]
            # rest sector, identity correction
            psi_i, psi_j = code_states[i], code_states[j]
            sij = psi_i.conj() @ (sig @ psi_j)
            t1 = Qpsi[:, i].conj() @ (Ball.conj().T @ (sig @ psi_j))
            t2 = (psi_i.conj() @ Y) @ Qpsi[:, j]
            t3 = Qpsi[:, i].conj() @ G @ Qpsi[:, j]
            val += sij - t1 - t2 + t3
            f[i, j] = val
            if j > i:
                f[j, i] = np.conj(val)
    return f


def entanglement_fidelity_projective(inst: dict, blocks: dict) -> dict:
    """Exact figures of merit of the projective syndrome decoder after one
    noise round.

    Args:
      inst: output of ``build_syndrome_sectors`` / ``build_projective_instrument``.
      blocks: codeword blocks from ``exact_channel.evolve_code_blocks``.

    Returns:
      dict with 'F_e' (entanglement fidelity), 'F_avg' (Haar-average
      fidelity), 'rate_1mFavg' (the channel infidelity 1 - F_avg quoted in
      the paper) and 'score_matrix_real' (Re f as nested lists).
    """
    f = projective_map_score_matrix(inst, blocks)
    K = inst['code_states'].shape[0]
    F_e = _fe_from_score_matrix(f)
    F_avg = favg_from_fe(F_e, K)
    return {'F_e': F_e, 'F_avg': F_avg, 'rate_1mFavg': 1.0 - F_avg,
            'score_matrix_real': np.real(f).tolist()}


# ── (b) exact Petz on the full dose ──────────────────────────────────


def build_petz_exact_dose(code_states: np.ndarray, steps: List[dict],
                          d: int, n_qudit: int,
                          rcond: float = 1e-12) -> dict:
    """N(P)^{-1/2} on supp N(P), with N the exact full noise round
    (the ingredient of the Petz map, App. F Eq. (F2)).

    Args:
      code_states: (K, d^n) orthonormal codewords; P = sum_k |psi_k><psi_k|.
      steps: noise-round steps from ``exact_channel.build_dose_steps``.
      d, n_qudit: local dimension and register size.
      rcond: relative eigenvalue cutoff defining the support -- eigenvalues
        of N(P) above rcond * max are inverted, the rest are dropped
        (Moore-Penrose inverse square root, no Tikhonov regularization).

    Returns:
      dict with 'inv_sqrt' ((d^n, d^n) Hermitian N(P)^{-1/2} on the
      support, zero off it), 'support_rank', 'support_frac' (rank / d^n),
      'eigs_kept_minmax', and the ('steps', 'd', 'n_qudit') needed to apply
      N^dagger later.
    """
    from src.decoders.exact_channel import apply_dose
    K, dim = code_states.shape
    P_C = code_states.T @ code_states.conj()      # (dim, dim), rank K
    NPC = apply_dose(P_C, steps, d, n_qudit)
    evals, W = np.linalg.eigh(NPC)
    keep = evals > rcond * evals.max()
    lam = evals[keep]
    Wk = W[:, keep]
    inv_sqrt = (Wk * (1.0 / np.sqrt(lam))[None, :]) @ Wk.conj().T
    supp_mass_of_identity = float(keep.sum()) / dim
    return {'inv_sqrt': inv_sqrt, 'support_rank': int(keep.sum()),
            'support_frac': supp_mass_of_identity,
            'eigs_kept_minmax': (float(lam.min()), float(lam.max())),
            'steps': steps, 'd': d, 'n_qudit': n_qudit}


def entanglement_fidelity_petz(petz: dict, code_states: np.ndarray,
                               blocks: dict) -> dict:
    """Exact figures of merit of the Petz recovery R o N (App. F Eq. (F2))
    via exact superoperators.

    For each codeword block sig = N(|psi_i><psi_j|):
        R(sig) = P N^dagger( S sig S ) P,   S = N(P)^{-1/2},
    and f[i,j] = <psi_i| R(sig) |psi_j> = <psi_i| N^dagger(S sig S) |psi_j>
    (the outer P's are absorbed by the codeword sandwich).  Mass of N(rho)
    outside supp N(P) is scored as unrecovered and reported per codeword
    as 'leak' = 1 - Tr[R(N(|psi_i><psi_i|))].

    Args:
      petz: output of ``build_petz_exact_dose``.
      code_states: (K, d^n) orthonormal codewords.
      blocks: codeword blocks from ``exact_channel.evolve_code_blocks``.

    Returns:
      dict with 'F_e', 'F_avg', 'rate_1mFavg', 'score_matrix_real',
      'recovery_leak_per_codeword' (list of K floats) and 'support_rank'.
    """
    from src.decoders.exact_channel import apply_dose
    S_half_inv = petz['inv_sqrt']
    steps, d, n = petz['steps'], petz['d'], petz['n_qudit']
    K = code_states.shape[0]
    f = np.zeros((K, K), dtype=complex)
    leaks = []
    for i in range(K):
        for j in range(i, K):
            sig = blocks[(i, j)]
            Y = S_half_inv @ sig @ S_half_inv
            Z = apply_dose(Y, steps, d, n, adjoint=True)
            val = code_states[i].conj() @ (Z @ code_states[j])
            f[i, j] = val
            if j > i:
                f[j, i] = np.conj(val)
            if i == j:
                # recovered mass Tr[R(N(rho_ii))] = Tr[P_C Z P_C]
                supp_mass = float(np.real(
                    np.einsum('ki,ik->', code_states.conj(),
                              Z @ code_states.T)))
                leaks.append(1.0 - supp_mass)
    F_e = _fe_from_score_matrix(f)
    F_avg = favg_from_fe(F_e, K)
    return {'F_e': F_e, 'F_avg': F_avg, 'rate_1mFavg': 1.0 - F_avg,
            'score_matrix_real': np.real(f).tolist(),
            'recovery_leak_per_codeword': leaks,
            'support_rank': petz['support_rank']}


# ── (d) sampled sector-averaged estimator (n >= 8) ───────────────────


def _pair_ensembles(code_states: np.ndarray):
    """Input ensembles for the polarization identity: K diagonal states
    plus, per pair i<j, (psi_i+psi_j)/sqrt2 and (psi_i+i psi_j)/sqrt2.

    Returns a list of (kind, i, j, chi) with kind in {'diag','plus','yplus'}
    and chi the (d^n,) pure input state; K + K(K-1) ensembles in all."""
    K = code_states.shape[0]
    ens = []
    for i in range(K):
        ens.append(("diag", i, i, code_states[i]))
    for i in range(K):
        for j in range(i + 1, K):
            ens.append(("plus", i, j,
                        (code_states[i] + code_states[j]) / np.sqrt(2.0)))
            ens.append(("yplus", i, j,
                        (code_states[i] + 1j * code_states[j]) / np.sqrt(2.0)))
    return ens


def sampled_score_matrix(inst: dict, chi: np.ndarray, noise_fn: Callable,
                         n_traj: int, rng: np.random.Generator,
                         batch: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Monte-Carlo estimate of f[i,j] = <psi_i|Lambda(N(|chi><chi|))|psi_j>
    via the per-trajectory sector-averaged score matrix.

    Args:
      inst: output of ``build_syndrome_sectors``.
      chi: (d^n,) pure input state.
      noise_fn: ``noise_fn(state, rng) -> trajectory`` for one unraveling of
        the channel (either ``make_channel_v2_noise_fn`` or
        ``make_meth_mc_noise_fn``; the expectation is the same).
      n_traj: number of trajectories; rng: generator; batch: trajectories
        scored per matmul.

    Returns:
      (mean (K,K) complex, SE (K,K) real of the real part).

    Per trajectory phi, with U = Ball^dag phi:
      sector k:  c_k = A_k^dag U_k  (K,)  gives  Sc += c_k c_k^dag,
      rest:      c_rest = psi^dag phi - Qpsi^dag U  gives  Sc += c c^dag,
    i.e. Sc[i,j] = sum_k <psi_i|C_k Pi_k|phi><phi|Pi_k C_k^dag|psi_j> +
    the identity-corrected rest term, linear in |phi><phi|.
    """
    code_states = inst['code_states']
    K, dim = code_states.shape
    Ball, Qpsi = inst['Ball'], inst['Qpsi']
    acc = np.zeros((K, K), dtype=complex)
    acc2 = np.zeros((K, K), dtype=float)
    done = 0
    while done < n_traj:
        b = min(batch, n_traj - done)
        Phi = np.empty((dim, b), dtype=complex)
        for t in range(b):
            Phi[:, t] = noise_fn(chi, rng)
        U = Ball.conj().T @ Phi                    # (R, b)
        Sc = np.zeros((K, K, b), dtype=complex)
        for s_ in inst['sectors']:
            sl = slice(s_['offset'], s_['offset'] + s_['dim'])
            Ck = s_['A'].conj().T @ U[sl, :]       # (K, b)
            Sc += Ck[:, None, :] * Ck.conj()[None, :, :]
        Crest = code_states.conj() @ Phi - Qpsi.conj().T @ U   # (K, b)
        Sc += Crest[:, None, :] * Crest.conj()[None, :, :]
        acc += Sc.sum(axis=2)
        acc2 += (np.real(Sc) ** 2).sum(axis=2)
        done += b
    mean = acc / n_traj
    var = acc2 / n_traj - np.real(mean) ** 2
    se = np.sqrt(np.clip(var, 0.0, None) / n_traj)
    return mean, se


def sampled_entanglement_fidelity(inst: dict, noise_fn: Callable,
                                  n_traj: int, seed: int,
                                  batch: int = 64) -> dict:
    """Unraveling-independent sampled F_e for registers too large for the
    exact engine (n >= 8): run every polarization ensemble through
    `noise_fn` (either unraveling) and assemble F_e exactly as the exact
    evaluator does, with each matrix element linear in the trajectory
    projector.

    Args:
      inst: output of ``build_syndrome_sectors`` / ``build_projective_instrument``.
      noise_fn: trajectory sampler ``noise_fn(state, rng)``.
      n_traj: trajectories per ensemble; seed: ``default_rng`` seed.

    Returns:
      dict with 'F_e', 'F_e_se', 'F_avg', 'F_avg_se', 'rate_1mFavg',
      'n_traj_per_ensemble', 'n_ensembles' (= K + K(K-1)).

    Polarization identity: with f(chi)[i,j] := <psi_i|Lambda(N(|chi><chi|))|psi_j>
    linear in |chi><chi|, the off-diagonal block element
    f_ij := <psi_i|Lambda(N(|psi_i><psi_j|))|psi_j> is recovered from the
    four pure inputs psi_i, psi_j, (psi_i+psi_j)/sqrt2, (psi_i+i psi_j)/sqrt2 as
        f_ij = 1/2 [ (2 f(plus) - f(ii) - f(jj)) + i (2 f(yplus) - f(ii) - f(jj)) ]
    (all evaluated at matrix index [i,j]); F_e then sums the K diagonal
    elements and 2 Re f_ij over pairs.  The SE is first-order propagation
    of the per-ensemble standard errors.
    """
    code_states = inst['code_states']
    K = code_states.shape[0]
    rng = np.random.default_rng(seed)
    means = {}
    ses = {}
    for kind, i, j, chi in _pair_ensembles(code_states):
        m, se = sampled_score_matrix(inst, chi, noise_fn, n_traj, rng,
                                     batch=batch)
        means[(kind, i, j)] = m
        ses[(kind, i, j)] = se

    total = 0.0 + 0.0j
    var_sum = 0.0
    for i in range(K):
        m = means[("diag", i, i)]
        total += m[i, i]
        var_sum += ses[("diag", i, i)][i, i] ** 2
    for i in range(K):
        for j in range(i + 1, K):
            A = means[("plus", i, j)][i, j]
            B = means[("yplus", i, j)][i, j]
            fii = means[("diag", i, i)][i, j]
            fjj = means[("diag", j, j)][i, j]
            f_ij = 0.5 * ((2.0 * A - fii - fjj)
                          + 1j * (2.0 * B - fii - fjj))
            total += 2.0 * np.real(f_ij)
            # first-order SE propagation on Re f_ij
            var_ij = (ses[("plus", i, j)][i, j] ** 2
                      + 0.5 * (ses[("diag", i, i)][i, j] ** 2
                               + ses[("diag", j, j)][i, j] ** 2)
                      + ses[("yplus", i, j)][i, j] ** 2)
            var_sum += 4.0 * var_ij
    F_e = float(np.real(total)) / K ** 2
    se_Fe = float(np.sqrt(var_sum)) / K ** 2
    F_avg = favg_from_fe(F_e, K)
    return {'F_e': F_e, 'F_e_se': se_Fe, 'F_avg': F_avg,
            'F_avg_se': se_Fe * K / (K + 1.0),
            'rate_1mFavg': 1.0 - F_avg, 'n_traj_per_ensemble': int(n_traj),
            'n_ensembles': K + K * (K - 1)}


# ── convenience: decoder for a (code, channel setting) ───────────────


def build_projective_instrument(code_states: np.ndarray, d: int, n: int,
                                eta: float, sigma_a2: float = 0.0,
                                phi: float = 0.0, distance: int = 3,
                                tol: float = 1e-8) -> dict:
    """The paper's App. F projective syndrome decoder for one code and one
    channel setting: correction set from the weight-<=2 closure basis with
    the calibrated-channel prior at (eta, sigma_a2, phi), ordered by
    descending prior -> sectors.

    Args:
      code_states: (K, d^n) orthonormal codewords.
      d, n: local dimension and register size.
      eta, sigma_a2, phi: channel parameters (see ``exact_channel``); the
        prior comes from ``src.meth_weights.compute_meth_group_weights``
        (dephasing only) or ``compute_meth_group_weights_v2`` (with the
        angle-fluctuation term), both at Kraus order n_max = 2.
      distance: closure-basis distance (3: weight-<=2 basis).
      tol: sector tolerance passed to ``build_syndrome_sectors``.

    Returns:
      the ``build_syndrome_sectors`` dict.
    """
    from src.errors import ErrorModel
    from src.meth_weights import (compute_meth_group_weights,
                                  compute_meth_group_weights_v2)
    from src.decoders.weighted_map import build_weighted_correction_set
    E_full = ErrorModel(d=d, n_qudit=n, distance=distance,
                        closed=True).build_grouped(verbose=False)
    if sigma_a2 > 0.0:
        _, prior = compute_meth_group_weights_v2(d, n, distance=distance,
                                                 eta=eta, sigma_a2=sigma_a2,
                                                 phi=phi, n_max=2)
    else:
        _, prior = compute_meth_group_weights(d, n, distance=distance,
                                              eta=eta, n_max=2)
    corr = build_weighted_correction_set(E_full, prior, n, d, max_weight=2)
    return build_syndrome_sectors(code_states, corr, d, n, tol=tol)
