#!/usr/bin/env python3
"""Exact density-matrix engine for one round of the calibrated channel.

Implements the exact evaluation of paper Sec. V B ("Benchmarking on the
calibrated channel"): an n-qudit density matrix is evolved EXACTLY through
one noise round, i.e. one gate step of the calibrated gate-error channel
(Sec. III, Eq. gate-channel)

    N_gate = A_{sigma_a^2} o ( D^(c) (x) D^(t) (x) prod_{spectators} D^(sp) )

per coupled pair of the all-to-all graph, with the Kraus families of
App. B.  Because every factor of a gate step is a dephasing channel in a
known basis, its action on rho is a Schur (elementwise) product:

  - the per-qudit stochastic-phase dephasing channels D^(c), D^(t), D^(sp)
    are diagonal in the COMPUTATIONAL basis.  A qudit in role r has an
    integer coupling vector f_r (how often the gate lasers touch each
    level), and the Gaussian average multiplies the coherence between
    levels i and j by the mask

        C[i, j] = exp(-1/2 sigma_p^2 (f_i - f_j)^2),    sigma_p^2 = -ln(eta)

    (Sec. III: the coherence between levels whose couplings differ by
    Delta f decays by e^{-sigma_p^2 Delta f^2 / 2}; eta := e^{-sigma_p^2});
  - the MS angle-fluctuation channel A_{sigma_a^2} on the gate pair is a
    dephasing channel in the eigenbasis of the generator S_phi^2 (Sec. III),
    which is the PRODUCT basis V1 (x) V1 of single-qudit sigma_phi
    eigenvectors (``src.correlated_noise._ms_jitter_eigensystem``).  Its
    action is rotate-mask-rotate: conjugate the pair into that eigenbasis,
    multiply elementwise by the pair mask J[(ab),(cd)] =
    exp(-1/2 sigma_a^2 (g_ab - g_cd)^2) with g the S_phi^2/4 eigenvalues,
    rotate back.

Masks come in two modes:
  mode='exact'  (default): the analytic Gaussian multipliers above -- the
    UNTRUNCATED Gaussian-averaged channel.  Exactly trace preserving and
    exactly positive (a Gaussian kernel is a PSD mask, and the Schur product
    of PSD matrices is PSD).  This is the mode behind every calibrated-
    channel number of the paper.
  mode='kraus': masks C[i,j] = sum_m diag_m[i] conj(diag_m[j]) built from
    the truncated Kraus series at the trajectory samplers' n_max --
    reproduces the finite-n_max channel that the Monte-Carlo samplers
    (``src.simulation.make_channel_v2_noise_fn``, ``make_meth_mc_noise_fn``)
    unravel, to numerical precision.

Both trajectory samplers must agree with this engine on average
(``tests/test_exact_channel.py::test_trajectory_average_converges``), and
any figure of merit computed from rho by linear matrix arithmetic is
independent of the unraveling by construction.  (A shot-by-shot decoder
such as ``weighted_map`` is not such a functional, which is why the
calibrated-channel benchmarks use only the linear recovery maps of
``projective_decode``.)

The adjoint channel N^dagger (needed by the Petz recovery, App. F Eq. (F2))
is the same sequence in reversed step order with conjugated masks and
inverse pair rotations; the masks here are real, so each factor is
self-adjoint and only the ordering reverses.

Memory: an n=7, d=3 density matrix is 2187^2 complex128 = 76.6 MB and the
engine keeps O(1) such arrays, so the exact route covers n <= 7 at q = 3
(and q <= 5 at n = 5).  Do not call at n >= 8; use the sampled
sector-averaged estimator in ``projective_decode`` instead.

Consumers: ``src.central_comparison`` (Table I cells),
``src.decoders.projective_decode`` (codeword blocks, Petz adjoint),
``tests/test_exact_channel.py``, ``tests/test_projective_decode.py``, and the
calibrated-channel campaign scripts under ``results/campaigns``.
"""
from __future__ import annotations
from typing import List, Tuple

import numpy as np

from src.correlated_noise import (
    control_qudit_kraus, control_qudit_kraus_simplified, target_qudit_kraus,
    spectator_qudit_kraus, _ms_jitter_eigensystem, _kraus_diagonal,
)


# ── mask builders ────────────────────────────────────────────────────


def _mask_exact(couplings: np.ndarray, sigma2: float) -> np.ndarray:
    """Analytic Gaussian coherence mask C[i,j] = e^{-sigma2 (f_i-f_j)^2 / 2}.

    `couplings` is the vector f of per-level (or per-pair-eigenvalue)
    couplings and `sigma2` the phase variance; the diagonal is 1 (populations
    are untouched) and the mask is a Gaussian kernel, hence PSD."""
    f = np.asarray(couplings, dtype=float)
    return np.exp(-0.5 * sigma2 * (f[:, None] - f[None, :]) ** 2)


def _mask_from_diags(diags: List[np.ndarray]) -> np.ndarray:
    """Truncated-series mask C[i,j] = sum_m diag_m[i] conj(diag_m[j]) from the
    diagonals of a family of diagonal Kraus operators (the Schur multiplier
    of rho -> sum_m E_m rho E_m^dagger with E_m = diag(diag_m))."""
    D = np.stack(diags)                      # (m, dim)
    return (D.T @ D.conj())


def _role_couplings(d: int, role: str, ctrl_level: int, tgt_level: int,
                    noise_model: str) -> np.ndarray:
    """Coupling vector f_k of a qudit role (same conventions as the Kraus
    builders in ``src.correlated_noise`` and ``make_meth_mc_noise_fn`` --
    keep in sync).

    Paper App. B: control qudit f_j = j with the driven level set to 0
    ('simplified' uses a flat coupling 2 on the undriven levels), target
    qudit f = 1 on the control level, 2 on the target level, 3 elsewhere,
    spectator f_k = k."""
    if role == "control":
        if noise_model == "simplified":
            f = np.full(d, 2.0)
        else:
            f = np.arange(d, dtype=float)
        f[ctrl_level] = 0.0
    elif role == "target":
        f = np.full(d, 3.0)
        f[ctrl_level] = 1.0
        f[tgt_level] = 2.0
    elif role == "spectator":
        f = np.arange(d, dtype=float)
    else:
        raise ValueError(f"unknown role '{role}'")
    return f


def build_dose_steps(n_qudits: int, d: int,
                     gate_pairs: List[Tuple[int, int, int, int]],
                     eta: float, sigma_a2: float = 0.0, phi: float = 0.0,
                     mode: str = "exact",
                     n_max_dephasing: int = 5, n_max_jitter: int = 10,
                     noise_model: str = "physical") -> List[dict]:
    """Precompute the per-gate-step masks/rotations of one noise round.

    Args:
      n_qudits, d: register size and local dimension.
      gate_pairs: list of (qa, qb, ctrl_level, tgt_level), one per gate step
        in application order (the all-to-all graph gives C(n,2) steps; see
        ``scripts/benchmark_ler_meth_kraus.py::build_gate_pairs`` or
        ``src.central_comparison.gate_pairs``).
      eta: dephasing parameter eta = e^{-sigma_p^2} (0.9296 calibrated).
      sigma_a2: MS angle-fluctuation variance (0.047 calibrated); 0 turns
        the angle term off (dephasing-only channel).
      phi: MS phase, selects sigma_phi = cos(phi) X + sin(phi) Y on the
        driven (0, 1) levels.
      mode: 'exact' (analytic Gaussian masks) or 'kraus' (truncated series).
      n_max_dephasing, n_max_jitter: Kraus truncation orders for mode='kraus'.
      noise_model: 'physical' (level-resolved control coupling, the paper's
        channel) or 'simplified' (flat control coupling).

    Returns a list of step dicts:
      'deph': (n_qudits, d, d) per-qudit coherence masks (control on qa,
              target on qb, spectator elsewhere)
      'qa', 'qb': the coupled pair (qa = control, qb = target)
      'V1', 'jmask': pair-jitter eigenvectors and (d, d, d, d) mask, with
              jmask index order [a_ket, b_ket, a_bra, b_bra], a = qa factor;
              both None at sigma_a2 = 0
    """
    if mode not in ("exact", "kraus"):
        raise ValueError(f"mode must be 'exact' or 'kraus', got '{mode}'")
    sigma_p2 = -np.log(eta)

    # The angle-fluctuation mask is the same for every pair (same driven
    # levels, same phi), so it is built once and shared by all steps.
    jit = None
    if sigma_a2 > 0.0:
        V1, gvals = _ms_jitter_eigensystem(d, phi, 0, 1)
        if mode == "exact":
            J = _mask_exact(gvals, sigma_a2)
        else:
            diags = _kraus_diagonal(d * d, gvals, float(np.exp(-sigma_a2)),
                                    n_max_jitter)
            J = _mask_from_diags(diags)
        jit = (V1, np.ascontiguousarray(J.reshape(d, d, d, d)))

    if noise_model == "simplified":
        ctrl_fn = control_qudit_kraus_simplified
    else:
        ctrl_fn = control_qudit_kraus

    steps = []
    for qa, qb, ctrl_level, tgt_level in gate_pairs:
        masks = np.empty((n_qudits, d, d), dtype=complex)
        for q in range(n_qudits):
            role = ("control" if q == qa else
                    "target" if q == qb else "spectator")
            if mode == "exact":
                f = _role_couplings(d, role, ctrl_level, tgt_level, noise_model)
                masks[q] = _mask_exact(f, sigma_p2)
            else:
                if role == "control":
                    diags = ctrl_fn(d, ctrl_level, eta, n_max_dephasing)
                elif role == "target":
                    diags = target_qudit_kraus(d, ctrl_level, tgt_level, eta,
                                               n_max_dephasing)
                else:
                    diags = spectator_qudit_kraus(d, eta, n_max_dephasing)
                masks[q] = _mask_from_diags(diags)
        step = {'deph': masks, 'qa': int(qa), 'qb': int(qb)}
        if jit is not None:
            step['V1'], step['jmask'] = jit
        else:
            step['V1'], step['jmask'] = None, None
        steps.append(step)
    return steps


# ── application ──────────────────────────────────────────────────────
#
# rho is handled as a rank-2n tensor of shape (d,)*2n: axes 0..n-1 are the
# ket indices of qudits 0..n-1, axes n..2n-1 the bra indices.


def _apply_axis(T: np.ndarray, M: np.ndarray, axis: int) -> np.ndarray:
    """Apply a d x d matrix on one axis of a rank-2n tensor."""
    out = np.tensordot(M, T, axes=([1], [axis]))
    return np.moveaxis(out, 0, axis)


def _rotate_pair(rho_t: np.ndarray, V: np.ndarray, qa: int, qb: int,
                 n: int) -> np.ndarray:
    """Conjugate the (qa, qb) pair: rho -> (V (x) V) rho (V (x) V)^dag,
    acting on ket axes (qa, qb) and bra axes (n+qa, n+qb).

    Reference implementation of the rotation; the production path is
    ``_jitter_pair_front`` (same values, better memory layout)."""
    rho_t = _apply_axis(rho_t, V, qa)
    rho_t = _apply_axis(rho_t, V, qb)
    rho_t = _apply_axis(rho_t, V.conj(), n + qa)
    rho_t = _apply_axis(rho_t, V.conj(), n + qb)
    return rho_t


def _schur_qudit(rho_t: np.ndarray, mask: np.ndarray, q: int, n: int,
                 d: int) -> None:
    """In-place elementwise multiply by mask[i_q, j_q] on (ket q, bra q):
    the action of a computational-basis dephasing channel on qudit q."""
    shape = [1] * (2 * n)
    shape[q] = d
    shape[n + q] = d
    rho_t *= mask.reshape(shape)


def _schur_pair(rho_t: np.ndarray, jmask: np.ndarray, qa: int, qb: int,
                n: int, d: int) -> None:
    """In-place elementwise multiply by jmask[a_ket, b_ket, a_bra, b_bra]
    on axes (qa, qb, n+qa, n+qb); a = qa factor of the kron eigenbasis."""
    shape = [1] * (2 * n)
    shape[qa] = d
    shape[qb] = d
    shape[n + qa] = d
    shape[n + qb] = d
    # jmask axes are (qa, qb, n+qa, n+qb) in that order; reshape places its
    # dims at the non-1 slots in ascending axis order, so permute to match.
    axes_sorted = np.argsort([qa, qb, n + qa, n + qb])
    jm = np.transpose(jmask, axes_sorted)
    rho_t *= jm.reshape(shape)


def _jitter_pair_front(rho_t: np.ndarray, V1: np.ndarray, jmask: np.ndarray,
                       qa: int, qb: int, n: int, d: int) -> np.ndarray:
    """Rotate-mask-rotate of the pair angle-fluctuation channel:

        rho -> (V1 (x) V1) [ J o ((V1 (x) V1)^dag rho (V1 (x) V1)) ] (V1 (x) V1)^dag

    on the (qa, qb) pair, with the four pair axes moved to the FRONT of a
    contiguous layout first.  tensordot canonicalizes its operand layout
    internally before the gemm, so every contraction receives bit-identical
    values in the identical canonical layout as the reference _rotate_pair
    path -- outputs are bit-identical (pinned by
    tests/test_exact_channel.py::test_fast_jitter_bit_identical); only the
    cost of the internal transpose-copies changes (awkward mid-tensor
    strides -> large contiguous blocks)."""
    src = (qa, qb, n + qa, n + qb)
    T = np.ascontiguousarray(np.moveaxis(rho_t, src, (0, 1, 2, 3)))
    Vd = V1.conj().T
    # into the sigma_phi product eigenbasis ...
    T = _apply_axis(T, Vd, 0)
    T = _apply_axis(T, Vd, 1)
    T = _apply_axis(T, Vd.conj(), 2)
    T = _apply_axis(T, Vd.conj(), 3)
    # ... dephase there (Schur product with the pair mask) ...
    T *= jmask.reshape((d, d, d, d) + (1,) * (2 * n - 4))
    # ... and back to the computational basis.
    T = _apply_axis(T, V1, 0)
    T = _apply_axis(T, V1, 1)
    T = _apply_axis(T, V1.conj(), 2)
    T = _apply_axis(T, V1.conj(), 3)
    return np.moveaxis(T, (0, 1, 2, 3), src)


def apply_step(rho_t: np.ndarray, step: dict, d: int, n: int,
               adjoint: bool = False) -> np.ndarray:
    """One gate step on a rank-2n rho tensor: dephasing then angle
    fluctuation (forward, the order of Eq. gate-channel: A o D), angle
    fluctuation then dephasing (adjoint).  Masks are real, so each factor
    is self-adjoint; conj() is applied anyway for generality."""
    def _deph():
        for q in range(rho_t.ndim // 2):
            m = step['deph'][q]
            _schur_qudit(rho_t, m.conj() if adjoint else m, q, n, d)

    def _jit():
        nonlocal rho_t
        if step['V1'] is None:
            return
        V1 = step['V1']
        qa, qb = step['qa'], step['qb']
        jm = step['jmask']
        rho_t = _jitter_pair_front(rho_t, V1,
                                   jm.conj() if adjoint else jm,
                                   qa, qb, n, d)

    if adjoint:
        _jit()
        _deph()
    else:
        _deph()
        _jit()
    return rho_t


def apply_dose(rho: np.ndarray, steps: List[dict], d: int, n: int,
               adjoint: bool = False) -> np.ndarray:
    """Apply one full noise round N (all steps, in order) -- or its
    Hilbert-Schmidt adjoint N^dagger (reversed order, adjoint=True) -- to a
    (d^n, d^n) matrix.

    Args:
      rho: (d^n, d^n) complex matrix.  Works on any matrix, not just density
        matrices: the off-diagonal codeword blocks |psi_i><psi_j| of the
        entanglement-fidelity computation are evolved with the same call.
      steps: output of ``build_dose_steps``.
      d, n: local dimension and register size.
      adjoint: apply N^dagger instead of N (used by the Petz recovery).

    Returns:
      (d^n, d^n) complex matrix N(rho) (or N^dagger(rho)); the input is not
      modified.
    """
    dim = d ** n
    assert rho.shape == (dim, dim), f"expected ({dim},{dim}), got {rho.shape}"
    rho_t = rho.reshape((d,) * (2 * n)).astype(complex, copy=True)
    seq = reversed(steps) if adjoint else steps
    for step in seq:
        rho_t = apply_step(rho_t, step, d, n, adjoint=adjoint)
    return rho_t.reshape(dim, dim)


def evolve_code_blocks(code_states: np.ndarray, steps: List[dict],
                       d: int, n: int) -> dict:
    """N(|psi_i><psi_j|) for all i <= j.

    Args:
      code_states: (K, d^n) orthonormal codewords.
      steps: output of ``build_dose_steps``.

    Returns:
      dict ``{(i, j): (d^n, d^n) array}`` for 0 <= i <= j < K.  Blocks for
      i > j follow from Hermiticity preservation, N(A^dagger) = N(A)^dagger,
      and are not stored.  These blocks are all the figure-of-merit routines
      in ``projective_decode`` need (the entanglement fidelity is a sum of
      matrix elements <psi_i| R(N(|psi_i><psi_j|)) |psi_j>).
    """
    K = code_states.shape[0]
    out = {}
    for i in range(K):
        for j in range(i, K):
            blk = np.outer(code_states[i], code_states[j].conj())
            out[(i, j)] = apply_dose(blk, steps, d, n)
    return out
