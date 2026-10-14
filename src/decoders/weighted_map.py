#!/usr/bin/env python3
"""Prior-weighted MAP decoder over weight-<=2 corrections.

This is the decoder referred to in the footnote of paper Sec. V A: the
lookup decoder of the i.i.d. benchmark (``src.simulation.lookup_table_decoder``)
treats every candidate correction as equally likely and only tries the
identity and the single-qudit inverses; the decoder here

  (a) extends the correction set to {I, weight-1, weight-2} taken from the
      same weight-<=2 closure basis the channel-weighted codes were trained
      against (``ErrorModel(..., closed=True).build_grouped()``), and
  (b) multiplies each candidate's score by its prior probability under the
      channel,

so that it picks  argmax_C  w_C * sum_k |<C psi_k | noisy>|^2, i.e. a
maximum-a-posteriori choice rather than the lookup table's uniform-prior
maximum-likelihood one.  The paper's finding is that this leaves the code
ranking unchanged and improves only the stabilizer benchmark.

Why weight-2 corrections do not help on single-error channels: a single
weight-<=2 error on a distance-3 code is already undone by projection alone
(Knill--Laflamme: P_C E |psi_L> is proportional to |psi_L>), which the lookup
decoder realizes through its identity-correction-plus-projection fallback.
The extension matters only on multi-error channels whose total weight
exceeds the distance.

The LER is the Monte-Carlo shot fraction of Sec. V A (Haar-random logical
state, fidelity below 0.5 counts as a failure).  This is a shot-by-shot
decision on the pure trajectory state, so unlike the recovery maps in
``projective_decode`` it is not linear in the density matrix and its LER
depends on how a non-Pauli channel is unraveled into trajectories; it is
therefore used for the Monte-Carlo (trajectory) benchmarks only, never for
the exact calibrated-channel evaluation of Sec. V B.

Consumers: ``src.decoders.projective_decode`` (correction set only),
``tests/test_weighted_map.py``, ``tests/test_projective_decode.py``, and
campaign scripts under ``results/campaigns``.
"""
from __future__ import annotations
from typing import Callable, List, Tuple

import numpy as np

from src.decoders._common import apply_local_np


def build_weighted_correction_set(E_full_grouped: list,
                                   prior_weights: np.ndarray,
                                   n_qudit: int, d: int,
                                   max_weight: int = 2,
                                   drop_zero: bool = True) -> List[Tuple]:
    """Return a list of (wires, C_dag_matrix, inverse_perm, prior_w) tuples.

    Args:
      E_full_grouped: grouped closure basis from ``ErrorModel.build_grouped``
        (list of dicts with ``'wires'``, ``'matrices'``, ``'inverse_perm'``).
      prior_weights: flat (n_total,) per-operator prior in the flat order of
        ``_common.flat_group_offsets`` (e.g. ``priors.meth_pauli_prior()`` or
        the flat output of ``src.meth_weights.compute_meth_group_weights``).
      n_qudit, d: register size and local dimension (unused here, kept for
        call-site symmetry with the simulators).
      max_weight: keep only operators acting on at most this many qudits.
      drop_zero: skip operators with prior <= 0 (they could never win the
        argmax).

    Returns:
      List of ``(wires, C_dag, inverse_perm, prior_w)`` in the flat basis
      order, where ``C_dag = E^dagger`` is the correction that undoes error
      ``E`` on `wires` and ``prior_w = prior_weights[flat index of E]``.
      At n = 5, q = 3 this is the 221-operator set of paper App. F; at
      n = 9, q = 3 it has 685 operators.
    """
    corrections = []
    flat = 0
    for g in E_full_grouped:
        wires = tuple(int(w) for w in g['wires'])
        inv = tuple(int(p) for p in g['inverse_perm'])
        mats = np.asarray(g['matrices'])
        for e in range(mats.shape[0]):
            w = float(prior_weights[flat])
            flat += 1
            if len(wires) > max_weight:
                continue
            if drop_zero and w <= 0.0:
                continue
            corrections.append((wires, mats[e].conj().T, inv, w))
    return corrections


def simulate_ler_with_weighted_map(code_states: np.ndarray,
                                   noise_fn: Callable,
                                   correction_set: List[Tuple],
                                   n_qudit: int, d: int,
                                   n_shots: int, seed: int,
                                   batch: int = 200) -> np.ndarray:
    """Per-shot bernoulli outcomes (1 = logical error) under weighted MAP.

    Args:
      code_states: (K, d^n) orthonormal codewords.
      noise_fn: ``noise_fn(state, rng) -> noisy_state`` sampling one
        trajectory of the channel on a (d^n,) statevector.
      correction_set: output of ``build_weighted_correction_set``.
      n_qudit, d: register size and local dimension.
      n_shots, seed: Monte-Carlo shot count and ``default_rng`` seed (the
        same seed reproduces the same logical states and noise draws).
      batch: shots scored per matmul (memory/speed trade-off only).

    Returns:
      (n_shots,) int8 array; entry 1 when the post-correction, code-space
      projected state has fidelity < 0.5 with the injected logical state.

    The MAP score for correction C_i (= E_i, applied as E_i^dagger to the
    noisy state) is  w_i * ||P_C E_i^dagger noisy||^2
                    = w_i * sum_k |<E_i psi_k | noisy>|^2.
    Precomputing CES[i,k] = E_i psi_k once and batching the <CES|noisy>
    contraction makes the per-shot cost one matmul over CES (read once per
    batch) instead of |corrections| separate tensordots -- roughly 16x
    faster, the same trick as the Petz K_psi evaluation.  Only the single
    best correction is then applied + projected per shot.  Algebraically
    identical to the per-correction argmax loop.
    """
    K, dim = code_states.shape
    # Precompute CES[i,k] = E_i psi_k  (E_i = C_dag^dagger) and the weight vector.
    CES = np.stack([
        np.stack([apply_local_np(code_states[k], C_dag.conj().T, wires, inv,
                                 d, n_qudit) for k in range(K)])
        for (wires, C_dag, inv, w) in correction_set])      # (n_corr, K, dim)
    wvec = np.array([w for (_, _, _, w) in correction_set])  # (n_corr,)
    corr_meta = [(wires, C_dag, inv) for (wires, C_dag, inv, _) in correction_set]

    rng = np.random.default_rng(seed)
    outcomes = np.zeros(n_shots, dtype=np.int8)
    for start in range(0, n_shots, batch):
        b = min(batch, n_shots - start)
        alphas = np.empty((b, K), dtype=complex)
        NOISY = np.empty((dim, b), dtype=complex)
        for j in range(b):
            # Haar-random logical state: normalized complex Gaussian amplitudes.
            a = rng.standard_normal(K) + 1j * rng.standard_normal(K)
            a /= np.linalg.norm(a)
            alphas[j] = a
            NOISY[:, j] = noise_fn(a @ code_states, rng)
        # inner[i,k,j] = <CES[i,k] | NOISY[:,j]> = conj(CES . NOISY*)
        inner = np.tensordot(CES, NOISY.conj(), axes=([2], [0])).conj()  # (n_corr,K,b)
        score = wvec[:, None] * np.sum(np.abs(inner) ** 2, axis=1)        # (n_corr, b)
        best = np.argmax(score, axis=0)                                   # (b,)
        for j in range(b):
            wires, C_dag, inv = corr_meta[int(best[j])]
            corrected = apply_local_np(NOISY[:, j], C_dag, wires, inv, d, n_qudit)
            # Project back onto the code space and renormalize (Sec. V A protocol).
            coeffs = code_states.conj() @ corrected            # (K,)
            projected = coeffs @ code_states
            nrm = np.linalg.norm(projected)
            if nrm > 1e-10:
                projected = projected / nrm
            logical = alphas[j] @ code_states
            fid = float(np.abs(np.vdot(logical, projected)) ** 2)
            if fid < 0.5:
                outcomes[start + j] = 1
    return outcomes
