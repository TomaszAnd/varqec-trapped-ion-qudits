#!/usr/bin/env python3
"""Low-rank Petz transpose recovery against a weighted-Pauli channel.

The Petz (transpose) recovery (paper App. F, Eq. (F2); Petz 1986; Barnum
and Knill 2002) of the code and a channel N is the near-optimal reference
recovery whose entanglement infidelity is at most twice that of the optimal
recovery map.  Paper Sec. VII B uses it to show that the loss--LER
decoupling is not the lookup decoder's fault.  For code projector
P_C = sum_k |psi_k><psi_k| and a weighted-Pauli channel
N(rho) = sum_i w_i E_i rho E_i^dagger (Kraus K_i = sqrt(w_i) E_i over the
weight-<=2 closure basis, paper Sec. IV C):

    Petz Kraus   R_i = P_C K_i^dagger N(P_C)^{-1/2}
    recovery     R(omega) = sum_i R_i omega R_i^dagger
    fidelity     F = sum_i |<psi_L| R_i |noisy>|^2  (pure-state input + noise outcome)

which is Eq. (F2), R(X) = P N^dagger(N(P)^{-1/2} X N(P)^{-1/2}) P, written
in Kraus form.  N(P_C)^{-1/2} is the Moore-Penrose inverse square root on
supp(N(P_C)) (invert the singular values above a relative threshold, zero
the rest -- NOT Tikhonov).

Low-rank construction (never materialize the dense d^n x d^n N(P_C)):
    M in C^{d^n x (n_kept*K)},  M[:, i*K+k] = sqrt(w_i) E_i psi_k.
    Then N(P_C) = M M^dagger exactly.  SVD M = U diag(s) V^dagger gives
    N(P_C) = U diag(s^2) U^dagger,  N(P_C)^{-1/2} = U diag(1/s * [s>tol]) U^dagger.
The same per-(op, codeword) tensor K_psi[i,k] = sqrt(w_i) E_i psi_k serves
both the M build and the per-shot fidelity.  The SVD itself is done through
the Gram matrix M^dagger M (see ``build_petz_recovery``).

Channel model versus evaluation channel.  The composed calibrated Kraus
channel has no tractable Kraus list at n = 9, so the Petz map for the
calibrated channel is built against its Pauli-twirled version (the
685-operator weights the channel-weighted codes were trained against,
``priors.meth_pauli_prior``) and then EVALUATED by Monte Carlo on whichever
``noise_fn`` is passed -- the twirled Pauli channel or the physical Kraus
channel.  The Petz operators are the same in both evaluations; only the
sampled noise differs ("recovery designed against the channel model,
tested on the physical channel").  Build once per code, evaluate on both.

This module is the Monte-Carlo (shot-fraction LER) Petz; the exact Petz
on the full calibrated noise round used for every calibrated-channel
number in the paper is ``projective_decode.build_petz_exact_dose`` /
``entanglement_fidelity_petz``.

Consumers: ``src/decoders/__init__.py`` (re-exports), ``tests/test_petz.py``,
and ``results/campaigns/r1_campaign/{t3_petz,t3b_decoder_matrix}.py``.
"""
from __future__ import annotations
from typing import Callable, Optional

import numpy as np

from src.decoders._common import apply_local_np


def build_petz_recovery(code_states: np.ndarray,
                        E_full_grouped: list,
                        weights: np.ndarray,
                        d: int, n_qudit: int,
                        rcond: float = 1e-10,
                        weight_floor: float = 1e-15) -> dict:
    """Build the low-rank Petz factors against the weighted-Pauli channel
    N(rho) = sum_i weights[i] E_i rho E_i^dagger on the weight-<=2 closure basis.

    Args:
      code_states: (K, d^n) orthonormal codewords.
      E_full_grouped: ErrorModel(...).build_grouped() (flat order matches `weights`).
      weights: (n_total,) per-op channel weights (the twirled calibrated
        prior, the depolarizing prior, or a sparse single-op channel for tests).
      d, n_qudit: local dimension and register size.
      rcond: RELATIVE singular-value cutoff for the Moore-Penrose
        inverse-sqrt -- keep sigma > rcond*sigma_max, drop (regularize to
        zero) the rest.  Relative (not absolute) because the Gram-matrix
        build squares the conditioning: an absolute sigma-threshold would
        sit in the eigh-noise floor of M^dagger M and keep spurious
        near-null modes.  rcond*sigma_max is the standard pinv
        regularization and matches np.linalg.pinv semantics.
      weight_floor: drop ops with weight <= this (zero-weight Kraus
        contribute nothing to N(P_C) or to the fidelity).

    Returns dict with:
      U: (d^n, rank) left singular vectors of M for kept modes (s > tol).
      Uh: (rank, d^n) precomputed U^dagger.
      s_inv: (rank,) 1/s on the kept modes -- for N(P_C)^{-1/2}.
      s: (rank,) kept singular values of M (= sqrt of N(P_C) eigenvalues).
      K_psi: (n_kept, K, d^n) tensor sqrt(w_i) E_i psi_k (reused for evaluation).
      rank: # singular values > tol (= # columns of U).
      n_kept: # ops with weight > weight_floor.

    Build uses the Gram-matrix economy SVD: instead of np.linalg.svd on the
    tall (d^n x r) M (which swaps and takes ~90 min at d^n = 19683 because
    the LAPACK gesdd workspace + the 647 MB U exceed RAM), form G = M^dagger M
    (r x r, small), eigendecompose it, and recover U = M V / s on the kept
    modes.  Everything except M itself stays small -> ~1-2 min, no swap.
    """
    K, dim = code_states.shape
    K_psi_list = []
    flat = 0
    for g in E_full_grouped:
        wires = tuple(int(w) for w in g['wires'])
        inv = tuple(int(p) for p in g['inverse_perm'])
        mats = np.asarray(g['matrices'])
        for e in range(mats.shape[0]):
            w = float(weights[flat])
            flat += 1
            if w <= weight_floor:
                continue
            sw = np.sqrt(w)
            cols = np.stack([
                sw * apply_local_np(code_states[k], mats[e], wires, inv,
                                    d, n_qudit)
                for k in range(K)])  # (K, d^n)
            K_psi_list.append(cols)

    if not K_psi_list:
        raise ValueError("no Kraus ops above weight_floor")
    K_psi = np.stack(K_psi_list)               # (n_kept, K, d^n)
    M = K_psi.reshape(-1, dim).T               # (d^n, r), r = n_kept*K

    # Gram-matrix economy SVD: G = M^dagger M = V diag(s^2) V^dagger.
    G = M.conj().T @ M                          # (r, r) Hermitian PSD
    evals, V = np.linalg.eigh(G)                # ascending
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    V = V[:, order]
    s_all = np.sqrt(np.clip(evals, 0.0, None))
    keep = s_all > rcond * s_all.max()
    s = s_all[keep]
    # U_kept = M V_kept / s_kept  (candidate left singular vectors).
    U = (M @ V[:, keep]) / s[np.newaxis, :]
    # Genuine-mode filter. Forming G = M^dagger M squares the conditioning,
    # so eigh produces spurious near-null eigenvalues whose sqrt can exceed
    # rcond*sigma_max and overlap the genuine small-sigma range (no rcond
    # cleanly separates them). A genuine left singular vector satisfies
    # M V_i = s_i U_i with ||U_i|| = 1; a spurious mode has M V_i ~ 0, so
    # ||U_i|| = ||M V_i||/s_i ~ 0. Keep unit-norm columns. (Spurious modes
    # are orthogonal to range(M) = supp(N(P_C)), so they never affected the
    # LER -- verified identical to full-SVD -- but this keeps U a clean
    # orthonormal basis of the support.)
    col_norms = np.linalg.norm(U, axis=0)
    genuine = col_norms > 0.5
    U = np.ascontiguousarray(U[:, genuine])
    s = s[genuine]
    s_inv = 1.0 / s
    Uh = np.ascontiguousarray(U.conj().T)   # precompute once (reused per batch)
    return {'U': U, 'Uh': Uh, 's_inv': s_inv, 's': s, 'K_psi': K_psi,
            'rank': int(genuine.sum()), 'n_kept': len(K_psi_list)}


def n_pc_inv_sqrt_apply(petz: dict, state: np.ndarray) -> np.ndarray:
    """Apply N(P_C)^{-1/2} = U diag(1/s) U^dagger to a statevector (or
    (dim, B) batch) via the low-rank factors. U holds only the kept modes,
    so s_inv has no zeros.  Uses the precomputed Uh = U^dagger to avoid a
    per-call conjugate copy."""
    U, Uh, s_inv = petz['U'], petz['Uh'], petz['s_inv']
    y = Uh @ state                       # (rank,) or (rank, B)
    if y.ndim == 1:
        return U @ (s_inv * y)
    return U @ (s_inv[:, None] * y)


def support_projector_apply(petz: dict, state: np.ndarray) -> np.ndarray:
    """Apply the support projector Pi_supp(N(P_C)) = U U^dagger to a
    statevector (U is kept-only, s > tol).  Used by the tests to measure
    how much of a noisy state lies outside the recoverable support."""
    U = petz['U']
    return U @ (U.conj().T @ state)


def fidelity_under_petz(petz: dict, code_states: np.ndarray,
                        alpha: np.ndarray, noisy_state: np.ndarray) -> float:
    """Post-recovery fidelity with the original logical state psi_L = alpha @ code_states.

    Args:
      petz: output of ``build_petz_recovery``.
      code_states: (K, d^n) codewords; alpha: (K,) logical amplitudes;
      noisy_state: (d^n,) one noise trajectory of psi_L.

    Returns:
      F = sum_i |<psi_L| R_i |noisy>|^2 = sum_i |< (sqrt(w_i) E_i psi_L) | v >|^2,
      where v = N(P_C)^{-1/2} |noisy> and sqrt(w_i) E_i psi_L = alpha @ K_psi[i].
      (The P_C in R_i is absorbed by the overlap with psi_L in the code space.)

    Computed as C[i,k] = <K_psi[i,k]|v> = conj(K_psi[i,k] . v*), then
    inner[i] = sum_k conj(alpha_k) C[i,k].  The single np.dot reads K_psi
    once (no transpose, no large temporaries) -- ~20x faster than
    tensordot-ing alpha into the (n_kept, K, d^n) tensor each shot.
    """
    v = n_pc_inv_sqrt_apply(petz, noisy_state)
    C = np.dot(petz['K_psi'], v.conj()).conj()   # (n_kept, K)
    inner = C @ alpha.conj()                      # (n_kept,)
    return float(np.sum(np.abs(inner) ** 2))


def simulate_ler_with_petz(code_states: np.ndarray, petz: dict,
                           noise_fn: Callable, n_qudit: int, d: int,
                           n_shots: int, seed: int,
                           fidelity_threshold: float = 0.5,
                           batch: int = 200) -> np.ndarray:
    """Per-shot bernoulli outcomes (1 = logical error) under Petz recovery
    (the Sec. V A shot protocol with the Petz map in place of the decoder).

    Args:
      code_states: (K, d^n) orthonormal codewords.
      petz: output of ``build_petz_recovery``.
      noise_fn: ``noise_fn(state, rng) -> noisy_state``.  May sample from a
        DIFFERENT channel than the one `petz` was built against (twirl-built
        Petz evaluated on the physical Kraus noise, see module docstring).
      n_qudit, d: register size and local dimension.
      n_shots, seed: Monte-Carlo shot count and ``default_rng`` seed.
      fidelity_threshold: a shot counts as a logical error when the
        post-recovery fidelity falls below it (0.5, paper Sec. V A).
      batch: shots whose Petz algebra is done in one pass.

    Returns:
      (n_shots,) int8 outcome array.

    Batched: the noisy states still come from per-shot `noise_fn` calls, but
    the Petz linear algebra (U/Uh and K_psi contractions) is applied to
    `batch` shots at once so the ~1.2 GB of Petz factors (n = 9) are read
    once per batch, not once per shot -- ~100x fewer memory passes than the
    naive per-shot loop.
    """
    K, dim = code_states.shape
    rng = np.random.default_rng(seed)
    K_psi = petz['K_psi']
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
        V = n_pc_inv_sqrt_apply(petz, NOISY)             # (dim, b)
        # C[i,k,j] = <K_psi[i,k] | V[:,j]> = conj(K_psi[i,k] . V[:,j]*)
        C = np.tensordot(K_psi, V.conj(), axes=([2], [0])).conj()  # (n_kept, K, b)
        inner = np.einsum('ikj,jk->ij', C, alphas.conj())          # (n_kept, b)
        F = np.sum(np.abs(inner) ** 2, axis=0)                     # (b,)
        outcomes[start:start + b] = (F < fidelity_threshold).astype(np.int8)
    return outcomes
