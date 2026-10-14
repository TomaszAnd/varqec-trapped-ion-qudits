"""Pin the batched simulate_ler_with_correction_factored against an inline
copy of the original per-shot reference implementation.

The batched rewrite (PERF_AUDIT 2026-07-13) must preserve, for a fixed seed:
  - the RNG stream (draws happen shot-by-shot in the original order), and
  - the decoded outcome of every shot (same correction argmax, same
    projection), up to floating-point association differences of order 1e-15.
"""
import numpy as np
import pytest

from src.errors import qudit_hardware_error_basis, make_hardware_noise_fn
from src.simulation import (simulate_ler_with_correction_factored,
                            _apply_factored_op)


def _reference_impl(code_states, noise_fn, single_errors, n_qudit, d,
                    n_shots, seed):
    """Verbatim pre-batching implementation (src/simulation.py before
    2026-07-13), returning per-shot corrected fidelities as well."""
    K = code_states.shape[0]
    rng = np.random.default_rng(seed)
    fids_corr = []

    for _ in range(n_shots):
        alpha = rng.standard_normal(K) + 1j * rng.standard_normal(K)
        alpha /= np.linalg.norm(alpha)
        logical_state = alpha @ code_states
        noisy = noise_fn(logical_state, rng)

        def _code_overlap(state):
            return sum(np.abs(np.vdot(code_states[k], state)) ** 2
                       for k in range(K))

        best_overlap = _code_overlap(noisy)
        best_corrected = noisy
        for q in range(n_qudit):
            for E in single_errors:
                corrected = _apply_factored_op(noisy, E.conj().T, q,
                                               n_qudit, d)
                overlap = _code_overlap(corrected)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_corrected = corrected

        coeffs = np.array([np.vdot(code_states[k], best_corrected)
                           for k in range(K)])
        projected = coeffs @ code_states
        norm = np.linalg.norm(projected)
        if norm > 1e-10:
            projected /= norm
        fids_corr.append(float(np.abs(np.vdot(logical_state, projected)) ** 2))
    return np.array(fids_corr)


def _random_code(K, dim, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim, K)) + 1j * rng.standard_normal((dim, K))
    Q, _ = np.linalg.qr(A)
    return Q[:, :K].T.copy()


@pytest.mark.parametrize("n_qudit,d,K,p,seed", [
    (3, 3, 3, 0.10, 0),
    (4, 3, 3, 0.05, 1),
    (3, 4, 4, 0.08, 2),
])
def test_batched_matches_reference_per_shot(n_qudit, d, K, p, seed):
    dim = d ** n_qudit
    code_states = _random_code(K, dim, seed)
    single = [np.array(E) for E in qudit_hardware_error_basis(d)]
    n_shots = 300

    ref_fids = _reference_impl(
        code_states, make_hardware_noise_fn(d, n_qudit, p), single,
        n_qudit, d, n_shots, seed=seed + 100)
    out = simulate_ler_with_correction_factored(
        code_states, make_hardware_noise_fn(d, n_qudit, p), single,
        n_qudit, d, n_shots=n_shots, seed=seed + 100, batch=128)

    # Same RNG stream + same decode decisions -> per-shot fidelities agree
    # to FP association error; identical error counts.
    assert np.mean(ref_fids < 0.5) == out['logical_error_rate']
    assert abs(float(np.mean(ref_fids)) - out['mean_fidelity']) < 1e-12


def test_batch_size_invariance():
    n_qudit, d, K = 3, 3, 3
    code_states = _random_code(K, d ** n_qudit, seed=5)
    single = [np.array(E) for E in qudit_hardware_error_basis(d)]
    outs = [simulate_ler_with_correction_factored(
        code_states, make_hardware_noise_fn(d, n_qudit, 0.1), single,
        n_qudit, d, n_shots=257, seed=9, batch=b)
        for b in (1, 64, 257, 1024)]
    for o in outs[1:]:
        assert o['logical_error_rate'] == outs[0]['logical_error_rate']
        assert abs(o['mean_fidelity'] - outs[0]['mean_fidelity']) < 1e-13
        assert abs(o['mean_raw_fidelity'] - outs[0]['mean_raw_fidelity']) < 1e-13
