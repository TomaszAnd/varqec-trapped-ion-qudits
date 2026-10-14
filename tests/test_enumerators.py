"""Shor-Laflamme weight-enumerator regressions.

Two invariants:
  1. [[5,1,3]]_Z3 is a perfect distance-3 code: over the FULL generalized-Pauli
     basis, A_j = B_j for j <= 2 AND A_1 = A_2 = 0 (pure code).
  2. The wire-grouped enumerator path matches the dense-projector path.
"""
import itertools
import os

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _generalized_paulis(d):
    X = np.roll(np.eye(d), 1, axis=0).astype(complex)
    Z = np.diag(np.exp(2j * np.pi * np.arange(d) / d))
    return [np.linalg.matrix_power(X, a) @ np.linalg.matrix_power(Z, b)
            for a in range(d) for b in range(d) if (a, b) != (0, 0)]


def _enumerators_grouped(code_states, n, d, weight, ops):
    """Wire-grouped A_w, B_w: A_w = (1/K^2) sum_E |Tr(E P)|^2 with
    Tr(E P) = sum_k <psi_k|E|psi_k>, B_w = (1/K) sum_E sum_jk |<psi_j|E|psi_k>|^2.
    Local applications only — no d^n x d^n projector."""
    from src.decoders._common import apply_local_np
    K = code_states.shape[0]
    A = 0.0
    B = 0.0
    for sub in itertools.combinations(range(n), weight):
        left = [i for i in range(n) if i not in sub]
        order = list(sub) + left
        inv = tuple(order.index(i) for i in range(n))
        for choice in itertools.product(ops, repeat=weight):
            E = choice[0]
            for e in choice[1:]:
                E = np.kron(E, e)
            E_psi = np.stack([apply_local_np(code_states[k], E, sub, inv,
                                             d=d, n_qudit=n) for k in range(K)])
            M = np.conj(code_states) @ E_psi.T
            A += abs(np.trace(M)) ** 2
            B += float(np.sum(np.abs(M) ** 2))
    return A / K ** 2, B / K


class TestPerfectCode513:
    def test_A_equals_B_and_pure_up_to_weight2(self):
        from src.catalog import five_qudit_code_states
        from src.analysis import compute_weight_enumerators
        cs = five_qudit_code_states(3)
        A, B = compute_weight_enumerators(cs, 5, 3, _generalized_paulis(3),
                                          max_weight=2)
        assert np.allclose(A - B, 0.0, atol=1e-10)
        assert A[1] < 1e-12 and A[2] < 1e-12  # pure code

    def test_grouped_path_matches_dense_path(self):
        from src.catalog import five_qudit_code_states
        from src.analysis import compute_weight_enumerators
        cs = five_qudit_code_states(3)
        paulis = _generalized_paulis(3)
        A_dense, B_dense = compute_weight_enumerators(cs, 5, 3, paulis,
                                                      max_weight=1)
        A1, B1 = _enumerators_grouped(cs, 5, 3, 1, paulis)
        assert abs(A1 - A_dense[1]) < 1e-10
        assert abs(B1 - B_dense[1]) < 1e-10
