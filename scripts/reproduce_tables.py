#!/usr/bin/env python3
"""Shor-Laflamme weight enumerators from committed parameters (paper Table II path).

`enumerators(code_states, n, d)` is the wire-grouped enumerator used by
scripts/table3_enumerators.py and scripts/print_paper_tables.py for the
paper's Table II: it applies each weight-w product of the single-qudit
channel error basis (qudit_hardware_error_basis) locally, never forming the
d^n x d^n projector, so n = 9 rows take seconds.

For the paper's Tables I-VI run scripts/print_paper_tables.py.
"""Reproduce Table III (Shor-Laflamme weight enumerators) and Table IV
(analytic per-code encoder resources) from committed trained parameters.

Table III enumerators are computed from the codeword projector over the
*hardware* single-qudit error basis (qudit_hardware_error_basis) -- this is
the basis the paper uses (it reproduces the [[5,1,3]] benchmark row exactly;
the generalized-Pauli basis instead gives A_j=0 for the pure code).

Usage:  PYTHONPATH=. python3 scripts/reproduce_tables.py
No retraining: reads results/params (via src.catalog) only.
"""
import itertools
import numpy as np
from src.errors import qudit_hardware_error_basis
from src.decoders._common import apply_local_np


def _enumerators_grouped(code_states, n, d, weight, ops):
    """Wire-grouped Shor-Laflamme A_w, B_w via LOCAL error application only
    (no d^n x d^n projector), so this scales to n=9 in well under a second --
    the same method the paper uses. For each weight-w wire subset and each
    tensor product of single-qudit `ops`, the error is applied to each codeword
    on its <=w wires and the K x K matrix M_kl = <psi_k|E|psi_l> is formed:
        A_w = (1/K^2) sum_E |Tr M|^2 ,   B_w = (1/K) sum_E sum_kl |M_kl|^2 .
    """
    K = code_states.shape[0]
    A = B = 0.0
    for sub in itertools.combinations(range(n), weight):
        left = [i for i in range(n) if i not in sub]
        inv = tuple((list(sub) + left).index(i) for i in range(n))
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


def enumerators(code_states, n, d):
    ops = [np.asarray(m) for m in qudit_hardware_error_basis(d)]
    A = np.ones(3)
    B = np.ones(3)  # A_0 = B_0 = 1 (identity)
    for w in (1, 2):
        A[w], B[w] = _enumerators_grouped(code_states, n, d, w, ops)
    return A, B
