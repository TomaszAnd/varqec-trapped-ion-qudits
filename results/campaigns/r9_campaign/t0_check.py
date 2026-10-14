#!/usr/bin/env python3
"""Bit-identity + timing check of the front-axes jitter fast path
against the reference _rotate_pair/_schur_pair implementation."""
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

import numpy as np

import src.jax_backend  # noqa: F401
from src.decoders.exact_channel import (build_dose_steps, apply_dose,
                                        _rotate_pair, _schur_pair)

ETA, SA2 = float(np.exp(-0.073)), 0.047


def reference_dose(rho, steps, d, n, adjoint=False):
    """The earlier apply_dose path, inlined (reference for bit-identity)."""
    dim = d ** n
    rho_t = rho.reshape((d,) * (2 * n)).astype(complex, copy=True)
    seq = reversed(steps) if adjoint else steps

    for step in seq:
        def _deph():
            for q in range(n):
                m = step['deph'][q]
                shape = [1] * (2 * n)
                shape[q] = d
                shape[n + q] = d
                mm = m.conj() if adjoint else m
                rho_t2[0] *= mm.reshape(shape)

        def _jit():
            if step['V1'] is None:
                return
            V1 = step['V1']
            qa, qb = step['qa'], step['qb']
            rho_t2[0] = _rotate_pair(rho_t2[0], V1.conj().T, qa, qb, n)
            jm = step['jmask']
            _schur_pair(rho_t2[0], jm.conj() if adjoint else jm,
                        qa, qb, n, d)
            rho_t2[0] = _rotate_pair(rho_t2[0], V1, qa, qb, n)

        rho_t2 = [rho_t]
        if adjoint:
            _jit()
            _deph()
        else:
            _deph()
            _jit()
        rho_t = rho_t2[0]
    return rho_t.reshape(dim, dim)


def check(d, n, label):
    gp = [(i, j, 0, 1) for i in range(n) for j in range(i + 1, n)]
    steps = build_dose_steps(n, d, gp, ETA, sigma_a2=SA2)
    rng = np.random.default_rng(7)
    dim = d ** n
    A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    rho = A @ A.conj().T
    rho /= np.trace(rho).real
    for adj in (False, True):
        t0 = time.time()
        ref = reference_dose(rho, steps, d, n, adjoint=adj)
        t_ref = time.time() - t0
        t0 = time.time()
        new = apply_dose(rho, steps, d, n, adjoint=adj)
        t_new = time.time() - t0
        ident = np.array_equal(ref, new)
        print(f"{label} adjoint={adj}: bit-identical={ident} "
              f"ref={t_ref:.2f}s new={t_new:.2f}s "
              f"speedup={t_ref / t_new:.2f}x", flush=True)
        assert ident, f"NOT bit-identical at {label} adjoint={adj}"


check(3, 5, "n=5 v2 full dose")
check(3, 7, "n=7 v2 full dose")
print("ALL BIT-IDENTICAL")
