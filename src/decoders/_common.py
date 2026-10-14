"""Shared primitives for the decoder package (``src/decoders``).

Three small routines that every decoder in this package, and several of the
benchmark scripts, rely on:

* ``apply_local_np``       apply a wire-grouped operator (paper App. D, the
                           wire-grouped representation of the channel basis)
                           to a numpy statevector; the numpy port of
                           ``src.jax_backend._apply_local_matrix``;
* ``flat_group_offsets``   flat-index bookkeeping for the grouped error basis
                           (the per-operator ordering that channel priors,
                           ``results/campaigns/n9_reference/meth_pauli_weights.npz``
                           and the correction sets all share);
* ``encoder_forward``      rebuild the trained encoder from a ``.npz``
                           checkpoint and return the codewords as numpy.

Consumers: ``src.decoders.{weighted_map,petz,projective_decode}``,
``scripts/benchmark_ler_meth_pauli.py``, ``scripts/benchmark_ler_meth_kraus.py``,
``scripts/fig6_panels.py``, ``scripts/headline_lsweep.py``,
``scripts/table3_enumerators.py``, and the
campaign scripts under ``results/campaigns``.
"""
from __future__ import annotations
import os
import sys

import numpy as np

# Repository root (three levels up: src/decoders/_common.py -> repo).  Added
# to sys.path so the package also imports when a script is run from
# elsewhere than the repo root.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def apply_local_np(state: np.ndarray, matrix: np.ndarray,
                   wires: tuple, inverse_perm: tuple,
                   d: int, n_qudit: int) -> np.ndarray:
    """Apply a (d^w, d^w) matrix to `wires` of an n-qudit statevector.

    Args:
      state: (d^n,) complex statevector, qudit 0 the most significant index
        (the ``reshape((d,)*n)`` axis convention of ``src.jax_backend``).
      matrix: (d^w, d^w) operator acting on the ``w = len(wires)`` qudits
        listed in `wires`, row/column index ordered like ``kron`` over
        `wires` in the given order.
      wires: tuple of the w qudit indices the operator acts on.
      inverse_perm: length-n axis permutation that restores the original
        qudit order after ``tensordot`` has moved the acted-on axes to the
        front (precomputed by ``ErrorModel.build_grouped`` per wire group).
      d, n_qudit: local dimension and register size.

    Returns:
      (d^n,) statevector ``(matrix on wires (x) identity elsewhere) @ state``.

    Identity sentinel (wires=(), matrix=[[1.0]]) is a scalar multiply.
    """
    if len(wires) == 0:
        return matrix[0, 0] * state
    shape = (d,) * n_qudit
    s = state.reshape(shape)
    w = len(wires)
    M = matrix.reshape((d,) * (2 * w))
    axes = list(range(w, 2 * w))
    # Contract the w "column" axes of M with the acted-on qudit axes of the
    # state; the result has the w output axes in front, inverse_perm puts
    # them back in register order.
    contracted = np.tensordot(M, s, axes=(axes, list(wires)))
    s = np.transpose(contracted, inverse_perm)
    return s.reshape(-1)


def flat_group_offsets(E_grouped: list) -> np.ndarray:
    """Cumulative offsets per group: offsets[g] = sum of sizes of groups 0..g-1.

    Args:
      E_grouped: list of wire-group dicts from ``ErrorModel.build_grouped``
        (each with a ``'matrices'`` array of shape (n_in_group, d^w, d^w)).

    Returns:
      (len(E_grouped)+1,) int array; ``offsets[g] + e`` is the flat index of
      op e in group g.  This flat ordering is the one shared by every
      per-operator weight vector in the repository (channel priors,
      ``meth_pauli_weights.npz['weights']``, correction sets).
    """
    sizes = [int(g['matrices'].shape[0]) for g in E_grouped]
    return np.concatenate([[0], np.cumsum(sizes)])


def encoder_forward(checkpoint_npz: str, connections: list):
    """Load a .npz checkpoint, rebuild the JAX encoder, return code_states (K, d^n)
    as numpy plus (d, n_qudit, K).

    Args:
      checkpoint_npz: path to a trained-parameter file with fields ``params``,
        ``d``, ``n_qudit``, ``K`` and optionally ``entangler``.
      connections: list of [i, j] edges for the entangling layer (the
        all-to-all graph for every campaign code of the paper).

    Returns:
      ``(code_states, d, n_qudit, K)`` with ``code_states`` a (K, d^n)
      complex numpy array whose rows are the orthonormal codewords
      |psi_0>, ..., |psi_{K-1}>.

    Dispatches on the checkpoint's `entangler` field (default "ms" for older
    files without it): variant-entangler codes (zz / ls / csum / cex, the
    entangling-gate study of paper Sec. VIII B) are rebuilt with
    create_jax_encoder_variant. Unknown values raise instead of silently
    evaluating a different circuit."""
    import jax
    import jax.numpy as jnp
    from src.jax_backend import (create_jax_encoder,
                                 create_jax_encoder_variant,
                                 VARIANT_ENTANGLERS)

    data = np.load(checkpoint_npz, allow_pickle=True)
    params = jnp.asarray(data['params'])
    d = int(data['d'])
    n_qudit = int(data['n_qudit'])
    K = int(data['K'])
    entangler = str(data['entangler']) if 'entangler' in data.files else 'ms'

    if entangler == 'ms':
        enc, _, _ = create_jax_encoder(n_qudit, d, connections=connections,
                                       use_scan=True)
    elif entangler in VARIANT_ENTANGLERS:
        enc, _, _ = create_jax_encoder_variant(
            n_qudit, d, connections=connections, entangler=entangler,
            use_scan=True)
    else:
        raise ValueError(
            f"unknown entangler '{entangler}' in {checkpoint_npz}")
    enc_vmapped = jax.jit(jax.vmap(enc, in_axes=(None, 0)))
    code_states = np.asarray(enc_vmapped(params, jnp.arange(K)))
    return code_states, d, n_qudit, K
