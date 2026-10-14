#!/usr/bin/env python3
"""Weight-enumerator (Table II) dist2-row reconciliation: training loss
3.5e-30 ("detection certifies exactly") vs B1-A1 = 0.0316, the largest
weight-1 deficit in the table.

Hypotheses tested:
 (H-prompt) the enumerator's single-qudit basis includes same-qudit closure
   products the dist-2 training never constrained — REFUTED by code reading:
   reproduce_tables.enumerators uses the bare 4-op hardware basis only.
 (H-diag) the dist-2 objective omits the diagonal codeword-independence
   condition: create_jax_loss_vmap_weighted_v2 applies Term 2 (K/4 * Var_k
   <psi_k|E|psi_k>) only when distance >= 3. Training with distance=2 zeroed
   only the off-diagonals; B1-A1 = sum_E [offdiag + diag-spread] should then
   be 100% diagonal spread.

Computes, for the dist-2 code states:
  1. B1-A1 on the bare hardware basis (must reproduce the committed 0.0316),
     split per op into off-diagonal vs diagonal-variance parts;
  2. B1-A1 on the closure-augmented hardware basis (bare + close_error_basis);
  3. the same split on the full 10-op training basis (off-diag part must be
     ~1e-30, matching the trained loss);
  4. the recomputed dist-2 training loss (off-diag only) and the hypothetical
     dist>=3-style loss (off-diag + diagonal term) on the same E_det.

Writes table3_dist2_reconcile.json.
x64 footgun: import src.jax_backend before any jnp array creation.
"""
import os, sys, json
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import src.jax_backend  # x64 first
import numpy as np
from src.errors import (ErrorModel, qudit_hardware_error_basis,
                        qudit_full_hardware_error_basis, close_error_basis)
from src.decoders._common import encoder_forward

D, N, K = 3, 5, 3
HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = 'results/campaigns/fig6_panels/d3_n5_dist2_16L.npz'

conns = [[i, j] for i in range(N) for j in range(i + 1, N)]
cs, *_ = encoder_forward(NPZ, conns)
cs = np.asarray(cs)  # (K, 3^5)


def overlap_matrix(E, q):
    """M_kl = <psi_k| E_q |psi_l> for single-qudit op E on wire q."""
    t = cs.reshape((K,) + (D,) * N)
    t = np.moveaxis(np.tensordot(t, E, axes=([1 + q], [1])), -1, 1 + q)
    Epsi = t.reshape(K, -1)
    return np.conj(cs) @ Epsi.T


def b1_a1_split(ops):
    """(B1-A1, offdiag part, diag-spread part) summed over ops x wires / K."""
    tot = off = dia = 0.0
    per_op = []
    for oi, E in enumerate(ops):
        o_op = d_op = 0.0
        for q in range(N):
            M = overlap_matrix(np.asarray(E, complex), q)
            c = np.trace(M) / K
            o = float(np.sum(np.abs(M) ** 2) - np.sum(np.abs(np.diag(M)) ** 2))
            dd = float(np.sum(np.abs(np.diag(M) - c) ** 2))
            o_op += o; d_op += dd
        per_op.append({'op': oi, 'offdiag': o_op / K, 'diag_spread': d_op / K})
        off += o_op; dia += d_op
    return (off + dia) / K, off / K, dia / K, per_op


bare = qudit_hardware_error_basis(D)
full = qudit_full_hardware_error_basis(D)
closure = close_error_basis(bare)

out = {'code': NPZ, 'committed_B1_minus_A1': 0.031601941144243195}

t, o, dd, per = b1_a1_split(bare)
out['bare_hw_basis'] = {'n_ops': len(bare), 'B1_minus_A1': t,
                        'offdiag_part': o, 'diag_spread_part': dd,
                        'per_op': per}
print(f"bare hw basis (4 ops):    B1-A1 = {t:.6f}  "
      f"(offdiag {o:.3e}, diag-spread {dd:.6f})", flush=True)

t2, o2, d2, per2 = b1_a1_split(list(bare) + list(closure))
out['closure_augmented_hw'] = {'n_ops': len(bare) + len(closure),
                               'B1_minus_A1': t2, 'offdiag_part': o2,
                               'diag_spread_part': d2, 'per_op': per2}
print(f"closure-augmented ({len(bare)+len(closure)} ops): B1-A1 = {t2:.6f}  "
      f"(offdiag {o2:.3e}, diag-spread {d2:.6f})", flush=True)

t3, o3, d3, per3 = b1_a1_split(full)
out['full_training_basis'] = {'n_ops': len(full), 'B1_minus_A1': t3,
                              'offdiag_part': o3, 'diag_spread_part': d3,
                              'per_op': per3}
print(f"full 10-op train basis:   B1-A1 = {t3:.6f}  "
      f"(offdiag {o3:.3e} <- trained to ~0, diag-spread {d3:.6f})", flush=True)

# 4. recomputed training loss (dist=2 core: off-diag only) + hypothetical
#    dist>=3-style loss on the same E_det
import jax.numpy as jnp
from src.jax_backend import (create_jax_loss_vmap_weighted_v2)
grouped = ErrorModel(d=D, n_qudit=N, distance=2, closed=False,
                     basis="full").build_grouped(verbose=False)
ones = tuple(jnp.ones((g['matrices'].shape[0],), dtype=jnp.float64)
             for g in grouped)
loss2 = create_jax_loss_vmap_weighted_v2(grouped, K, D, N, distance=2)
loss3 = create_jax_loss_vmap_weighted_v2(grouped, K, D, N, distance=3)
l2 = float(loss2(jnp.asarray(cs), ones))
l3 = float(loss3(jnp.asarray(cs), ones))
out['trained_loss_offdiag_only'] = l2
out['hypothetical_loss_with_diag_term'] = l3
print(f"dist=2 loss (as trained, off-diag only) = {l2:.3e}", flush=True)
print(f"same E_det with the dist>=3 diagonal term = {l3:.3e}", flush=True)

json.dump(out, open(os.path.join(HERE, 'table3_dist2_reconcile.json'), 'w'),
          indent=2)
print('DONE', flush=True)
