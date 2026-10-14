"""Round-13 Commit R13-2 — scripts-only w1/w2 loss split via masking.

The KL detection loss `build_varqec_loss` evaluates against the full
E_det_grouped (closure-under-products distance-3 set). This module
splits that loss into two pieces:

  loss_w1 — KL terms on the weight-1 invariance set (the literal
            n × n_single weight-1 single-qudit Paulis embedded on each
            qudit, plus the identity sentinel which contributes 0).
            This is the "weight-1 only" Path C target from Round-12.

  loss_w2 — KL terms on the weight-≤2 closure complement: same-qudit
            closure cross-products (Pauli products X·Z on the same
            qudit that appear in `close_error_basis`) + all weight-2
            two-qudit tensor products. These are the operators added
            by `closed=True, distance=3` beyond the weight-1 set.

The Round-13 structural recipe trains
  total_loss(theta) = loss_w1(theta) + loss_w2(theta, weights_w2),
where weights_w2 can sample the weight-2 orthogonality terms at any
fraction. Setting weights_w2 = ones recovers the full weight-≤2 loss
(= the campaign loss to numerical precision, since the identity
sentinel contributes 0 for orthonormal codes).

Construction of the masks relies on the order in which
`src.errors.build_native_error_set_factored` emits operators:
  - For weight w in [1, max_det = distance-1]:
      for each qudit subset, for each error tuple, append the factor.
  - If closed=True: at the END, for each qudit, for each cross-product,
    append `[(q, E_cross)]`.
Then `factored_to_grouped` groups by wire signature, preserving the
ORDER within each group (modulo dedup, which is order-preserving).

So in the (q,) wire group, the first `n_single` entries are the
weight-1 originals (from the w=1 loop) and the remaining entries are
the same-qudit cross-products (from the closed=True block). The (q1,q2)
groups are all weight-2 two-qudit products; the () group is the
identity sentinel.

This decomposition is verified at runtime by the
`assert_split_matches_full` helper, which checks the structural
identity loss_w1(theta) + loss_w2(theta, ones) == build_varqec_loss(theta)
to numerical precision on a randomly initialized theta.

src/ is untouched; this module is scripts-only and uses public API
(`build_varqec_loss`, `build_varqec_loss_weighted`).
"""
import os
import sys
from typing import Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_w1_w2_masks(E_det_grouped, d, dtype=None):
    """Return (w1_mask, w2_mask) tuples that select weight-1 invariance
    and weight-2 orthogonality terms respectively.

    Each mask is a tuple of jnp arrays, one per wire group, shape (n_g,),
    suitable to pass as the `weights_per_group` argument to
    `src.jax_backend.build_varqec_loss_weighted`.

    Args:
        E_det_grouped: output of ErrorModel(distance=3, closed=True).build_grouped().
        d: qudit dimension (used to determine n_single).
        dtype: jnp dtype for the masks. Defaults to jnp.float64.

    Returns:
        (w1_mask, w2_mask) — both tuples aligned with E_det_grouped.
        For each group g of size n_g:
          - wires = (): w1=zeros, w2=zeros (identity sentinel; contributes 0
            to the loss for orthonormal codes regardless of weight).
          - wires = (q,): w1[:n_single]=1, rest=0; w2 is the inverse —
            w1 selects the weight-1 single-qudit Paulis, w2 selects
            same-qudit closure cross-products.
          - len(wires) >= 2: w1=zeros, w2=ones (all weight-2 ops).
    """
    import jax.numpy as jnp
    from src.errors import qudit_hardware_error_basis

    if dtype is None:
        dtype = jnp.float64

    n_single = len(qudit_hardware_error_basis(d))

    w1_list = []
    w2_list = []
    for g in E_det_grouped:
        wires = g['wires']
        n_g = int(g['matrices'].shape[0])
        m1 = np.zeros(n_g, dtype=float)
        m2 = np.zeros(n_g, dtype=float)
        if len(wires) == 0:
            pass  # identity sentinel — both masks zero
        elif len(wires) == 1:
            # Single-qudit group: originals first (n_single of them),
            # closure cross-products after.
            n_orig = min(n_single, n_g)
            m1[:n_orig] = 1.0
            m2[n_orig:] = 1.0
        else:
            # Weight-2 (or higher) group: all orthogonality
            m2[:] = 1.0
        w1_list.append(jnp.asarray(m1, dtype=dtype))
        w2_list.append(jnp.asarray(m2, dtype=dtype))
    return tuple(w1_list), tuple(w2_list)


def ones_weights(E_det_grouped, dtype=None):
    """Tuple of all-ones weights aligned with E_det_grouped."""
    import jax.numpy as jnp
    if dtype is None:
        dtype = jnp.float64
    return tuple(jnp.ones(int(g['matrices'].shape[0]), dtype=dtype)
                 for g in E_det_grouped)


def build_w1w2_split_loss(enc, K, d, n_qudit, dist, E_det_grouped):
    """Return (loss_w1_fn, loss_w2_fn).

    loss_w1_fn(theta) -> scalar
        KL loss on the weight-1 invariance set only. Uses the w1_mask.

    loss_w2_fn(theta, weights_w2) -> scalar
        KL loss on the weight-2 orthogonality set with user-supplied
        sampling weights `weights_w2`. The user weights are
        element-wise multiplied by the w2_mask so the weight-1 terms
        cannot accidentally be included (even if the caller passes
        an all-ones vector). To recover the full weight-2 contribution
        pass `ones_weights(E_det_grouped)` — the w2_mask will zero out
        the w1 entries and pass the rest through.

    Both functions are JIT-compiled.
    """
    import jax
    from jax import jit
    from src.jax_backend import build_varqec_loss_weighted

    w1_mask, w2_mask = build_w1_w2_masks(E_det_grouped, d)

    loss_weighted = build_varqec_loss_weighted(
        enc, K, d, n_qudit, dist, E_det_grouped)

    @jit
    def loss_w1_fn(theta):
        return loss_weighted(theta, w1_mask)

    @jit
    def loss_w2_fn(theta, weights_w2):
        # Element-wise multiply with the w2 mask so the w1 entries are
        # zeroed even if the caller passes ones across all groups.
        import jax.numpy as jnp
        combined = tuple(w2_mask[i] * weights_w2[i]
                         for i in range(len(w2_mask)))
        return loss_weighted(theta, combined)

    return loss_w1_fn, loss_w2_fn


def assert_split_matches_full(enc, K, d, n_qudit, dist, E_det_grouped,
                              theta, atol=1e-9, rtol=1e-9, verbose=True):
    """Correctness check: loss_w1(theta) + loss_w2(theta, ones)
    == build_varqec_loss(theta) to numerical precision.

    Raises AssertionError on mismatch; prints the per-piece values
    when verbose=True.
    """
    from src.jax_backend import build_varqec_loss

    loss_full_fn = build_varqec_loss(
        enc, K, d, n_qudit, dist, E_det_grouped)
    loss_w1_fn, loss_w2_fn = build_w1w2_split_loss(
        enc, K, d, n_qudit, dist, E_det_grouped)
    ones = ones_weights(E_det_grouped)

    L_full = float(loss_full_fn(theta))
    L_w1 = float(loss_w1_fn(theta))
    L_w2 = float(loss_w2_fn(theta, ones))
    L_split = L_w1 + L_w2

    if verbose:
        print(f"  loss_full              = {L_full:.10e}")
        print(f"  loss_w1                = {L_w1:.10e}")
        print(f"  loss_w2(ones)          = {L_w2:.10e}")
        print(f"  loss_w1 + loss_w2(ones) = {L_split:.10e}")
        print(f"  abs diff               = {abs(L_split - L_full):.2e}")
        print(f"  rel diff               = {abs(L_split - L_full)/max(abs(L_full), 1e-30):.2e}")

    if not np.isclose(L_split, L_full, atol=atol, rtol=rtol):
        raise AssertionError(
            f"split loss does not match full loss: "
            f"{L_split} vs {L_full} (diff {L_split - L_full})")
    return {"loss_full": L_full, "loss_w1": L_w1, "loss_w2": L_w2,
            "split": L_split}


def _self_test():
    """Tiny standalone correctness check, runnable as
    `python3 scripts/loss_split_helpers.py`. Trains nothing — just
    builds a random theta on ((5,1,3))_3 and verifies the split."""
    import jax
    import jax.numpy as jnp
    from src.errors import ErrorModel
    from src.jax_backend import create_jax_encoder

    d, n_qudit = 3, 5
    K = d
    dist = 3
    n_layers = 4

    print(f"=== loss_split_helpers self-test on (({n_qudit},{K},{dist}))_{d}, "
          f"{n_layers}L ===")
    model = ErrorModel(d=d, n_qudit=n_qudit, distance=dist, closed=True)
    E_det_grouped = model.build_grouped(verbose=False)
    n_total = sum(int(g['matrices'].shape[0]) for g in E_det_grouped)
    n_w1_ops = sum(1 for g in E_det_grouped if len(g['wires']) <= 1)
    print(f"  |E_det| = {n_total} across {len(E_det_grouped)} groups; "
          f"{n_w1_ops} groups with wires len ≤ 1")

    enc, _, ppl = create_jax_encoder(n_qudit, d, use_scan=True)
    key = jax.random.PRNGKey(0)
    theta = jax.random.uniform(key, (n_layers, ppl), minval=0.0,
                               maxval=2 * np.pi)

    result = assert_split_matches_full(
        enc, K, d, n_qudit, dist, E_det_grouped, theta,
        atol=1e-9, rtol=1e-9, verbose=True)
    print("\n  ✓ split matches full to numerical precision")
    return result


if __name__ == "__main__":
    _self_test()
