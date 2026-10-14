"""Bit-level agreement gate for the cross-group batched weighted loss


`create_jax_loss_vmap_weighted_v2` batches all same-arity wire groups into
one einsum GEMM. It must be numerically indistinguishable from the original
per-group vmap path: on (n=3, 5, 7) x (full basis, dephasing basis) x 3
random thetas the composed params-space losses must agree to 1e-12 and
every gradient component to 1e-10 — with ones weights (full batch) and
with a seeded stratified draw. This is the adoption contract for making
fast=True the default in build_varqec_loss_weighted.
"""
import numpy as np
import pytest

CONFIGS = [(3, "full"), (3, None), (5, "full"), (5, None),
           (7, "full"), (7, None)]
D = 3
DISTANCE = 3
LAYERS = 2
N_THETAS = 3
LOSS_TOL = 1e-12
GRAD_TOL = 1e-10


@pytest.fixture(scope="module", params=CONFIGS,
                ids=[f"n{n}-{b or 'dephasing'}" for n, b in CONFIGS])
def pair(request):
    """(val_grad_v1, val_grad_v2, group_sizes, ppl, layers) for one config."""
    import jax
    from src.errors import ErrorModel
    from src.jax_backend import create_jax_encoder, build_varqec_loss_weighted

    n, basis = request.param
    connections = [[i, j] for i in range(n) for j in range(i + 1, n)]
    enc, _, ppl = create_jax_encoder(n, D, connections=connections,
                                     use_scan=False)
    model = ErrorModel(d=D, n_qudit=n, distance=DISTANCE, closed=True,
                       basis=basis)
    grouped = model.build_grouped(verbose=False)
    group_sizes = [g['matrices'].shape[0] for g in grouped]
    kw = dict(encoder_fn=enc, K=D, d=D, n_qudit=n, distance=DISTANCE,
              E_det_grouped=grouped)
    vg1 = jax.jit(jax.value_and_grad(
        build_varqec_loss_weighted(fast=False, **kw)))
    vg2 = jax.jit(jax.value_and_grad(
        build_varqec_loss_weighted(fast=True, **kw)))
    return vg1, vg2, group_sizes, ppl, n


def _assert_agree(vg1, vg2, theta, weights, tag):
    l1, g1 = vg1(theta, weights)
    l2, g2 = vg2(theta, weights)
    dl = abs(float(l1) - float(l2))
    dg = float(np.max(np.abs(np.asarray(g1) - np.asarray(g2))))
    assert dl < LOSS_TOL, f"{tag}: |loss_v2 - loss_v1| = {dl:.3e}"
    assert dg < GRAD_TOL, f"{tag}: max grad diff = {dg:.3e}"


def test_fastpath_matches_reference(pair):
    import jax
    import jax.numpy as jnp
    from src.jax_backend import stratified_weights

    vg1, vg2, group_sizes, ppl, n = pair
    ones = tuple(jnp.ones((m,), dtype=jnp.float64) for m in group_sizes)
    rng = np.random.default_rng(42)
    for t in range(N_THETAS):
        key = jax.random.PRNGKey(1000 * n + t)
        theta = jax.random.uniform(key, (LAYERS, ppl),
                                   minval=0.0, maxval=2 * np.pi)
        _assert_agree(vg1, vg2, theta, ones, f"n={n} theta{t} ones")
        w = stratified_weights(group_sizes, 0.1, rng)
        _assert_agree(vg1, vg2, theta, w, f"n={n} theta{t} stratified")
