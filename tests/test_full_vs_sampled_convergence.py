"""Lock in: a sampled-subset loss below threshold must
never certify convergence — certification is full-batch only.

Construction: code states |000>, |111>, |222> (d=3, n=3, unclosed dist-3
model). Every off-diagonal detectability term vanishes (a weight-<=2 operator
cannot map |iii> to |jjj>), but ops with non-constant diagonal action (all
four single-qudit hardware ops, and diagonal-correlated pairs) give a nonzero
variance term, so the full KL loss is bounded away from zero. Ops like
X01 (x) X12 contribute exactly zero (diagonal product is identically 0).

A Hoeffding-uniform subsample that happens to draw only zero-contribution ops
reports a sampled loss of exactly 0 — the situation an uncertified seed race accepts
as "TARGET REACHED" (full loss
76.3, sampled 0.0 on 11% of keys). Any convergence/victory check must instead
certify on the full batch.
"""
import numpy as np
import pytest

TOL_CONVERGED = 1e-6
N, D, DIST = 3, 3, 3


@pytest.fixture(scope="module")
def setup():
    import jax.numpy as jnp
    from src.errors import ErrorModel
    from src.jax_backend import create_jax_loss_vmap_weighted

    model = ErrorModel(d=D, n_qudit=N, distance=DIST, closed=False)
    grouped = model.build_grouped(verbose=False)
    loss_fn = create_jax_loss_vmap_weighted(grouped, D, D, N, DIST)
    group_sizes = [g['matrices'].shape[0] for g in grouped]

    dim = D ** N
    cs = np.zeros((D, dim), complex)
    for k in range(D):
        cs[k, k * (D ** 2) + k * D + k] = 1.0  # |kkk>
    code_states = jnp.asarray(cs)
    return loss_fn, group_sizes, code_states


def test_sampled_zero_does_not_certify_convergence(setup):
    import jax.numpy as jnp
    from src.jax_backend import hoeffding_weights
    loss_fn, group_sizes, code_states = setup

    ones = tuple(jnp.ones((n,)) for n in group_sizes)
    full = float(loss_fn(code_states, ones))
    assert full > 1e-2, "fixture broken: full loss should be far from zero"

    # Find a seeded Hoeffding draw that misses every violated op —
    # deterministic given the seed scan order. Only 6 of the 60 ops have zero
    # contribution for this fixture, so use the extreme 1-op subsample
    # (P ~ 10% per seed); larger draws make fooling rarer but not impossible,
    # which is exactly why thresholding a subsampled loss cannot certify.
    fooled_seed = None
    for seed in range(500):
        rng = np.random.default_rng(seed)
        w = hoeffding_weights(group_sizes, 1, rng)
        sampled = float(loss_fn(code_states, w))
        if sampled < TOL_CONVERGED:
            fooled_seed = seed
            break
    assert fooled_seed is not None, \
        "expected some subsample to miss all violated ops"

    # The unsound rule ('sampled < tol' => converged) accepts this theta;
    # full-batch certification MUST reject it.
    assert not (full < TOL_CONVERGED), \
        "full-batch certification must reject a sampled-only zero"


def test_certification_accepts_actual_code():
    """Sanity: certification is not vacuously rejecting — the analytic
    [[5,1,3]]_Z3 code passes the full-batch KL check on the hardware model."""
    import jax.numpy as jnp
    from src.catalog import five_qudit_code_states
    from src.errors import ErrorModel
    from src.jax_backend import create_jax_loss_vmap_weighted

    model = ErrorModel(d=3, n_qudit=5, distance=3, closed=False)
    grouped = model.build_grouped(verbose=False)
    loss_fn = create_jax_loss_vmap_weighted(grouped, 3, 3, 5, 3)
    sizes = [g['matrices'].shape[0] for g in grouped]
    cs = jnp.asarray(five_qudit_code_states(3))
    full = float(loss_fn(cs, tuple(jnp.ones((n,)) for n in sizes)))
    assert full < 1e-10
