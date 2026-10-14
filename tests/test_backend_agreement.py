"""JAX vs PennyLane backend agreement on a small (n=3, d=2) instance.

Loss values and gradients agree to 1e-10. Optimizer trajectories agree to a
documented 1e-3 over 3 steps: PennyLane's AdamOptimizer and optax.adam use
different epsilon conventions (inside vs outside the bias-corrected sqrt), so
bit-identical trajectories are impossible by design; the meaningful invariant
is loss/grad agreement, asserted tightly below.
"""
import numpy as np
import pytest

N, D = 3, 2
DIST = 2
SEED = 0


@pytest.fixture(scope="module")
def setup():
    import jax.numpy as jnp
    from src.encoder import create_native_encoder
    from src.jax_backend import create_jax_encoder, build_varqec_loss
    from src.errors import ErrorModel

    enc_pl, _, ppl = create_native_encoder(N, D, force_manual=True)
    enc_jx, _, ppl_jx = create_jax_encoder(N, D)
    assert ppl == ppl_jx
    model = ErrorModel(d=D, n_qudit=N, distance=DIST)
    E_det_dense, _ = model.build_dense()
    loss_jx = build_varqec_loss(enc_jx, D, D, N, DIST,
                                model.build_grouped(verbose=False))
    rng = np.random.default_rng(SEED)
    theta0 = rng.uniform(0, 2 * np.pi, (2, ppl))
    return enc_pl, enc_jx, E_det_dense, loss_jx, theta0


def _loss_pl(theta, enc_pl, E_det_dense):
    from src.loss import kl_loss_fast
    return kl_loss_fast(theta, enc_pl, E_det_dense, [], D, DIST)


def test_encoder_states_agree(setup):
    import jax
    import jax.numpy as jnp
    enc_pl, enc_jx, _, _, theta0 = setup
    s_pl = np.stack([np.array(enc_pl(theta0, k)) for k in range(D)])
    s_jx = np.array(jax.vmap(enc_jx, in_axes=(None, 0))(
        jnp.array(theta0), jnp.arange(D)))
    assert np.max(np.abs(s_pl - s_jx)) < 1e-12


def test_loss_agreement_1e10(setup):
    import jax.numpy as jnp
    enc_pl, _, E_det_dense, loss_jx, theta0 = setup
    lp = float(_loss_pl(theta0, enc_pl, E_det_dense))
    lj = float(loss_jx(jnp.array(theta0)))
    assert abs(lp - lj) < 1e-10


def test_gradient_agreement(setup):
    import jax
    import jax.numpy as jnp
    from pennylane import numpy as pnp
    enc_pl, _, E_det_dense, loss_jx, theta0 = setup
    g_jx = np.array(jax.grad(loss_jx)(jnp.array(theta0)))

    t = pnp.array(theta0, requires_grad=True)
    import pennylane as qml
    g_pl = np.array(qml.grad(
        lambda th: _loss_pl(th, enc_pl, E_det_dense))(t))
    assert np.max(np.abs(g_pl - g_jx)) < 1e-10


def test_optimizer_trajectories_3_steps(setup):
    import jax
    import jax.numpy as jnp
    import optax
    from pennylane import numpy as pnp
    from pennylane.optimize import AdamOptimizer

    enc_pl, _, E_det_dense, loss_jx, theta0 = setup

    theta = pnp.array(theta0, requires_grad=True)
    opt = AdamOptimizer(0.05)
    traj_pl = []
    for _ in range(3):
        theta, _ = opt.step_and_cost(
            lambda th: _loss_pl(th, enc_pl, E_det_dense), theta)
        traj_pl.append(np.array(theta))

    tj = jnp.array(theta0)
    o = optax.adam(0.05)
    st = o.init(tj)
    g = jax.grad(loss_jx)
    traj_jx = []
    for _ in range(3):
        up, st = o.update(g(tj), st, tj)
        tj = optax.apply_updates(tj, up)
        traj_jx.append(np.array(tj))

    for i in range(3):
        # eps-convention divergence between the two Adam implementations
        # grows ~1e-5 -> ~2.5e-4 over 3 steps (measured); 1e-3 bound.
        assert np.max(np.abs(traj_pl[i] - traj_jx[i])) < 1e-3, f"step {i+1}"
