"""
Tests for JAX backend: encoder, vectorized loss, and training.
Run: python3 -m pytest tests/test_jax_backend.py -v
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import value_and_grad

from src.jax_backend import create_jax_encoder, create_jax_loss, train_jax
from src.errors import build_native_error_set


class TestJAXEncoderMatchesPennyLane:
    def test_d3_n3_same_states(self):
        """JAX and PennyLane encoders produce identical states for d=3, n=3."""
        from pennylane import numpy as pnp
        from src.encoder import create_native_encoder

        enc_pl, _, ppl = create_native_encoder(3, 3, force_manual=True)
        enc_jax, _, _ = create_jax_encoder(3, 3)

        np.random.seed(42)
        params_np = np.random.uniform(0, 2 * np.pi, (1, ppl))
        params_pl = pnp.array(params_np)
        params_jax = jnp.array(params_np)

        for k in range(3):
            s_pl = np.array(enc_pl(params_pl, k))
            s_jax = np.array(enc_jax(params_jax, k))
            overlap = abs(np.vdot(s_pl, s_jax))
            assert overlap > 1 - 1e-10, f"code_ind={k}: overlap={overlap}"

    def test_d4_n3_same_states(self):
        """JAX and PennyLane encoders match for d=4, n=3."""
        from pennylane import numpy as pnp
        from src.encoder import create_native_encoder

        enc_pl, _, ppl = create_native_encoder(3, 4)
        enc_jax, _, _ = create_jax_encoder(3, 4)

        np.random.seed(42)
        params_np = np.random.uniform(0, 2 * np.pi, (1, ppl))
        params_pl = pnp.array(params_np)
        params_jax = jnp.array(params_np)

        for k in range(4):
            s_pl = np.array(enc_pl(params_pl, k))
            s_jax = np.array(enc_jax(params_jax, k))
            overlap = abs(np.vdot(s_pl, s_jax))
            assert overlap > 1 - 1e-10, f"code_ind={k}: overlap={overlap}"


class TestJAXVectorizedLoss:
    def test_matches_pennylane_loss(self):
        """JAX vectorized loss matches PennyLane detection loss."""
        from pennylane import numpy as pnp
        from src.encoder import create_native_encoder
        from src.loss import kl_loss_detection_minibatch

        n, d, dist, K = 3, 3, 2, 3
        E_det, _ = build_native_error_set(n, d, dist)

        enc_pl, _, ppl = create_native_encoder(n, d, force_manual=True)
        enc_jax, _, _ = create_jax_encoder(n, d)

        np.random.seed(42)
        params_np = np.random.uniform(0, 2 * np.pi, (1, ppl))

        # PennyLane loss (full batch)
        rng = np.random.default_rng(99)
        loss_pl = float(kl_loss_detection_minibatch(
            pnp.array(params_np), enc_pl, E_det, K, dist,
            batch_fraction=1.0, rng=rng))

        # JAX loss
        loss_fn = create_jax_loss(enc_jax, E_det, K, dist)
        loss_jax = float(loss_fn(jnp.array(params_np)))

        assert abs(loss_pl - loss_jax) < 1e-8, f"PL={loss_pl}, JAX={loss_jax}"

    def test_gradient_nonzero(self):
        """JAX gradient is non-zero for random parameters."""
        n, d, dist, K = 3, 3, 2, 3
        E_det, _ = build_native_error_set(n, d, dist)
        enc_jax, _, ppl = create_jax_encoder(n, d)
        loss_fn = create_jax_loss(enc_jax, E_det, K, dist)

        key = jax.random.PRNGKey(42)
        params = jax.random.uniform(key, (1, ppl), minval=0, maxval=2 * np.pi)

        grad = jax.grad(loss_fn)(params)
        assert np.any(np.abs(np.array(grad)) > 1e-10)

    def test_d3_dist3_loss(self):
        """Vectorized loss works for distance-3 (includes diagonal variance)."""
        n, d, dist, K = 3, 3, 3, 3
        E_det, _ = build_native_error_set(n, d, dist, closed=True)
        enc_jax, _, ppl = create_jax_encoder(n, d)
        loss_fn = create_jax_loss(enc_jax, E_det, K, dist)

        key = jax.random.PRNGKey(42)
        params = jax.random.uniform(key, (1, ppl), minval=0, maxval=2 * np.pi)
        loss_val = float(loss_fn(params))
        assert np.isfinite(loss_val)
        assert loss_val >= 0


class TestJAXTraining:
    def test_loss_decreases(self):
        """A few JAX training steps decrease the loss."""
        n, d, dist, K = 3, 3, 2, 3
        E_det, _ = build_native_error_set(n, d, dist)
        enc_jax, _, ppl = create_jax_encoder(n, d)
        loss_fn = create_jax_loss(enc_jax, E_det, K, dist)

        best_params, losses = train_jax(loss_fn, 1, ppl, n_steps=20, seed=42)
        assert losses[-1] < losses[0], "Loss should decrease"

    def test_returns_valid_shape(self):
        """train_jax returns params of correct shape."""
        n, d, dist, K = 3, 3, 2, 3
        E_det, _ = build_native_error_set(n, d, dist)
        enc_jax, _, ppl = create_jax_encoder(n, d)
        loss_fn = create_jax_loss(enc_jax, E_det, K, dist)

        best_params, losses = train_jax(loss_fn, 1, ppl, n_steps=5, seed=42)
        assert best_params.shape == (1, ppl)
        assert len(losses) == 5
        assert all(np.isfinite(l) for l in losses)


class TestJAXScanEncoder:
    def test_matches_unrolled(self):
        """Scan encoder produces same states as the unrolled encoder."""
        from src.jax_backend import create_jax_encoder_scan

        n, d = 3, 3
        enc_unrolled, _, ppl = create_jax_encoder(n, d)
        enc_scan, _, ppl2 = create_jax_encoder_scan(n, d)
        assert ppl == ppl2

        key = jax.random.PRNGKey(42)
        params = jax.random.uniform(key, (2, ppl), minval=0, maxval=2 * np.pi)

        for k in range(3):
            s_unrolled = np.array(enc_unrolled(params, k))
            s_scan = np.array(enc_scan(params, k))
            overlap = abs(np.vdot(s_unrolled, s_scan))
            assert overlap > 1 - 1e-10, f"code_ind={k}: overlap={overlap}"

    def test_gradient_works(self):
        """Scan encoder gradient is non-zero."""
        from src.jax_backend import create_jax_encoder_scan

        n, d, dist, K = 3, 3, 2, 3
        E_det, _ = build_native_error_set(n, d, dist)
        enc_scan, _, ppl = create_jax_encoder_scan(n, d)
        loss_fn = create_jax_loss(enc_scan, E_det, K, dist)

        key = jax.random.PRNGKey(42)
        params = jax.random.uniform(key, (2, ppl), minval=0, maxval=2 * np.pi)
        grad = jax.grad(loss_fn)(params)
        assert np.any(np.abs(np.array(grad)) > 1e-10)

    def test_loss_matches_unrolled(self):
        """Scan-based loss matches unrolled loss value."""
        from src.jax_backend import create_jax_encoder_scan

        n, d, dist, K = 3, 3, 2, 3
        E_det, _ = build_native_error_set(n, d, dist)

        enc_unrolled, _, ppl = create_jax_encoder(n, d)
        enc_scan, _, _ = create_jax_encoder_scan(n, d)

        loss_unrolled_fn = create_jax_loss(enc_unrolled, E_det, K, dist)
        loss_scan_fn = create_jax_loss(enc_scan, E_det, K, dist)

        key = jax.random.PRNGKey(42)
        params = jax.random.uniform(key, (2, ppl), minval=0, maxval=2 * np.pi)

        l_unrolled = float(loss_unrolled_fn(params))
        l_scan = float(loss_scan_fn(params))
        assert abs(l_unrolled - l_scan) < 1e-8, f"Unrolled={l_unrolled}, Scan={l_scan}"




class TestJAXFactoredLoss:
    def test_matches_dense(self):
        """Factored loss matches dense loss for same code states."""
        from src.jax_backend import create_jax_loss_factored
        from src.errors import build_native_error_set_factored

        n, d, dist, K = 3, 3, 3, 3
        E_det_dense, _ = build_native_error_set(n, d, dist, closed=True)
        E_det_factors, _ = build_native_error_set_factored(n, d, dist, closed=True)

        enc_jax, _, ppl = create_jax_encoder(n, d)

        key = jax.random.PRNGKey(42)
        params = jax.random.uniform(key, (1, ppl), minval=0, maxval=2 * np.pi)

        loss_dense_fn = create_jax_loss(enc_jax, E_det_dense, K, dist)
        loss_factored_fn = create_jax_loss_factored(enc_jax, E_det_factors, n, d, K, dist)

        loss_dense = float(loss_dense_fn(params))
        loss_factored = float(loss_factored_fn(params))

        assert abs(loss_dense - loss_factored) < 1e-8, \
            f"Dense={loss_dense}, Factored={loss_factored}"

