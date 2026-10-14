"""Tests for the vmap JAX backend path.

Covers _apply_local_matrix (step 2), create_jax_loss_vmap (step 3),
and the jax_vmap training-backend option (step 4).
Kept separate from test_jax_backend.py so the scan and vmap paths
can be exercised or retired independently.
"""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# ── step 2: _apply_local_matrix ───────────────────────────────────────

def _kron_embed(M_small, wires, n_qudit, d):
    """Embed a (d**w x d**w) operator on `wires` into a full d**n x d**n matrix."""
    dim = d ** n_qudit
    shape = (d,) * n_qudit
    Id = np.eye(d, dtype=complex)
    if len(wires) == 0:
        return np.eye(dim, dtype=complex) * M_small[0, 0]
    w = len(wires)
    full = np.zeros((dim, dim), dtype=complex)
    M_tensor = M_small.reshape((d,) * (2 * w))
    new_order = list(wires) + [i for i in range(n_qudit) if i not in wires]
    inverse_perm = tuple(new_order.index(i) for i in range(n_qudit))
    for col in range(dim):
        v = np.zeros(dim, dtype=complex); v[col] = 1.0
        v_t = v.reshape(shape)
        contracted = np.tensordot(
            M_tensor, v_t, axes=(list(range(w, 2 * w)), list(wires)))
        v_t = np.transpose(contracted, inverse_perm)
        full[:, col] = v_t.reshape(dim)
    return full


def _inverse_perm(wires, n_qudit):
    new_order = list(wires) + [i for i in range(n_qudit) if i not in wires]
    return tuple(new_order.index(i) for i in range(n_qudit))


class TestApplyLocalMatrix:
    def _run(self, d, n_qudit, wires, M_small, seed=0):
        from src.jax_backend import _apply_local_matrix
        rng = np.random.default_rng(seed)
        dim = d ** n_qudit
        state = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        state /= np.linalg.norm(state)

        dense = _kron_embed(M_small, wires, n_qudit, d)
        expected = dense @ state

        inv = _inverse_perm(wires, n_qudit)
        got = np.asarray(_apply_local_matrix(
            jnp.array(state, dtype=jnp.complex128),
            jnp.array(M_small, dtype=jnp.complex128),
            wires, inv, d, n_qudit))
        assert np.allclose(expected, got, atol=1e-12), (
            f"mismatch d={d} n={n_qudit} wires={wires}")

    def test_d3_n3_wires_1(self):
        rng = np.random.default_rng(1)
        M = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
        self._run(3, 3, (1,), M, seed=1)

    def test_d4_n3_wires_0_2(self):
        rng = np.random.default_rng(2)
        M = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))
        self._run(4, 3, (0, 2), M, seed=2)

    def test_d5_n2_full_system(self):
        rng = np.random.default_rng(3)
        M = rng.standard_normal((25, 25)) + 1j * rng.standard_normal((25, 25))
        self._run(5, 2, (0, 1), M, seed=3)

    def test_identity_sentinel_wires_empty(self):
        from src.jax_backend import _apply_local_matrix
        d, n = 3, 4
        dim = d ** n
        rng = np.random.default_rng(4)
        state = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        M = jnp.array([[1.0 + 0j]])
        got = np.asarray(_apply_local_matrix(
            jnp.array(state, dtype=jnp.complex128), M, (),
            tuple(range(n)), d, n))
        assert np.allclose(got, state, atol=1e-12)

    def test_vmap_over_matrix_axis(self):
        from src.jax_backend import _apply_local_matrix
        d, n_qudit, wires = 3, 3, (0, 1)
        inv = _inverse_perm(wires, n_qudit)
        rng = np.random.default_rng(5)
        dim = d ** n_qudit
        state = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        state = jnp.array(state, dtype=jnp.complex128)
        stack = np.stack([rng.standard_normal((9, 9)) +
                          1j * rng.standard_normal((9, 9)) for _ in range(4)])
        stack = jnp.array(stack, dtype=jnp.complex128)

        batched = jax.vmap(
            _apply_local_matrix,
            in_axes=(None, 0, None, None, None, None))(
                state, stack, wires, inv, d, n_qudit)

        for k in range(4):
            serial = _apply_local_matrix(state, stack[k], wires, inv, d, n_qudit)
            assert np.allclose(np.asarray(batched[k]), np.asarray(serial),
                               atol=1e-12)


# ── step 3: create_jax_loss_vmap ──────────────────────────────────────

def _scan_reference_loss(code_states, E_det_dense, K, distance):
    """Python-loop reference loss over dense errors, takes code_states directly.

    Independent implementation used as the cross-check against the vmap path
    in the scalar/gradient parity tests. Iterates over a flat `E_det` list
    in native Python/JAX with no vmap and no grouping, to guarantee the
    parity check isn't measuring a shared bug in the grouping code.
    """
    E_det_jax = jnp.stack(
        [jnp.asarray(E, dtype=jnp.complex128) for E in E_det_dense])
    total = jnp.array(0.0, dtype=jnp.float64)
    for E in E_det_jax:
        E_applied = E @ code_states.T
        overlaps = jnp.conj(code_states) @ E_applied
        for i in range(K):
            for j in range(i + 1, K):
                total = total + jnp.abs(overlaps[i, j]) ** 2
        if distance >= 3:
            diag_vals = jnp.diag(overlaps)
            total = total + (K / 4) * jnp.real(jnp.var(diag_vals))
    return total


class TestCreateJaxLossVmap:
    def _random_code_states(self, K, dim, seed):
        rng = np.random.default_rng(seed)
        arr = (rng.standard_normal((K, dim)) +
               1j * rng.standard_normal((K, dim)))
        # orthonormalize via QR so they are a valid (but random) codespace
        q, _ = np.linalg.qr(arr.T)
        return jnp.array(q.T[:K], dtype=jnp.complex128)

    def test_scalar_parity_d3_n5_dist3(self):
        from src.errors import ErrorModel
        from src.jax_backend import create_jax_loss_vmap

        d, n, dist, K = 3, 5, 3, 3
        model = ErrorModel(d=d, n_qudit=n, distance=dist, closed=True)
        E_det_dense, _ = model.build_dense()
        grouped = model.build_grouped(verbose=False)

        cs = self._random_code_states(K, d ** n, seed=42)

        loss_vmap_fn = create_jax_loss_vmap(grouped, K, d, n, dist)
        val_vmap = float(loss_vmap_fn(cs))
        val_scan = float(_scan_reference_loss(cs, E_det_dense, K, dist))

        assert abs(val_vmap - val_scan) < 1e-6, (
            f"vmap={val_vmap}  scan={val_scan}  diff={val_vmap - val_scan}")

    def test_gradient_parity_d3_n5_dist3(self):
        from src.errors import ErrorModel
        from src.jax_backend import create_jax_loss_vmap

        d, n, dist, K = 3, 5, 3, 3
        model = ErrorModel(d=d, n_qudit=n, distance=dist, closed=True)
        E_det_dense, _ = model.build_dense()
        grouped = model.build_grouped(verbose=False)

        cs = self._random_code_states(K, d ** n, seed=7)
        loss_vmap_fn = create_jax_loss_vmap(grouped, K, d, n, dist)

        grad_vmap = jax.grad(
            lambda s: jnp.real(loss_vmap_fn(s)), holomorphic=False)(cs)
        grad_scan = jax.grad(
            lambda s: jnp.real(_scan_reference_loss(s, E_det_dense, K, dist)),
            holomorphic=False)(cs)

        diff = jnp.linalg.norm(grad_vmap - grad_scan)
        assert float(diff) < 1e-6, f"gradient L2 diff = {float(diff)}"

    def test_dedup_equivalence_d3_n5_dist3(self):
        """Loss is unchanged whether dedup is on or off on main's closed basis."""
        from src.errors import ErrorModel
        from src.jax_backend import create_jax_loss_vmap

        d, n, dist, K = 3, 5, 3, 3
        model = ErrorModel(d=d, n_qudit=n, distance=dist, closed=True)
        g_dedup = model.build_grouped(dedup=True, verbose=False)
        g_nodedup = model.build_grouped(dedup=False, verbose=False)

        cs = self._random_code_states(K, d ** n, seed=11)
        v_dedup = float(create_jax_loss_vmap(g_dedup, K, d, n, dist)(cs))
        v_nodedup = float(create_jax_loss_vmap(g_nodedup, K, d, n, dist)(cs))
        assert abs(v_dedup - v_nodedup) < 1e-9

    def test_smoke_d4_n5_dist3(self):
        from src.errors import ErrorModel
        from src.jax_backend import create_jax_loss_vmap

        d, n, dist, K = 4, 5, 3, 4
        model = ErrorModel(d=d, n_qudit=n, distance=dist, closed=True)
        grouped = model.build_grouped(verbose=False)
        cs = self._random_code_states(K, d ** n, seed=13)
        val = float(create_jax_loss_vmap(grouped, K, d, n, dist)(cs))
        assert np.isfinite(val)

    def test_smoke_d5_n3_dist2(self):
        from src.errors import ErrorModel
        from src.jax_backend import create_jax_loss_vmap

        d, n, dist, K = 5, 3, 2, 5
        model = ErrorModel(d=d, n_qudit=n, distance=dist, closed=False)
        grouped = model.build_grouped(verbose=False)
        cs = self._random_code_states(K, d ** n, seed=17)
        val = float(create_jax_loss_vmap(grouped, K, d, n, dist)(cs))
        assert np.isfinite(val)


# ── step 4: build_varqec_loss vmap path + 5-step training ────────────

class TestBuildVarqecLoss:
    def test_missing_grouped_raises(self):
        from src.jax_backend import build_varqec_loss
        with pytest.raises(ValueError):
            build_varqec_loss(encoder_fn=lambda p, k: jnp.zeros(9),
                              K=3, d=3, n_qudit=2, distance=2,
                              E_det_grouped=None)

    def test_jax_vmap_5step_training_decreases(self):
        """Loss strictly decreases between step 0 and step 5 with jax_vmap."""
        from src.errors import ErrorModel
        from src.jax_backend import (
            create_jax_encoder, build_varqec_loss, train_jax)

        d, n, dist, K = 3, 5, 3, 3
        enc, _, ppl = create_jax_encoder(n, d)
        model = ErrorModel(d=d, n_qudit=n, distance=dist, closed=True)
        grouped = model.build_grouped(verbose=False)

        loss_fn = build_varqec_loss(
            encoder_fn=enc, K=K, d=d, n_qudit=n, distance=dist,
            E_det_grouped=grouped)

        # 5-step training smoke: last < first.
        _, losses = train_jax(
            loss_fn, n_layers=4, params_per_layer=ppl,
            n_steps=5, lr=0.05, lr_switch=0.01, seed=0)
        assert len(losses) == 5
        assert np.isfinite(losses[0]) and np.isfinite(losses[-1])
        assert losses[-1] < losses[0], (
            f"loss did not decrease: step0={losses[0]} step5={losses[-1]}")


# ── F.1: vmap K-codeword stack equivalence + shape ───────────────────

class TestCodewordStackVmap:
    def test_shape_is_K_dim(self):
        """jax.vmap(encoder, in_axes=(None, 0)) produces (K, dim), no extra axis."""
        from src.jax_backend import create_jax_encoder
        d, n, K = 3, 3, 3
        enc, _, ppl = create_jax_encoder(n, d)
        params = jax.random.uniform(
            jax.random.PRNGKey(0), (1, ppl), minval=0.0, maxval=2 * np.pi)
        cs = jax.vmap(enc, in_axes=(None, 0))(params, jnp.arange(K))
        assert cs.shape == (K, d ** n), f"got {cs.shape}, expected ({K}, {d**n})"

    def test_equivalence_with_python_stack(self):
        """build_varqec_loss with vmap stack matches a python-stack reference."""
        from src.errors import ErrorModel
        from src.jax_backend import (
            create_jax_encoder, create_jax_loss_vmap, build_varqec_loss)

        d, n, dist, K = 3, 4, 2, 3
        enc, _, ppl = create_jax_encoder(n, d)
        model = ErrorModel(d=d, n_qudit=n, distance=dist, closed=False)
        grouped = model.build_grouped(verbose=False)

        # reference: python-stack version, same math, different composition
        core = create_jax_loss_vmap(grouped, K, d, n, dist)
        ref_loss_fn = jax.jit(lambda p: core(
            jnp.stack([enc(p, k) for k in range(K)])))

        cur_loss_fn = build_varqec_loss(
            encoder_fn=enc, K=K, d=d, n_qudit=n, distance=dist,
            E_det_grouped=grouped)

        params = jax.random.uniform(
            jax.random.PRNGKey(3), (2, ppl), minval=0.0, maxval=2 * np.pi)
        v_ref = float(ref_loss_fn(params))
        v_cur = float(cur_loss_fn(params))
        assert abs(v_ref - v_cur) < 1e-10, f"ref={v_ref} cur={v_cur}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
