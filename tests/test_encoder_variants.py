"""Pin create_jax_encoder_variant (ansatz/entangler study wiring).

(a) entangler="ms" delegates to create_jax_encoder — identical states;
(b) each variant encoder matches a dense numpy reference that applies the
    same layer structure with gate matrices taken from src.gates (an
    independent embedding path via apply_local_np);
(c) norm preservation;
(d) params_per_layer formulas;
(e) encoder_forward dispatches on the npz entangler field and raises on
    unknown values.
"""
import os
import tempfile

import numpy as np
import pytest

import src.jax_backend as jb  # sets jax_enable_x64 before any jnp array
import jax
import jax.numpy as jnp

from src.gates import CSUM_gate, CEX_gate, zz_subspace_gate, ls_global_gate
from src.decoders._common import apply_local_np, encoder_forward

D, N, L = 3, 3, 2
CONNS = [[0, 1], [0, 2], [1, 2]]
NT = D - 1


def _apply_np(state, U, wires):
    left = [i for i in range(N) if i not in wires]
    perm = list(wires) + left
    inv = tuple(perm.index(i) for i in range(N))
    return apply_local_np(state, np.asarray(U, complex), tuple(wires), inv,
                          d=D, n_qudit=N)


def _reference_encoder(params, code_ind, entangler):
    """Dense numpy re-implementation of the variant layer structure."""
    state = np.zeros(D ** N, dtype=complex)
    state[code_ind * D ** (N - 1)] = 1.0
    for l in range(params.shape[0]):
        p = np.asarray(params[l])
        pi = 0
        for q in range(N):
            for _round in range(2):
                for t in range(NT):
                    U = np.asarray(jb._xy_gate_jax(p[pi], p[pi + 1],
                                                   t, t + 1, D))
                    state = _apply_np(state, U, (q,))
                    pi += 2
        for (q1, q2) in CONNS:
            if entangler == "zz":
                for t in range(NT):
                    U = zz_subspace_gate(p[pi], t, t + 1, d=D)
                    state = _apply_np(state, U, (q1, q2))
                    pi += 1
            elif entangler == "ls":
                U = ls_global_gate(p[pi], d=D)
                state = _apply_np(state, U, (q1, q2))
                pi += 1
            elif entangler == "csum":
                state = _apply_np(state, CSUM_gate(d=D), (q1, q2))
            elif entangler == "cex":
                state = _apply_np(state, CEX_gate(d=D), (q1, q2))
        for q in range(N):
            for t in range(NT):
                U = np.asarray(jb._z_gate_jax(p[pi], t, t + 1, D))
                state = _apply_np(state, U, (q,))
                pi += 1
    return state


EXPECTED_PPL = {
    "ms": (5 * N + 2 * len(CONNS)) * NT,
    "zz": 5 * N * NT + len(CONNS) * NT,
    "ls": 5 * N * NT + len(CONNS),
    "csum": 5 * N * NT,
    "cex": 5 * N * NT,
}


def _params(ppl, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(0, 2 * np.pi, size=(L, ppl)))


def test_ms_delegates_to_create_jax_encoder():
    enc_v, _, ppl_v = jb.create_jax_encoder_variant(
        N, D, connections=CONNS, entangler="ms", use_scan=True)
    enc_o, _, ppl_o = jb.create_jax_encoder(
        N, D, connections=CONNS, use_scan=True)
    assert ppl_v == ppl_o == EXPECTED_PPL["ms"]
    params = _params(ppl_v)
    for k in range(D):
        np.testing.assert_array_equal(np.asarray(enc_v(params, k)),
                                      np.asarray(enc_o(params, k)))


@pytest.mark.parametrize("entangler", ["zz", "ls", "csum", "cex"])
def test_variant_matches_dense_reference(entangler):
    enc, _, ppl = jb.create_jax_encoder_variant(
        N, D, connections=CONNS, entangler=entangler, use_scan=False)
    assert ppl == EXPECTED_PPL[entangler]
    params = _params(ppl, seed=42)
    for k in range(D):
        got = np.asarray(enc(params, k))
        want = _reference_encoder(params, k, entangler)
        assert np.linalg.norm(got) == pytest.approx(1.0, abs=1e-12)
        np.testing.assert_allclose(got, want, atol=1e-12)


@pytest.mark.parametrize("entangler", ["zz", "ls", "csum", "cex"])
def test_scan_matches_unrolled(entangler):
    enc_s, _, ppl = jb.create_jax_encoder_variant(
        N, D, connections=CONNS, entangler=entangler, use_scan=True)
    enc_u, _, _ = jb.create_jax_encoder_variant(
        N, D, connections=CONNS, entangler=entangler, use_scan=False)
    params = _params(ppl, seed=7)
    for k in range(D):
        np.testing.assert_allclose(np.asarray(enc_s(params, k)),
                                   np.asarray(enc_u(params, k)), atol=1e-14)


def test_unknown_entangler_raises():
    with pytest.raises(ValueError):
        jb.create_jax_encoder_variant(N, D, connections=CONNS,
                                      entangler="nope")


def test_encoder_forward_dispatch(tmp_path):
    enc, _, ppl = jb.create_jax_encoder_variant(
        N, D, connections=CONNS, entangler="zz", use_scan=True)
    params = _params(ppl, seed=3)
    want = np.stack([np.asarray(enc(params, k)) for k in range(D)])

    npz = os.path.join(tmp_path, "zz_code.npz")
    np.savez(npz, params=np.asarray(params), d=D, n_qudit=N, K=D,
             entangler="zz")
    got, d, n, K = encoder_forward(npz, CONNS)
    assert (d, n, K) == (D, N, D)
    np.testing.assert_allclose(np.asarray(got), want, atol=1e-14)

    bad = os.path.join(tmp_path, "bad_code.npz")
    np.savez(bad, params=np.asarray(params), d=D, n_qudit=N, K=D,
             entangler="mystery")
    with pytest.raises(ValueError):
        encoder_forward(bad, CONNS)
