"""single-transition MS entangler variant ('ms1').

Pins: (1) parameter layout; (2) EXACT equivalence to the full ms encoder
with the transition-(1,2) MS angles zeroed (theta=0 -> identity), which
also pins that the ms path itself is untouched; (3) build_race dispatch
produces a finite loss on the ms1 arm; (4) encoder_forward npz dispatch
accepts 'ms1'.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.jax_backend  # noqa: E402  (x64 first — PERF_AUDIT footgun)
import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from src.jax_backend import (create_jax_encoder,  # noqa: E402
                             create_jax_encoder_variant, VARIANT_ENTANGLERS)

N, D, L = 5, 3, 2
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]


def test_ms1_registered():
    assert "ms1" in VARIANT_ENTANGLERS


def test_ms1_params_per_layer():
    _, _, ppl = create_jax_encoder_variant(N, D, connections=CONNS,
                                           entangler="ms1")
    assert ppl == 5 * N * (D - 1) + 2 * len(CONNS)  # 70 at n=5, d=3


def test_ms1_equals_ms_with_second_transition_zeroed():
    enc1, _, ppl1 = create_jax_encoder_variant(N, D, connections=CONNS,
                                               entangler="ms1")
    encf, _, pplf = create_jax_encoder(N, D, connections=CONNS)
    rng = np.random.default_rng(7)
    p1 = rng.uniform(0, 2 * np.pi, size=(L, ppl1))
    nsing = 4 * N * (D - 1)          # two XY rounds
    nc = len(CONNS)
    pf = np.zeros((L, pplf))
    pf[:, :nsing] = p1[:, :nsing]
    for c in range(nc):
        # ms layout per conn: (phi,theta) at t=(0,1) then (phi,theta) at
        # t=(1,2); theta=0 makes the (1,2) MS the identity for any phi.
        pf[:, nsing + 4 * c: nsing + 4 * c + 2] = \
            p1[:, nsing + 2 * c: nsing + 2 * c + 2]
        pf[:, nsing + 4 * c + 2] = 0.7   # arbitrary phi, inert at theta=0
        pf[:, nsing + 4 * c + 3] = 0.0   # theta = 0
    pf[:, nsing + 4 * nc:] = p1[:, nsing + 2 * nc:]   # Z round
    for k in range(D):
        s1 = np.asarray(enc1(jnp.asarray(p1), k))
        sf = np.asarray(encf(jnp.asarray(pf), k))
        assert np.max(np.abs(s1 - sf)) < 1e-12
        assert abs(np.linalg.norm(s1) - 1.0) < 1e-12


def test_ms1_build_race_dispatch():
    from src.seed_race import build_race
    (vg, full_loss_fn, gs, ppl, conns, fw, sgw) = build_race(
        D, N, 3, L, noise="hw", entangler="ms1")
    assert ppl == 5 * N * (D - 1) + 2 * len(conns)
    theta = jnp.asarray(np.random.default_rng(0).uniform(
        0, 2 * np.pi, size=(L, ppl)))
    lv = float(full_loss_fn(theta))
    assert np.isfinite(lv) and lv > 0


def test_ms1_encoder_forward_dispatch(tmp_path):
    from src.loss import save_varqec_result
    from src.decoders._common import encoder_forward
    _, _, ppl = create_jax_encoder_variant(N, D, connections=CONNS,
                                           entangler="ms1")
    rng = np.random.default_rng(3)
    params = rng.uniform(0, 2 * np.pi, size=(L, ppl))
    npz = os.path.join(tmp_path, "ms1_smoke.npz")
    save_varqec_result(npz, params, [1.0], f"native_d{D}", 3, L,
                       metadata={"K": D, "n_qudit": N, "d": D,
                                 "entangler": "ms1"})
    cs, d_, n_, K_ = encoder_forward(npz, CONNS)
    assert cs.shape == (D, D ** N)
    g = np.asarray(cs) @ np.asarray(cs).conj().T
    assert np.max(np.abs(g - np.eye(D))) < 1e-10
