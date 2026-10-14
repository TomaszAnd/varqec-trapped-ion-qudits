"""channel-weighted i.i.d. benchmark: weights + noise_fn.

Pins hardware_benchmark_weights (src/meth_weights.py) against the T1a
derivation values and validates make_weighted_hardware_noise_fn
(src/errors.py) draw statistics, calibration identity, and protocol
equivalence with the uniform instrument.
"""
import numpy as np
import pytest

import src.jax_backend  # noqa: F401  (x64 first)
from src.errors import (qudit_hardware_error_basis, make_hardware_noise_fn,
                        make_weighted_hardware_noise_fn)
from src.meth_weights import hardware_benchmark_weights, build_per_role_weights


# T1a derivation pins (r3_campaign/t1a_derivation.json, eta=0.9296, n_max=2)
PIN_METH_D3 = [0.210199, 0.306935, 0.241029, 0.241837]
PIN_DEPH_D3 = [0.014409, 0.655734, 0.163933, 0.165924]
PIN_METH_D4 = [0.152879, 0.154376, 0.197637, 0.171121, 0.150574, 0.173413]
PIN_DEPH_D5 = [0.099370, 0.016110, 0.112872, 0.306815,
               0.204116, 0.028793, 0.039913, 0.192012]


class TestHardwareBenchmarkWeights:
    def test_probability_vector(self):
        for d in (3, 4, 5):
            for kind in ("meth-spectator", "dephasing-exact"):
                w = hardware_benchmark_weights(d, kind=kind)
                assert w.shape == (2 * (d - 1),)
                assert np.all(w >= 0)
                assert np.isclose(w.sum(), 1.0)

    def test_pinned_values(self):
        assert np.allclose(hardware_benchmark_weights(3), PIN_METH_D3,
                           atol=1e-5)
        assert np.allclose(
            hardware_benchmark_weights(3, kind="dephasing-exact"),
            PIN_DEPH_D3, atol=1e-5)
        assert np.allclose(hardware_benchmark_weights(4), PIN_METH_D4,
                           atol=1e-5)
        assert np.allclose(
            hardware_benchmark_weights(5, kind="dephasing-exact"),
            PIN_DEPH_D5, atol=1e-5)

    def test_consistent_with_training_prior(self):
        # meth-spectator weights = the training prior's spectator-twirl
        # projections restricted to the benchmark ops, renormalized.
        d = 3
        pr = build_per_role_weights(d, eta=0.9296, n_max=2)
        lam = np.asarray(pr["spectator"][1:1 + 2 * (d - 1)])
        assert np.allclose(hardware_benchmark_weights(d), lam / lam.sum(),
                           atol=1e-12)

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            hardware_benchmark_weights(3, kind="nope")

    def test_calibration_identity_trace(self):
        # Every benchmark op has Tr = d-2 -> identical single-op average
        # infidelity; injection probability alone calibrates the benchmark.
        for d in (3, 4, 5):
            for E in qudit_hardware_error_basis(d):
                assert np.isclose(abs(np.trace(np.asarray(E))), d - 2)


class TestWeightedNoiseFn:
    def test_rejects_bad_weights(self):
        with pytest.raises(ValueError):
            make_weighted_hardware_noise_fn(3, 2, 0.05, [0.5, 0.5])  # len
        with pytest.raises(ValueError):
            make_weighted_hardware_noise_fn(3, 2, 0.05, [0.7, 0.2, 0.2, 0.2])

    def test_deterministic_single_op(self):
        # weights concentrated on Z_1: any injected error must be Z_1
        d, n = 3, 2
        fn = make_weighted_hardware_noise_fn(d, n, 1.0, [1.0, 0, 0, 0])
        rng = np.random.default_rng(0)
        state = np.zeros(d ** n, complex)
        state[4] = 1.0  # |11>
        out = fn(state, rng)
        # Z_1 on both qudits: |11> -> (-1)^2 |11>
        assert np.allclose(np.abs(out), np.abs(state))
        assert np.isclose(abs(np.vdot(state, out)), 1.0)

    def test_draw_frequencies_match_weights(self):
        d, n = 3, 1
        w = hardware_benchmark_weights(d)
        fn = make_weighted_hardware_noise_fn(d, n, 1.0, w)
        errs = [np.asarray(E) for E in qudit_hardware_error_basis(d)]
        # distinguish ops by action on a probe state
        probe = np.array([1.0, 0.7, 0.4], complex)
        probe /= np.linalg.norm(probe)
        sigs = [E @ probe for E in errs]
        rng = np.random.default_rng(123)
        counts = np.zeros(len(errs))
        n_draws = 20000
        for _ in range(n_draws):
            out = fn(probe, rng)
            ov = [abs(np.vdot(s, out)) for s in sigs]
            counts[int(np.argmax(ov))] += 1
        freq = counts / n_draws
        assert np.allclose(freq, w, atol=0.01)

    def test_uniform_weights_match_uniform_statistics(self):
        # with uniform weights the injected-error statistics coincide with
        # make_hardware_noise_fn (not stream-identical; statistically equal)
        d, n, p, shots = 3, 2, 0.3, 4000
        state = np.ones(d ** n, complex) / np.sqrt(d ** n)
        uni = make_hardware_noise_fn(d, n, p)
        wfn = make_weighted_hardware_noise_fn(d, n, p, np.ones(4) / 4)
        f_u, f_w = [], []
        rng1, rng2 = (np.random.default_rng(7), np.random.default_rng(8))
        for _ in range(shots):
            f_u.append(abs(np.vdot(state, uni(state, rng1))) ** 2)
            f_w.append(abs(np.vdot(state, wfn(state, rng2))) ** 2)
        assert abs(np.mean(f_u) - np.mean(f_w)) < 0.02

    def test_injection_probability_unchanged(self):
        # marginal per-qudit error probability is p for any weights
        d, n, p = 3, 1, 0.4
        w = hardware_benchmark_weights(d, kind="dephasing-exact")
        fn = make_weighted_hardware_noise_fn(d, n, p, w)
        probe = np.array([1.0, 0.5, 0.25], complex)
        probe /= np.linalg.norm(probe)
        rng = np.random.default_rng(42)
        n_draws, changed = 20000, 0
        for _ in range(n_draws):
            out = fn(probe, rng)
            if abs(abs(np.vdot(probe, out)) - 1.0) > 1e-12:
                changed += 1
        # every benchmark op moves this probe; changed/n estimates p
        assert abs(changed / n_draws - p) < 0.02
