"""
Tests for the logical error rate simulation module.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation import (
    build_code_projector, apply_diagonal_noise, apply_pauli_noise,
    code_space_fidelity, simulate_logical_error_rate,
    sweep_logical_error_rate, make_correlated_dephasing_noise_fn,
)


def _make_simple_code(K=2, dim=4):
    """Create a simple K-dimensional code in dim-dimensional Hilbert space."""
    # Use computational basis states as codewords
    code_states = np.zeros((K, dim), dtype=complex)
    for k in range(K):
        code_states[k, k] = 1.0
    return code_states


class TestCodeProjector:
    def test_projector_rank(self):
        code = _make_simple_code(K=2, dim=4)
        P = build_code_projector(code)
        assert np.isclose(np.trace(P), 2.0), "Rank should equal K=2"

    def test_projector_idempotent(self):
        code = _make_simple_code(K=2, dim=4)
        P = build_code_projector(code)
        assert np.allclose(P @ P, P), "Projector should be idempotent"


class TestApplyDiagonalNoise:
    def test_noiseless(self):
        """Identity Kraus (eta=1) should not change the state."""
        state = np.array([1, 0, 0, 0], dtype=complex)
        # Noiseless: E_0 = I (all ones), no higher orders
        kraus_diags = [np.ones(4, dtype=complex)]
        rng = np.random.default_rng(42)
        noisy = apply_diagonal_noise(state, kraus_diags, rng)
        assert np.allclose(noisy, state)

    def test_normalized(self):
        """Output should be normalized."""
        state = np.array([1, 1, 0, 0], dtype=complex) / np.sqrt(2)
        kraus_diags = [np.array([0.9, 0.8, 0.7, 0.6], dtype=complex),
                       np.array([0.1, 0.2, 0.3, 0.4], dtype=complex)]
        rng = np.random.default_rng(42)
        noisy = apply_diagonal_noise(state, kraus_diags, rng)
        assert np.isclose(np.linalg.norm(noisy), 1.0)


class TestCodeSpaceFidelity:
    def test_perfect_fidelity(self):
        code = _make_simple_code(K=2, dim=4)
        coeffs = np.array([1, 0], dtype=complex)
        state = code[0]
        assert np.isclose(code_space_fidelity(state, code, coeffs), 1.0)

    def test_orthogonal_fidelity(self):
        code = _make_simple_code(K=2, dim=4)
        coeffs = np.array([1, 0], dtype=complex)
        state = np.array([0, 0, 0, 1], dtype=complex)  # outside code space
        assert np.isclose(code_space_fidelity(state, code, coeffs), 0.0)


class TestSimulateLogicalErrorRate:
    def test_no_noise_no_errors(self):
        """With identity noise, logical error rate should be 0."""
        code = _make_simple_code(K=2, dim=4)

        def no_noise(state, rng):
            return state.copy()

        result = simulate_logical_error_rate(code, no_noise, n_shots=100, seed=42)
        assert result['logical_error_rate'] == 0.0
        assert np.isclose(result['mean_fidelity'], 1.0, atol=1e-10)

    def test_heavy_noise_has_errors(self):
        """With strong noise, should have some logical errors."""
        code = _make_simple_code(K=2, dim=4)

        def scramble_noise(state, rng):
            # Return a random state — should cause errors
            s = rng.standard_normal(len(state)) + 1j * rng.standard_normal(len(state))
            return s / np.linalg.norm(s)

        result = simulate_logical_error_rate(code, scramble_noise, n_shots=100, seed=42)
        assert result['logical_error_rate'] > 0.0

    def test_returns_correct_fields(self):
        code = _make_simple_code(K=2, dim=4)

        def no_noise(state, rng):
            return state.copy()

        result = simulate_logical_error_rate(code, no_noise, n_shots=50, seed=42)
        assert 'logical_error_rate' in result
        assert 'mean_fidelity' in result
        assert 'std_fidelity' in result
        assert result['n_shots'] == 50


class TestSweepLogicalErrorRate:
    def test_sweep_increasing_errors(self):
        """Logical error rate should increase with noise strength."""
        code = _make_simple_code(K=2, dim=4)

        # Simple Pauli Z noise on qudit 0
        Z = np.diag([1, -1, 1, 1]).astype(complex)

        def factory(p):
            def noise_fn(state, rng):
                return apply_pauli_noise(state, [Z], p, rng)
            return noise_fn

        result = sweep_logical_error_rate(
            code, factory,
            physical_error_rates=[0.0, 0.5, 1.0],
            n_shots=200, seed=42
        )
        # At p=0, no errors. At p=1, all errors.
        assert result['logical_rates'][0] == 0.0
        assert result['logical_rates'][-1] > 0.0


class TestCorrelatedDephasing:
    def test_noiseless_correlated(self):
        """eta=1 should not change the state."""
        noise_fn = make_correlated_dephasing_noise_fn(
            n_qudits=2, d=3, gate_pairs=[(0, 1, 0, 1)],
            eta=1.0, n_max=5, noise_model="physical"
        )
        state = np.zeros(9, dtype=complex)
        state[0] = 1.0
        rng = np.random.default_rng(42)
        noisy = noise_fn(state, rng)
        assert np.allclose(np.abs(noisy), np.abs(state), atol=1e-10)

    def test_both_models_produce_output(self):
        """Both noise models should produce valid noise functions."""
        for model in ["physical", "simplified"]:
            noise_fn = make_correlated_dephasing_noise_fn(
                n_qudits=2, d=3, gate_pairs=[(0, 1, 0, 1)],
                eta=0.9, n_max=5, noise_model=model
            )
            state = np.zeros(9, dtype=complex)
            state[0] = 1.0
            rng = np.random.default_rng(42)
            noisy = noise_fn(state, rng)
            assert np.isclose(np.linalg.norm(noisy), 1.0)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
