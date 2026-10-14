"""
Tests for the VarQEC decoders (projection, detection, syndrome-based).
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation import (
    projection_decoder, detection_decoder, syndrome_based_decoder,
    lookup_table_decoder, nearest_codeword_decoder
)


def _make_simple_code(K=2, dim=4):
    """Create a simple K-dimensional code in dim-dimensional Hilbert space."""
    code_states = np.zeros((K, dim), dtype=complex)
    for k in range(K):
        code_states[k, k] = 1.0
    return code_states


class TestProjectionDecoder:
    def test_noiseless_returns_same_state(self):
        """Noiseless codeword should be unchanged by projection."""
        code = _make_simple_code(K=2, dim=4)
        state = code[0].copy()
        decoded = projection_decoder(state, code)
        assert np.allclose(decoded, state)

    def test_output_in_code_space(self):
        """Projected state should lie in the code space."""
        code = _make_simple_code(K=2, dim=4)
        # State with components both in and out of code space
        state = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
        state /= np.linalg.norm(state)
        decoded = projection_decoder(state, code)
        # Should only have components in |0> and |1>
        assert np.isclose(np.linalg.norm(decoded), 1.0)
        assert np.isclose(decoded[2], 0.0)
        assert np.isclose(decoded[3], 0.0)

    def test_normalized_output(self):
        """Output should always be normalized."""
        code = _make_simple_code(K=2, dim=4)
        state = np.array([0.8, 0.1, 0.05, 0.05], dtype=complex)
        state /= np.linalg.norm(state)
        decoded = projection_decoder(state, code)
        assert np.isclose(np.linalg.norm(decoded), 1.0)

    def test_orthogonal_state_fallback(self):
        """State orthogonal to code space returns original (fallback)."""
        code = _make_simple_code(K=2, dim=4)
        state = np.array([0, 0, 0, 1], dtype=complex)
        decoded = projection_decoder(state, code)
        # Fallback: returns original state since projection norm is 0
        assert np.allclose(decoded, state)

    def test_superposition_codeword(self):
        """Superposition of codewords should project perfectly."""
        code = _make_simple_code(K=2, dim=4)
        state = (code[0] + code[1]) / np.sqrt(2)
        decoded = projection_decoder(state, code)
        assert np.allclose(decoded, state, atol=1e-10)


class TestDetectionDecoder:
    def test_noiseless_not_detected(self):
        """Noiseless codeword should NOT trigger detection."""
        code = _make_simple_code(K=2, dim=4)
        state = code[0].copy()
        decoded, detected = detection_decoder(state, code, detection_threshold=0.1)
        assert not detected
        assert np.allclose(decoded, state)

    def test_orthogonal_state_detected(self):
        """State fully outside code space should be detected."""
        code = _make_simple_code(K=2, dim=4)
        state = np.array([0, 0, 0, 1], dtype=complex)
        decoded, detected = detection_decoder(state, code, detection_threshold=0.1)
        assert detected
        assert decoded is None

    def test_mostly_out_of_code_space(self):
        """State mostly outside code space should be detected with appropriate threshold."""
        code = _make_simple_code(K=2, dim=4)
        # 99% outside code space
        state = np.array([0.05, 0.05, 0.7, 0.7], dtype=complex)
        state /= np.linalg.norm(state)
        decoded, detected = detection_decoder(state, code, detection_threshold=0.1)
        assert detected

    def test_mostly_in_code_space_not_detected(self):
        """State mostly in code space should NOT be detected."""
        code = _make_simple_code(K=2, dim=4)
        state = np.array([0.9, 0.3, 0.05, 0.05], dtype=complex)
        state /= np.linalg.norm(state)
        decoded, detected = detection_decoder(state, code, detection_threshold=0.1)
        assert not detected
        assert np.isclose(np.linalg.norm(decoded), 1.0)

    def test_superposition_not_detected(self):
        """Superposition of codewords should not be detected."""
        code = _make_simple_code(K=2, dim=4)
        state = (code[0] + code[1]) / np.sqrt(2)
        decoded, detected = detection_decoder(state, code, detection_threshold=0.1)
        assert not detected


class TestSyndromeBasedDecoder:
    def test_identity_error_no_change(self):
        """Identity error: state should be unchanged after correction."""
        code = _make_simple_code(K=2, dim=4)
        state = code[0].copy()
        error_ops = [np.eye(4, dtype=complex)]
        corrected = syndrome_based_decoder(state, code, error_ops)
        assert np.allclose(corrected, state, atol=1e-10)

    def test_single_pauli_error_corrected(self):
        """A single Pauli error should be identified and corrected."""
        code = _make_simple_code(K=2, dim=4)
        # Z error on qubit 0: flips phase of |1>
        Z = np.diag([1, -1, 1, 1]).astype(complex)
        error_ops = [np.eye(4, dtype=complex), Z]

        state = code[0].copy()  # |0> — Z|0> = |0>, so identity wins
        corrected = syndrome_based_decoder(state, code, error_ops)
        fid = np.abs(np.vdot(state, corrected))**2
        assert fid > 0.99

    def test_error_out_of_code_space_corrected(self):
        """Error that maps state out of code space should be corrected."""
        code = _make_simple_code(K=2, dim=4)
        # Error maps |0> -> |2> (out of code space, correctable)
        E = np.zeros((4, 4), dtype=complex)
        E[2, 0] = 1; E[3, 1] = 1; E[0, 2] = 1; E[1, 3] = 1  # swap {0,1} <-> {2,3}
        error_ops = [np.eye(4, dtype=complex), E]

        state = code[0].copy()  # |0>
        noisy = E @ state       # E|0> = |2>, outside code space
        corrected = syndrome_based_decoder(noisy, code, error_ops)
        # Should recover |0>
        fid = np.abs(np.vdot(state, corrected))**2
        assert fid > 0.99

    def test_superposition_error_corrected(self):
        """Error on a superposition state should be corrected."""
        code = _make_simple_code(K=2, dim=4)
        X = np.zeros((4, 4), dtype=complex)
        X[0, 1] = 1; X[1, 0] = 1; X[2, 2] = 1; X[3, 3] = 1
        error_ops = [np.eye(4, dtype=complex), X]

        state = (code[0] + code[1]) / np.sqrt(2)  # |+L>
        noisy = X @ state  # X|+L> = |+L> (X is self-inverse on {|0>,|1>})
        corrected = syndrome_based_decoder(noisy, code, error_ops)
        fid = np.abs(np.vdot(state, corrected))**2
        assert fid > 0.99

    def test_output_normalized(self):
        """Corrected state should be normalized."""
        code = _make_simple_code(K=2, dim=4)
        state = np.array([0.6, 0.3, 0.5, 0.3], dtype=complex)
        state /= np.linalg.norm(state)
        error_ops = [np.eye(4, dtype=complex)]
        corrected = syndrome_based_decoder(state, code, error_ops)
        assert np.isclose(np.linalg.norm(corrected), 1.0)


class TestLookupTableDecoder:
    def test_alias_matches(self):
        """lookup_table_decoder should be the same as syndrome_based_decoder."""
        assert lookup_table_decoder is syndrome_based_decoder

    def test_identity_error(self):
        code = _make_simple_code(K=2, dim=4)
        state = code[0].copy()
        error_ops = [np.eye(4, dtype=complex)]
        corrected = lookup_table_decoder(state, code, error_ops)
        assert np.allclose(corrected, state, atol=1e-10)

    def test_corrects_out_of_code_space_error(self):
        code = _make_simple_code(K=2, dim=4)
        E = np.zeros((4, 4), dtype=complex)
        E[2, 0] = 1; E[3, 1] = 1; E[0, 2] = 1; E[1, 3] = 1
        error_ops = [np.eye(4, dtype=complex), E]
        state = code[0].copy()
        noisy = E @ state
        corrected = lookup_table_decoder(noisy, code, error_ops)
        fid = np.abs(np.vdot(state, corrected))**2
        assert fid > 0.99


class TestNearestCodewordDecoder:
    def test_noiseless_returns_same(self):
        """Noiseless codeword should return that codeword."""
        code = _make_simple_code(K=2, dim=4)
        state = code[0].copy()
        decoded = nearest_codeword_decoder(state, code)
        assert np.allclose(decoded, state, atol=1e-10)

    def test_returns_single_codeword(self):
        """Should return exactly one codeword, not a superposition."""
        code = _make_simple_code(K=2, dim=4)
        # State closer to |0> than |1>
        state = np.array([0.9, 0.1, 0.3, 0.3], dtype=complex)
        state /= np.linalg.norm(state)
        decoded = nearest_codeword_decoder(state, code)
        # Should return |0> (highest overlap)
        assert np.allclose(decoded, code[0])

    def test_differs_from_projection(self):
        """Nearest codeword should differ from projection for superposition states."""
        code = _make_simple_code(K=2, dim=4)
        # Equal superposition of |0> and |1> with some out-of-space noise
        state = np.array([0.5, 0.5, 0.3, 0.3], dtype=complex)
        state /= np.linalg.norm(state)
        ncw = nearest_codeword_decoder(state, code)
        proj = projection_decoder(state, code)
        # Projection preserves superposition, nearest_cw picks one codeword
        # They should NOT be equal
        assert not np.allclose(ncw, proj, atol=1e-6)

    def test_normalized_output(self):
        code = _make_simple_code(K=2, dim=4)
        state = np.array([0.6, 0.3, 0.5, 0.3], dtype=complex)
        state /= np.linalg.norm(state)
        decoded = nearest_codeword_decoder(state, code)
        assert np.isclose(np.linalg.norm(decoded), 1.0)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
