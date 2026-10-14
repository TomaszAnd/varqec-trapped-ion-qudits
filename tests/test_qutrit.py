"""
Tests for native qutrit gates, error sets, and encoder.
Run: python3 -m pytest tests/test_qutrit.py -v
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gates import XY_gate, Z_gate, MS_gate
from src.errors import hardware_error_basis, extended_hardware_error_basis, build_qutrit_error_set
from src.encoder import create_native_encoder


class TestQutritGates:
    def test_XY_gate_unitary(self):
        for phi, alpha in [(0.0, 0.5), (1.2, np.pi), (0.3, 2.1)]:
            for j, k in [(0, 1), (1, 2)]:
                U = np.array(XY_gate(phi, alpha, j, k))
                assert np.allclose(U @ U.conj().T, np.eye(3), atol=1e-12), \
                    f"XY_gate({phi},{alpha},{j},{k}) not unitary"

    def test_XY_gate_identity_at_zero(self):
        U = np.array(XY_gate(0.0, 0.0, 0, 1))
        assert np.allclose(U, np.eye(3), atol=1e-12)

    def test_XY_gate_pi_is_swap(self):
        """XY(0, pi, 0, 1) should swap |0> and |1> (up to phase)."""
        U = np.array(XY_gate(0.0, np.pi, 0, 1))
        # |0> -> -i|1>, |1> -> -i|0>, |2> -> |2>
        assert abs(abs(U[1, 0]) - 1.0) < 1e-12
        assert abs(abs(U[0, 1]) - 1.0) < 1e-12
        assert abs(abs(U[2, 2]) - 1.0) < 1e-12

    def test_Z_gate_unitary(self):
        for theta in [0.0, 0.5, np.pi, 3.7]:
            for j, k in [(0, 1), (1, 2)]:
                U = np.array(Z_gate(theta, j, k))
                assert np.allclose(U @ U.conj().T, np.eye(3), atol=1e-12)

    def test_Z_gate_identity_at_zero(self):
        U = np.array(Z_gate(0.0, 0, 1))
        assert np.allclose(U, np.eye(3), atol=1e-12)

    def test_Z_gate_diagonal(self):
        U = np.array(Z_gate(1.3, 0, 1))
        assert np.allclose(U - np.diag(np.diag(U)), 0, atol=1e-12), "Z gate should be diagonal"

    def test_MS_gate_unitary_01(self):
        for phi, theta in [(0.0, 0.5), (1.2, np.pi/4), (0.3, 1.1)]:
            U = np.array(MS_gate(phi, theta, 0, 1))
            assert np.allclose(U @ U.conj().T, np.eye(9), atol=1e-12), \
                f"MS_gate({phi},{theta},0,1) not unitary"

    def test_MS_gate_unitary_12(self):
        for phi, theta in [(0.0, 0.5), (1.2, np.pi/4), (0.3, 1.1)]:
            U = np.array(MS_gate(phi, theta, 1, 2))
            assert np.allclose(U @ U.conj().T, np.eye(9), atol=1e-12), \
                f"MS_gate({phi},{theta},1,2) not unitary"

    def test_MS_gate_identity_at_zero(self):
        U = np.array(MS_gate(0.0, 0.0, 0, 1))
        assert np.allclose(U, np.eye(9), atol=1e-12)


class TestQutritErrors:
    def test_hardware_basis_count(self):
        assert len(hardware_error_basis()) == 4

    def test_extended_basis_count(self):
        assert len(extended_hardware_error_basis()) == 6

    def test_hardware_errors_unitary(self):
        for E in hardware_error_basis():
            assert np.allclose(E @ E.conj().T, np.eye(3), atol=1e-12), \
                "Hardware error should be unitary"

    def test_extended_errors_unitary(self):
        for E in extended_hardware_error_basis():
            assert np.allclose(E @ E.conj().T, np.eye(3), atol=1e-12), \
                "Extended error should be unitary"

    def test_error_set_sizes_d2(self):
        """((3,3,2))_3: 1 identity + 3 qutrits * 4 errors = 13 E_det."""
        E_det, E_corr = build_qutrit_error_set(3, 2)
        assert len(E_det) == 13, f"Expected 13, got {len(E_det)}"
        assert len(E_corr) == 1, f"Expected 1 (identity only), got {len(E_corr)}"

    def test_error_set_identity_first(self):
        E_det, _ = build_qutrit_error_set(3, 2)
        assert np.allclose(E_det[0], np.eye(27), atol=1e-12)

    def test_error_set_shapes(self):
        E_det, _ = build_qutrit_error_set(3, 2)
        for E in E_det:
            assert E.shape == (27, 27), f"Expected 27x27, got {E.shape}"

    def test_error_operators_unitary(self):
        E_det, _ = build_qutrit_error_set(3, 2)
        for E in E_det:
            assert np.allclose(E @ E.conj().T, np.eye(27), atol=1e-12)

    def test_extended_error_set_sizes_d2(self):
        E_det, _ = build_qutrit_error_set(3, 2, extended=True)
        # 1 + 3*6 = 19
        assert len(E_det) == 19, f"Expected 19, got {len(E_det)}"


class TestQutritEncoder:
    def test_encoder_returns_normalized_states(self):
        encoder, connections, ppl = create_native_encoder(3, 3)
        params = np.random.uniform(0, 2*np.pi, (1, ppl))
        for k in range(3):
            state = encoder(params, k)
            norm = np.sum(np.abs(np.array(state))**2)
            assert abs(norm - 1.0) < 1e-10, f"State {k} norm = {norm}"

    def test_encoder_state_dimension(self):
        encoder, _, ppl = create_native_encoder(3, 3)
        params = np.random.uniform(0, 2*np.pi, (1, ppl))
        state = encoder(params, 0)
        assert len(state) == 27, f"Expected 27-dim, got {len(state)}"

    def test_codewords_orthogonal_at_init(self):
        """Initial computational basis states should be orthogonal."""
        encoder, _, ppl = create_native_encoder(3, 3)
        # Zero params -> identity circuit, states should be |000>, |100>, |200>
        params = np.zeros((1, ppl))
        states = [np.array(encoder(params, k)) for k in range(3)]
        for i in range(3):
            for j in range(i+1, 3):
                overlap = abs(np.dot(states[i].conj(), states[j]))
                assert overlap < 1e-10, f"States {i},{j} overlap = {overlap}"

    def test_params_per_layer(self):
        _, connections, ppl = create_native_encoder(3, 3)
        # 3*8 + 3*4 + 3*2 = 24 + 12 + 6 = 42
        assert ppl == 42, f"Expected 42, got {ppl}"

    def test_loss_finite_positive(self):
        """Loss function returns a finite positive number."""
        from src.loss import kl_loss_fast
        encoder, _, ppl = create_native_encoder(3, 3)
        E_det, _ = build_qutrit_error_set(3, 2)
        params = np.random.uniform(0, 2*np.pi, (1, ppl))
        loss = kl_loss_fast(params, encoder, E_det, [], 3, 2)
        loss_val = float(loss)
        assert np.isfinite(loss_val), f"Loss not finite: {loss_val}"
        assert loss_val >= 0, f"Loss negative: {loss_val}"
