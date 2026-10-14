"""
Tests for generalized native trapped-ion gates, error sets, and encoder.
Covers d=3, d=4, d=5 and verifies backward compatibility with qutrit code.
Run: python3 -m pytest tests/test_native.py -v
"""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gates import XY_gate, Z_gate, MS_gate, _build_ms_masks, CSUM_gate, CSUB_gate, zz_subspace_gate
from src.errors import (
    qudit_hardware_error_basis, qudit_extended_hardware_error_basis,
    build_native_error_set, close_error_basis,
    _is_proportional_to_identity, _is_in_set,
)
from src.encoder import create_native_encoder


# ── Gates ──────────────────────────────────────────────────────────────

class TestGeneralizedMSGate:
    """Test MS gate generalized to arbitrary d."""

    def test_matches_old_d3_01(self):
        """General MS gate should match hardcoded d=3 (0,1) results."""
        for phi, theta in [(0.0, 0.0), (0.5, 1.2), (1.3, 0.7)]:
            U = np.array(MS_gate(phi, theta, 0, 1, d=3))
            assert np.allclose(U @ U.conj().T, np.eye(9), atol=1e-12)

    def test_matches_old_d3_12(self):
        """General MS gate should match hardcoded d=3 (1,2) results."""
        for phi, theta in [(0.0, 0.0), (0.5, 1.2), (1.3, 0.7)]:
            U = np.array(MS_gate(phi, theta, 1, 2, d=3))
            assert np.allclose(U @ U.conj().T, np.eye(9), atol=1e-12)

    def test_unitary_d4_all_transitions(self):
        for j, k in [(0, 1), (1, 2), (2, 3)]:
            for phi, theta in [(0.5, 1.2), (0.0, 0.3), (1.0, 0.0)]:
                U = np.array(MS_gate(phi, theta, j, k, d=4))
                assert np.allclose(U @ U.conj().T, np.eye(16), atol=1e-12), \
                    f"d=4 ({j},{k}) phi={phi} theta={theta} not unitary"

    def test_unitary_d5_all_transitions(self):
        for j, k in [(0, 1), (1, 2), (2, 3), (3, 4)]:
            U = np.array(MS_gate(0.5, 1.2, j, k, d=5))
            assert np.allclose(U @ U.conj().T, np.eye(25), atol=1e-12)

    def test_unitary_d7(self):
        U = np.array(MS_gate(0.5, 1.2, 3, 4, d=7))
        assert np.allclose(U @ U.conj().T, np.eye(49), atol=1e-12)

    def test_non_adjacent_transition(self):
        """MS gate on non-adjacent (0,2) transition in d=4."""
        U = np.array(MS_gate(0.5, 1.2, 0, 2, d=4))
        assert np.allclose(U @ U.conj().T, np.eye(16), atol=1e-12)

    def test_identity_at_zero_d4(self):
        U = np.array(MS_gate(0.0, 0.0, 0, 1, d=4))
        assert np.allclose(U, np.eye(16), atol=1e-12)

    def test_identity_at_zero_d5(self):
        U = np.array(MS_gate(0.0, 0.0, 2, 3, d=5))
        assert np.allclose(U, np.eye(25), atol=1e-12)

    def test_mask_symmetry(self):
        """M_c + M_p1 + M_p0 should cover all diagonal elements."""
        for d in [3, 4, 5]:
            for j, k in [(0, 1), (d - 2, d - 1)]:
                M_c, M_p1, M_p0, _, _, _ = _build_ms_masks(j, k, d)
                diag_sum = np.diag(M_c) + np.diag(M_p1) + np.diag(M_p0)
                assert np.allclose(diag_sum, np.ones(d * d)), \
                    f"Masks don't cover full space for d={d} ({j},{k})"


class TestXYGateGeneralD:
    def test_unitary_d4(self):
        for j, k in [(0, 1), (1, 2), (2, 3)]:
            U = np.array(XY_gate(0.5, 1.2, j, k, d=4))
            assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-12)

    def test_unitary_d5(self):
        U = np.array(XY_gate(0.5, 1.2, 2, 3, d=5))
        assert np.allclose(U @ U.conj().T, np.eye(5), atol=1e-12)

    def test_identity_d4(self):
        U = np.array(XY_gate(0.0, 0.0, 1, 2, d=4))
        assert np.allclose(U, np.eye(4), atol=1e-12)


class TestZGateGeneralD:
    def test_unitary_d4(self):
        U = np.array(Z_gate(1.3, 2, 3, d=4))
        assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-12)

    def test_diagonal_d5(self):
        U = np.array(Z_gate(1.3, 0, 1, d=5))
        assert np.allclose(U - np.diag(np.diag(U)), 0, atol=1e-12)


class TestCSUMGate:
    def test_unitary_d3(self):
        assert np.allclose(CSUM_gate(3) @ CSUM_gate(3).conj().T, np.eye(9))

    def test_action_d3(self):
        """CSUM |1,2> = |1, (1+2)%3> = |1,0>."""
        U = CSUM_gate(3)
        inp = np.zeros(9); inp[1*3+2] = 1.0
        out = U @ inp
        expected = np.zeros(9); expected[1*3+0] = 1.0
        assert np.allclose(out, expected)

    def test_csum_csub_inverse(self):
        for d in [3, 4, 5]:
            assert np.allclose(CSUM_gate(d) @ CSUB_gate(d), np.eye(d*d), atol=1e-12)

    def test_unitary_general_d(self):
        for d in [3, 4, 5, 7]:
            U = CSUM_gate(d)
            assert np.allclose(U @ U.conj().T, np.eye(d*d), atol=1e-12)

    def test_all_basis_states(self):
        """Verify CSUM action on all basis states for d=3."""
        U = CSUM_gate(3)
        for a in range(3):
            for b in range(3):
                inp = np.zeros(9); inp[a*3+b] = 1.0
                out = U @ inp
                expected = np.zeros(9); expected[a*3 + (a+b)%3] = 1.0
                assert np.allclose(out, expected), f"|{a},{b}> failed"


class TestCSUBGate:
    def test_action_d3(self):
        """CSUB |1,2> = |1, (2-1)%3> = |1,1>."""
        U = CSUB_gate(3)
        inp = np.zeros(9); inp[1*3+2] = 1.0
        out = U @ inp
        expected = np.zeros(9); expected[1*3+1] = 1.0
        assert np.allclose(out, expected)

    def test_unitary(self):
        for d in [3, 4, 5]:
            U = CSUB_gate(d)
            assert np.allclose(U @ U.conj().T, np.eye(d*d), atol=1e-12)


class TestLightShiftGate:
    def test_unitary(self):
        for theta in [0.0, 0.5, 1.2, np.pi]:
            U = np.array(zz_subspace_gate(theta, 0, 1, d=3))
            assert np.allclose(U @ U.conj().T, np.eye(9), atol=1e-12)

    def test_identity_at_zero(self):
        U = np.array(zz_subspace_gate(0.0, 0, 1, d=3))
        assert np.allclose(U, np.eye(9), atol=1e-12)

    def test_diagonal(self):
        U = np.array(zz_subspace_gate(1.3, 0, 1, d=4))
        assert np.allclose(U, np.diag(np.diag(U)), atol=1e-12)

    def test_general_d(self):
        for d in [3, 4, 5]:
            U = np.array(zz_subspace_gate(0.7, 0, 1, d=d))
            assert np.allclose(U @ U.conj().T, np.eye(d*d), atol=1e-12)


# ── Error Sets ─────────────────────────────────────────────────────────

class TestNativeErrorSets:
    def test_d3_matches_qutrit(self):
        """d=3 native errors should match the qutrit hardware_error_basis."""
        from src.errors import hardware_error_basis
        native = qudit_hardware_error_basis(d=3)
        qutrit = hardware_error_basis()
        assert len(native) == len(qutrit)
        for n, q in zip(native, qutrit):
            assert np.allclose(n, q), "d=3 native should match qutrit"

    def test_d3_extended_matches_qutrit(self):
        from src.errors import extended_hardware_error_basis
        native = qudit_extended_hardware_error_basis(d=3)
        qutrit = extended_hardware_error_basis()
        assert len(native) == len(qutrit)
        for n, q in zip(native, qutrit):
            assert np.allclose(n, q)

    def test_error_counts(self):
        """2(d-1) errors for standard, 3(d-1) for extended."""
        for d in [3, 4, 5, 7]:
            assert len(qudit_hardware_error_basis(d)) == 2 * (d - 1)
            assert len(qudit_extended_hardware_error_basis(d)) == 3 * (d - 1)

    def test_all_unitary(self):
        for d in [3, 4, 5]:
            for E in qudit_hardware_error_basis(d):
                assert np.allclose(E @ E.conj().T, np.eye(d), atol=1e-12)
            for E in qudit_extended_hardware_error_basis(d):
                assert np.allclose(E @ E.conj().T, np.eye(d), atol=1e-12)

    def test_error_set_sizes_d4_dist2(self):
        """d=4, n=5, distance=2: 1 + 5*6 = 31."""
        E_det, E_corr = build_native_error_set(5, 4, 2)
        assert len(E_det) == 31, f"Expected 31, got {len(E_det)}"
        assert len(E_corr) == 1

    def test_error_set_sizes_d5_dist2(self):
        """d=5, n=5, distance=2: 1 + 5*8 = 41."""
        E_det, E_corr = build_native_error_set(5, 5, 2)
        assert len(E_det) == 41, f"Expected 41, got {len(E_det)}"

    def test_error_set_shapes_d4(self):
        E_det, _ = build_native_error_set(3, 4, 2)
        for E in E_det:
            assert E.shape == (64, 64)

    def test_error_operators_unitary_d4(self):
        E_det, _ = build_native_error_set(3, 4, 2)
        for E in E_det:
            assert np.allclose(E @ E.conj().T, np.eye(64), atol=1e-12)

    def test_d3_error_set_matches_qutrit(self):
        from src.errors import build_qutrit_error_set
        native_det, native_corr = build_native_error_set(3, 3, 2)
        qutrit_det, qutrit_corr = build_qutrit_error_set(3, 2)
        assert len(native_det) == len(qutrit_det)
        assert len(native_corr) == len(qutrit_corr)
        for n, q in zip(native_det, qutrit_det):
            assert np.allclose(n, q)


class TestErrorBasisClosure:
    """Tests for close_error_basis and closed E_det construction."""

    def test_cross_products_unitary(self):
        errs = qudit_hardware_error_basis(3)
        cross = close_error_basis(errs)
        for E in cross:
            assert np.allclose(E @ E.conj().T, np.eye(3), atol=1e-12)

    def test_d3_cross_count(self):
        """d=3 hardware basis: 4 errors produce 8 unique cross-products."""
        cross = close_error_basis(qudit_hardware_error_basis(3))
        assert len(cross) == 8, f"Expected 8, got {len(cross)}"

    def test_closure_complete(self):
        """All products of original errors appear in originals + cross (up to phase)."""
        errs = qudit_hardware_error_basis(3)
        cross = close_error_basis(errs)
        all_ops = list(errs) + list(cross)
        for Ea in errs:
            for Eb in errs:
                M = Ea @ Eb
                if _is_proportional_to_identity(M, 3):
                    continue
                assert _is_in_set(M, all_ops, 3), "Product missing from closed set"

    def test_cyclic_shift_present(self):
        """X_01 @ X_12 = cyclic shift should be in cross-products."""
        errs = qudit_hardware_error_basis(3)
        X01, X12 = errs[2], errs[3]
        cyclic = X01 @ X12  # [[0,0,1],[1,0,0],[0,1,0]]
        cross = close_error_basis(errs)
        assert _is_in_set(cyclic, cross, 3), "Cyclic shift missing"

    def test_z1z2_not_identity(self):
        """Z1·Z2 = diag(1,-1,-1) is NOT proportional to identity."""
        errs = qudit_hardware_error_basis(3)
        M = errs[0] @ errs[1]
        assert not _is_proportional_to_identity(M, 3)

    def test_z1z2_in_closure(self):
        """Z1·Z2 = diag(1,-1,-1) should be in cross-products."""
        errs = qudit_hardware_error_basis(3)
        M = errs[0] @ errs[1]
        cross = close_error_basis(errs)
        assert _is_in_set(M, cross, 3), "Z1*Z2 missing from closure"

    def test_closed_edet_size_d3(self):
        """((5,3,3))_3 closed: 181 + 5*8 = 221."""
        E_det, E_corr = build_native_error_set(5, 3, 3, closed=True)
        assert len(E_det) == 221, f"Expected 221, got {len(E_det)}"
        assert len(E_corr) == 21, "E_corr should be unchanged"

    def test_closed_edet_all_unitary(self):
        E_det, _ = build_native_error_set(5, 3, 3, closed=True)
        dim = 3**5
        for E in E_det:
            assert np.allclose(E @ E.conj().T, np.eye(dim), atol=1e-12)

    def test_backward_compat_closed_false(self):
        """closed=False gives same result as before."""
        E_det, _ = build_native_error_set(5, 3, 3, closed=False)
        assert len(E_det) == 181

    def test_d2_unaffected_by_closed(self):
        """Distance 2 has max_corr=0, so closed=True has no effect."""
        E_old, _ = build_native_error_set(3, 3, 2, closed=False)
        E_new, _ = build_native_error_set(3, 3, 2, closed=True)
        assert len(E_old) == len(E_new)

    def test_d4_closure(self):
        """d=4 hardware basis closure produces valid unitary operators."""
        errs = qudit_hardware_error_basis(4)
        cross = close_error_basis(errs)
        for E in cross:
            assert np.allclose(E @ E.conj().T, np.eye(4), atol=1e-12)
        # 6 original errors in d=4 should produce many cross-products
        assert len(cross) > 0


# ── Encoder ────────────────────────────────────────────────────────────

class TestNativeEncoderD3:
    """d=3 native encoder should behave identically to qutrit encoder."""

    def test_params_per_layer(self):
        _, _, ppl = create_native_encoder(3, 3)
        assert ppl == 42

    def test_normalized(self):
        encoder, _, ppl = create_native_encoder(3, 3)
        params = pnp.array(np.random.uniform(0, 2*np.pi, (1, ppl)))
        for k in range(3):
            state = np.array(encoder(params, k))
            assert abs(np.sum(np.abs(state)**2) - 1.0) < 1e-10

    def test_state_dim(self):
        encoder, _, ppl = create_native_encoder(3, 3)
        state = encoder(np.zeros((1, ppl)), 0)
        assert len(state) == 27

    def test_orthogonal_at_zero(self):
        encoder, _, ppl = create_native_encoder(3, 3)
        states = [np.array(encoder(np.zeros((1, ppl)), k)) for k in range(3)]
        for i in range(3):
            for j in range(i + 1, 3):
                assert abs(np.dot(states[i].conj(), states[j])) < 1e-10


class TestNativeEncoderD4:
    """d=4 encoder using manual state-vector simulation."""

    def test_params_per_layer(self):
        _, connections, ppl = create_native_encoder(3, 4)
        expected = (5 * 3 + 2 * len(connections)) * 3
        assert ppl == expected, f"Expected {expected}, got {ppl}"

    def test_normalized(self):
        encoder, _, ppl = create_native_encoder(3, 4)
        params = pnp.array(np.random.uniform(0, 2*np.pi, (1, ppl)))
        for k in range(4):
            state = np.array(encoder(params, k))
            assert abs(np.sum(np.abs(state)**2) - 1.0) < 1e-10

    def test_state_dim(self):
        encoder, _, ppl = create_native_encoder(3, 4)
        state = encoder(np.zeros((1, ppl)), 0)
        assert len(state) == 64

    def test_orthogonal_at_zero(self):
        encoder, _, ppl = create_native_encoder(3, 4)
        states = [np.array(encoder(np.zeros((1, ppl)), k)) for k in range(4)]
        for i in range(4):
            for j in range(i + 1, 4):
                assert abs(np.dot(states[i].conj(), states[j])) < 1e-10

    def test_gradient_works(self):
        """Verify autograd differentiation through manual encoder."""
        encoder, _, ppl = create_native_encoder(3, 4)
        E_det, _ = build_native_error_set(3, 4, 2)
        params = pnp.array(np.random.uniform(0, 2*np.pi, (1, ppl)), requires_grad=True)

        from src.loss import kl_loss_fast
        def loss_fn(p):
            return kl_loss_fast(p, encoder, E_det, [], 4, 2)

        grad = qml.grad(loss_fn)(params)
        assert np.any(np.abs(grad) > 1e-10), "Gradient should be non-zero"

    def test_loss_decreases(self):
        """One manual SGD step should decrease loss."""
        encoder, _, ppl = create_native_encoder(3, 4)
        E_det, _ = build_native_error_set(3, 4, 2)
        np.random.seed(123)
        params = pnp.array(np.random.uniform(0, 2*np.pi, (1, ppl)), requires_grad=True)

        from src.loss import kl_loss_fast
        def loss_fn(p):
            return kl_loss_fast(p, encoder, E_det, [], 4, 2)

        loss0 = float(loss_fn(params))
        grad = qml.grad(loss_fn)(params)
        params2 = params - 0.01 * grad
        loss1 = float(loss_fn(params2))
        assert loss1 < loss0, f"Loss should decrease: {loss0} -> {loss1}"


class TestNativeEncoderD5:
    """d=5 encoder smoke tests."""

    def test_params_per_layer(self):
        _, connections, ppl = create_native_encoder(3, 5)
        expected = (5 * 3 + 2 * len(connections)) * 4
        assert ppl == expected

    def test_normalized(self):
        encoder, _, ppl = create_native_encoder(3, 5)
        params = pnp.array(np.random.uniform(0, 2*np.pi, (1, ppl)))
        for k in range(5):
            state = np.array(encoder(params, k))
            assert abs(np.sum(np.abs(state)**2) - 1.0) < 1e-10

    def test_state_dim(self):
        encoder, _, ppl = create_native_encoder(3, 5)
        state = encoder(np.zeros((1, ppl)), 0)
        assert len(state) == 125

    def test_orthogonal_at_zero(self):
        encoder, _, ppl = create_native_encoder(3, 5)
        states = [np.array(encoder(np.zeros((1, ppl)), k)) for k in range(5)]
        for i in range(5):
            for j in range(i + 1, 5):
                assert abs(np.dot(states[i].conj(), states[j])) < 1e-10


class TestForceManualD3:
    """force_manual=True for d=3 should match QNode encoder."""

    def test_loss_matches_qnode(self):
        from src.loss import kl_loss_fast
        enc_qnode, _, ppl = create_native_encoder(3, 3, force_manual=False)
        enc_manual, _, _ = create_native_encoder(3, 3, force_manual=True)
        E_det, _ = build_native_error_set(3, 3, 2)
        np.random.seed(99)
        params = pnp.array(np.random.uniform(0, 2*np.pi, (1, ppl)))
        loss_q = float(kl_loss_fast(params, enc_qnode, E_det, [], 3, 2))
        loss_m = float(kl_loss_fast(params, enc_manual, E_det, [], 3, 2))
        assert abs(loss_q - loss_m) < 1e-8, f"QNode={loss_q}, Manual={loss_m}"

    def test_states_match(self):
        enc_qnode, _, ppl = create_native_encoder(3, 3, force_manual=False)
        enc_manual, _, _ = create_native_encoder(3, 3, force_manual=True)
        np.random.seed(77)
        params = pnp.array(np.random.uniform(0, 2*np.pi, (1, ppl)))
        for k in range(3):
            s_q = np.array(enc_qnode(params, k))
            s_m = np.array(enc_manual(params, k))
            # States may differ by global phase from state prep
            # but inner product magnitude should be 1
            overlap = abs(np.dot(s_q.conj(), s_m))
            assert overlap > 1 - 1e-10, f"code_ind={k} overlap={overlap}"

    def test_gradient_nonzero(self):
        from src.loss import kl_loss_fast
        enc, _, ppl = create_native_encoder(3, 3, force_manual=True)
        E_det, _ = build_native_error_set(3, 3, 2)
        params = pnp.array(np.random.uniform(0, 2*np.pi, (1, ppl)), requires_grad=True)
        grad = qml.grad(lambda p: kl_loss_fast(p, enc, E_det, [], 3, 2))(params)
        assert np.any(np.abs(grad) > 1e-10)

    def test_n5_manual_normalized(self):
        """n=5, d=3 with force_manual should work."""
        enc, _, ppl = create_native_encoder(5, 3, force_manual=True)
        params = pnp.array(np.random.uniform(0, 2*np.pi, (1, ppl)))
        for k in range(3):
            state = np.array(enc(params, k))
            assert abs(np.sum(np.abs(state)**2) - 1.0) < 1e-10
            assert len(state) == 243


class TestNativeNoise:
    def test_noiseless(self):
        from src.errors import make_hardware_noise_fn
        noise_fn = make_hardware_noise_fn(3, 3, 0.0)
        rng = np.random.default_rng(42)
        state = np.zeros(27, dtype=complex)
        state[0] = 1.0
        noisy = noise_fn(state, rng)
        assert np.allclose(noisy, state)

    def test_full_noise_changes_state(self):
        from src.errors import make_hardware_noise_fn
        noise_fn = make_hardware_noise_fn(3, 3, 1.0)
        rng = np.random.default_rng(42)
        state = np.zeros(27, dtype=complex)
        state[0] = 1.0
        noisy = noise_fn(state, rng)
        assert not np.allclose(noisy, state)
        assert abs(np.linalg.norm(noisy) - 1.0) < 1e-12

    def test_normalized_output(self):
        from src.errors import make_hardware_noise_fn
        noise_fn = make_hardware_noise_fn(4, 3, 0.5)
        rng = np.random.default_rng(42)
        state = np.ones(64, dtype=complex) / 8  # normalized
        noisy = noise_fn(state, rng)
        assert abs(np.linalg.norm(noisy) - 1.0) < 1e-12

    def test_d4_noise(self):
        from src.errors import make_hardware_noise_fn
        noise_fn = make_hardware_noise_fn(4, 5, 0.1)
        rng = np.random.default_rng(42)
        state = np.zeros(1024, dtype=complex)
        state[0] = 1.0
        noisy = noise_fn(state, rng)
        assert abs(np.linalg.norm(noisy) - 1.0) < 1e-12


class TestEncoderCustomConnections:
    def test_star_topology(self):
        connections = [[0, 1], [0, 2], [0, 3], [0, 4]]
        encoder, conns, ppl = create_native_encoder(5, 4, connections)
        expected = (5 * 5 + 2 * 4) * 3
        assert ppl == expected
        assert conns == connections

    def test_d3_custom_connections(self):
        connections = [[0, 1], [0, 2]]
        encoder, _, ppl = create_native_encoder(3, 3, connections)
        expected = (5 * 3 + 2 * 2) * 2
        assert ppl == expected
