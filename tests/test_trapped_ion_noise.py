"""
Tests for the trapped-ion correlated noise model.
Run with: python3 -m pytest tests/test_trapped_ion_noise.py -v
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.correlated_noise import (
    control_qudit_kraus, target_qudit_kraus, spectator_qudit_kraus,
    verify_kraus_completeness, verify_matrix_kraus_completeness,
    _kraus_diagonal, subspace_depolarizing_kraus, combined_gate_kraus
)


class TestKrausCompleteness:
    """sum_n E_n^dag E_n ~ I for all qudit roles."""

    def test_control_d3(self):
        diags = control_qudit_kraus(d=3, control_level=0, eta=0.95, n_max=10)
        ok, err = verify_kraus_completeness(diags)
        assert ok, f"Control d=3 completeness failed: max deviation {err:.2e}"

    def test_target_d3(self):
        diags = target_qudit_kraus(d=3, control_level=0, target_level=1, eta=0.95, n_max=15)
        ok, err = verify_kraus_completeness(diags)
        assert ok, f"Target d=3 completeness failed: max deviation {err:.2e}"

    def test_spectator_d3(self):
        diags = spectator_qudit_kraus(d=3, eta=0.95, n_max=10)
        ok, err = verify_kraus_completeness(diags)
        assert ok, f"Spectator d=3 completeness failed: max deviation {err:.2e}"

    def test_control_d5(self):
        diags = control_qudit_kraus(d=5, control_level=0, eta=0.95, n_max=15)
        ok, err = verify_kraus_completeness(diags)
        assert ok, f"Control d=5 completeness failed: max deviation {err:.2e}"

    def test_target_d5(self):
        diags = target_qudit_kraus(d=5, control_level=0, target_level=1, eta=0.95, n_max=15)
        ok, err = verify_kraus_completeness(diags)
        assert ok, f"Target d=5 completeness failed: max deviation {err:.2e}"

    def test_completeness_strong_noise(self):
        """Completeness should hold even with strong noise (eta=0.5), needs more terms."""
        diags = control_qudit_kraus(d=3, control_level=0, eta=0.5, n_max=30)
        ok, err = verify_kraus_completeness(diags)
        assert ok, f"Strong noise completeness failed: max deviation {err:.2e}"


class TestNoiselessLimit:
    """When eta=1 (no noise), E_0 = I and all E_{n>0} = 0."""

    def test_control_noiseless(self):
        diags = control_qudit_kraus(d=3, control_level=0, eta=1.0, n_max=5)
        assert np.allclose(diags[0], np.ones(3)), "E_0 should be identity when eta=1"
        for n in range(1, len(diags)):
            assert np.allclose(diags[n], 0), f"E_{n} should be zero when eta=1"

    def test_target_noiseless(self):
        diags = target_qudit_kraus(d=3, control_level=0, target_level=1, eta=1.0, n_max=5)
        assert np.allclose(diags[0], np.ones(3)), "E_0 should be identity when eta=1"

    def test_spectator_noiseless(self):
        diags = spectator_qudit_kraus(d=3, eta=1.0, n_max=5)
        assert np.allclose(diags[0], np.ones(3)), "E_0 should be identity when eta=1"


class TestPhysics:
    """Verify physical properties of the noise model."""

    def test_control_level_unaffected(self):
        """The control level should have coupling=0, so E_0[i]=1, E_{n>0}[i]=0."""
        for ctrl in range(3):
            diags = control_qudit_kraus(d=3, control_level=ctrl, eta=0.9, n_max=5)
            assert abs(diags[0][ctrl] - 1.0) < 1e-14, f"E_0[{ctrl}] should be 1"
            for n in range(1, len(diags)):
                assert abs(diags[n][ctrl]) < 1e-14, f"E_{n}[{ctrl}] should be 0"

    def test_target_three_tiers(self):
        """Target qudit should have three distinct coupling strengths: 1, 2, 3."""
        diags = target_qudit_kraus(d=5, control_level=0, target_level=1, eta=0.9, n_max=5)
        # E_1 diagonal should show three distinct magnitudes
        e1 = np.abs(diags[1])
        # Level 0 (ctrl): coupling=1, Level 1 (tgt): coupling=2, Levels 2,3,4: coupling=3
        assert not np.isclose(e1[0], e1[1]), "Control and target should have different couplings"
        assert not np.isclose(e1[1], e1[2]), "Target and spectator should have different couplings"
        assert np.isclose(e1[2], e1[3]) and np.isclose(e1[3], e1[4]), "All spectator levels equal"

    def test_all_diagonal(self):
        """All Kraus operators should be diagonal (stored as 1D vectors by construction)."""
        diags = control_qudit_kraus(d=3, control_level=0, eta=0.9, n_max=3)
        for d_vec in diags:
            assert d_vec.ndim == 1, "Kraus diagonals should be 1D"
            assert len(d_vec) == 3, "Should have d elements"

    def test_dimensions_d3_d5_d7(self):
        """Check correct dimensions for d=3,5,7."""
        for d in [3, 5, 7]:
            diags = spectator_qudit_kraus(d=d, eta=0.95, n_max=3)
            assert len(diags) == 4, f"Should have n_max+1=4 operators for d={d}"
            for vec in diags:
                assert len(vec) == d, f"Each diagonal should have {d} elements"


class TestErrorProducts:
    """Test the error set construction."""

    def test_products_diagonal(self):
        """E_alpha^dag E_beta should be diagonal when E_alpha, E_beta are diagonal."""
        diags = control_qudit_kraus(d=3, control_level=0, eta=0.9, n_max=3)
        for da in diags:
            for db in diags:
                prod = np.conj(da) * db
                assert prod.ndim == 1, "Product of diagonals should be 1D"

    def test_error_set_small_system(self):
        """Build error set for n=2, d=3 and check it's manageable."""
        from src.correlated_noise import build_correlated_error_set
        gate_pairs = [(0, 1, 0, 1)]  # one gate between qudit 0 and 1, on levels 0,1
        E_det, E_corr = build_correlated_error_set(
            n_qudits=2, d=3, gate_pairs=gate_pairs, eta=0.95, n_max=2
        )
        print(f"Error set size for n=2, d=3: {len(E_det)}")
        assert len(E_det) > 1, "Should have more than just identity"
        assert len(E_det) < 1000, "Should be manageable for small system"
        # All should be 1D vectors of length d^n = 9
        for e in E_det:
            assert len(e) == 9, f"Expected length 9, got {len(e)}"


class TestSubspaceDepolarizing:
    """Tests for the subspace depolarizing noise."""

    def test_completeness(self):
        """Sum K_k^dag K_k = I."""
        for d in [3, 5, 7]:
            kraus = subspace_depolarizing_kraus(d, 0, 1, p=0.1)
            ok, err = verify_matrix_kraus_completeness(kraus)
            assert ok, f"Completeness failed for d={d}: deviation {err:.2e}"

    def test_no_noise(self):
        """When p=0, only K_0 = I survives."""
        kraus = subspace_depolarizing_kraus(d=5, level_i=0, level_j=1, p=0.0)
        assert np.allclose(kraus[0], np.eye(5)), "K_0 should be identity when p=0"
        for k in range(1, 4):
            assert np.allclose(kraus[k], 0), f"K_{k} should be zero when p=0"

    def test_subspace_action(self):
        """Error operators should only affect levels i and j."""
        d = 5
        kraus = subspace_depolarizing_kraus(d, level_i=1, level_j=3, p=0.5)
        # K_1 (X in subspace) should map |1> -> |3> and |3> -> |1>
        state_1 = np.zeros(d); state_1[1] = 1.0
        result = kraus[1] @ state_1
        expected = np.zeros(d); expected[3] = np.sqrt(0.5 / 3)
        assert np.allclose(result, expected), "X_{13} should swap levels 1<->3"

    def test_not_diagonal(self):
        """Subspace depolarizing Kraus ops K_1,K_2 should NOT be diagonal."""
        kraus = subspace_depolarizing_kraus(d=3, level_i=0, level_j=1, p=0.1)
        assert not np.allclose(kraus[1], np.diag(np.diag(kraus[1]))), \
            "K_1 should have off-diagonal elements"

    def test_different_level_pairs(self):
        """Test with various level pairs to ensure generality."""
        d = 5
        for i, j in [(0, 1), (0, 4), (2, 3), (1, 4)]:
            kraus = subspace_depolarizing_kraus(d, i, j, p=0.1)
            ok, err = verify_matrix_kraus_completeness(kraus)
            assert ok, f"Failed for levels ({i},{j}): deviation {err:.2e}"

    def test_full_depolarizing(self):
        """When p=1, K_0 projects onto complement, Paulis at full strength."""
        kraus = subspace_depolarizing_kraus(d=3, level_i=0, level_j=1, p=1.0)
        # K_0 should be P_perp (only |2><2|)
        expected_K0 = np.zeros((3, 3), dtype=complex)
        expected_K0[2, 2] = 1.0
        assert np.allclose(kraus[0], expected_K0)


class TestCombinedGateNoise:
    """Tests for the combined dephasing + depolarizing noise model."""

    def test_returns_all_components(self):
        result = combined_gate_kraus(d=5, control_level=0, target_level=1,
                                     eta_dephasing=0.95, p_depolarizing=0.01)
        assert 'control_dephasing' in result
        assert 'target_dephasing' in result
        assert 'subspace_depol' in result
        assert 'spectator_dephasing' in result

    def test_subspace_depol_count(self):
        result = combined_gate_kraus(d=3, control_level=0, target_level=1,
                                     eta_dephasing=0.95, p_depolarizing=0.05)
        assert len(result['subspace_depol']) == 4, "Should have 4 subspace depol operators"

    def test_dephasing_count(self):
        result = combined_gate_kraus(d=3, control_level=0, target_level=1,
                                     eta_dephasing=0.95, p_depolarizing=0.05,
                                     n_max_dephasing=3)
        assert len(result['control_dephasing']) == 4
        assert len(result['target_dephasing']) == 4
        assert len(result['spectator_dephasing']) == 4


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
