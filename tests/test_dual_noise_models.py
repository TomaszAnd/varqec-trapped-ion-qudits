"""
Tests for dual noise models: "simplified" (Eq. J3 literal) vs "physical" (f_k=k).

Both models are from arXiv:2310.12110v3, Appendix J.
The key difference is ONLY in the control qudit coupling:
- Simplified: f=0 for control, f=2 for all others (uniform)
- Physical: f=0 for control, f=k for level k (level-dependent)
Target qudit (f=1/2/3 tiers) and spectator (f_k=k) are identical in both.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.correlated_noise import (
    control_qudit_kraus, control_qudit_kraus_simplified,
    target_qudit_kraus, spectator_qudit_kraus,
    combined_gate_kraus, verify_kraus_completeness,
    _kraus_diagonal
)
from src.correlated_noise import build_correlated_error_set


class TestSimplifiedCoupling:
    """Verify simplified model has uniform f=2 for all k != control."""

    def test_coupling_uniform_f2(self):
        """Simplified: all non-control levels get f=2."""
        d = 5
        eta = 0.9
        diags_s = control_qudit_kraus_simplified(d, control_level=0, eta=eta, n_max=5)
        # Gaussian-averaged: E_0[k] = eta^{f_k^2/2}. For f=2: eta^2. For f=0: eta^0 = 1.
        e0 = np.real(diags_s[0])
        assert abs(e0[0] - 1.0) < 1e-14, "Control level should be 1"
        for k in range(1, d):
            assert abs(e0[k] - eta**2) < 1e-10, f"Level {k}: expected eta^2={eta**2}, got {e0[k]}"

    def test_all_non_control_equal(self):
        """Simplified: all non-control levels should be identical."""
        d = 7
        diags = control_qudit_kraus_simplified(d, control_level=2, eta=0.95, n_max=5)
        for n in range(len(diags)):
            non_ctrl = [diags[n][k] for k in range(d) if k != 2]
            for v in non_ctrl:
                assert abs(v - non_ctrl[0]) < 1e-14, \
                    f"E_{n}: all non-control levels should be identical"


class TestPhysicalCoupling:
    """Verify physical model has f_k = k."""

    def test_coupling_level_dependent(self):
        """Physical: level k gets coupling f_k = k."""
        d = 5
        eta = 0.9
        diags_p = control_qudit_kraus(d, control_level=0, eta=eta, n_max=5)
        e0 = np.real(diags_p[0])
        assert abs(e0[0] - 1.0) < 1e-14, "Control level should be 1"
        for k in range(1, d):
            expected = eta ** (k**2 / 2)  # Gaussian-averaged: eta^{f_k^2/2}
            assert abs(e0[k] - expected) < 1e-10, \
                f"Level {k}: expected eta^{k**2 / 2}={expected}, got {e0[k]}"


class TestBothModelsCompleteness:
    """Kraus completeness for both models."""

    def test_simplified_completeness(self):
        for d in [3, 5, 7]:
            diags = control_qudit_kraus_simplified(d, control_level=0, eta=0.95, n_max=15)
            ok, dev = verify_kraus_completeness(diags)
            assert ok, f"Simplified d={d} completeness failed: {dev:.2e}"

    def test_physical_completeness(self):
        # Physical model: f_k=k, so d=7 needs n_max>=20 for coupling f=6
        for d, nmax in [(3, 15), (5, 15), (7, 25)]:
            diags = control_qudit_kraus(d, control_level=0, eta=0.95, n_max=nmax)
            ok, dev = verify_kraus_completeness(diags)
            assert ok, f"Physical d={d} completeness failed: {dev:.2e}"


class TestModelsDiffer:
    """Models MUST produce different results for control qudit (except d=2 edge case)."""

    def test_d5_models_differ(self):
        """For d=5: physical f={0,1,2,3,4} vs simplified f={0,2,2,2,2} — must differ."""
        d = 5
        diags_p = control_qudit_kraus(d, 0, eta=0.9, n_max=5)
        diags_s = control_qudit_kraus_simplified(d, 0, eta=0.9, n_max=5)
        # Level 1: physical f=1, simplified f=2 → different
        assert not np.isclose(diags_p[0][1], diags_s[0][1]), \
            "Level 1 should differ: physical f=1 vs simplified f=2"
        # Level 2: physical f=2, simplified f=2 → same
        assert np.isclose(diags_p[0][2], diags_s[0][2]), \
            "Level 2 should be the same: both f=2"
        # Level 3: physical f=3, simplified f=2 → different
        assert not np.isclose(diags_p[0][3], diags_s[0][3]), \
            "Level 3 should differ: physical f=3 vs simplified f=2"

    def test_d3_models_differ(self):
        """For d=3: physical f={0,1,2} vs simplified f={0,2,2} — differ at level 1."""
        d = 3
        diags_p = control_qudit_kraus(d, 0, eta=0.9, n_max=5)
        diags_s = control_qudit_kraus_simplified(d, 0, eta=0.9, n_max=5)
        # Level 1: physical f=1, simplified f=2 → different
        assert not np.isclose(diags_p[0][1], diags_s[0][1])
        # Level 2: both f=2 → same
        assert np.isclose(diags_p[0][2], diags_s[0][2])


class TestTargetQuditIdentical:
    """Target qudit uses f=1/2/3 tiers in BOTH models — verify identical."""

    def test_target_identical_in_both(self):
        """combined_gate_kraus should return identical target dephasing for both models."""
        d = 5
        r_phys = combined_gate_kraus(d, 0, 1, 0.95, 0.05, n_max_dephasing=10,
                                     noise_model="physical")
        r_simp = combined_gate_kraus(d, 0, 1, 0.95, 0.05, n_max_dephasing=10,
                                     noise_model="simplified")
        for n in range(len(r_phys['target_dephasing'])):
            assert np.allclose(r_phys['target_dephasing'][n], r_simp['target_dephasing'][n]), \
                f"Target dephasing E_{n} should be identical in both models"

    def test_spectator_identical_in_both(self):
        """Spectator dephasing should be identical in both models."""
        d = 5
        r_phys = combined_gate_kraus(d, 0, 1, 0.95, 0.05, noise_model="physical")
        r_simp = combined_gate_kraus(d, 0, 1, 0.95, 0.05, noise_model="simplified")
        for n in range(len(r_phys['spectator_dephasing'])):
            assert np.allclose(r_phys['spectator_dephasing'][n],
                              r_simp['spectator_dephasing'][n])

    def test_subspace_depol_identical(self):
        """Subspace depolarizing doesn't depend on dephasing model."""
        d = 5
        r_phys = combined_gate_kraus(d, 0, 1, 0.95, 0.05, noise_model="physical")
        r_simp = combined_gate_kraus(d, 0, 1, 0.95, 0.05, noise_model="simplified")
        for k in range(4):
            assert np.allclose(r_phys['subspace_depol'][k], r_simp['subspace_depol'][k])


class TestControlQuditDiffers:
    """Control qudit Kraus ops MUST differ between models."""

    def test_combined_control_differs(self):
        d = 5
        r_phys = combined_gate_kraus(d, 0, 1, 0.95, 0.05, noise_model="physical")
        r_simp = combined_gate_kraus(d, 0, 1, 0.95, 0.05, noise_model="simplified")
        # E_0 should differ at levels with f_k != 2
        assert not np.allclose(r_phys['control_dephasing'][0],
                              r_simp['control_dephasing'][0])


class TestErrorSetComparison:
    """Compare error set sizes between models."""

    def test_both_models_produce_error_sets(self):
        """Both models should produce valid error sets."""
        gate_pairs = [(0, 1, 0, 1)]
        for model in ["physical", "simplified"]:
            E_det, E_corr = build_correlated_error_set(
                n_qudits=2, d=3, gate_pairs=gate_pairs, eta=0.95,
                n_max=2, noise_model=model
            )
            assert len(E_det) > 1, f"Model {model}: should have errors beyond identity"
            # All should be 1D diagonal vectors
            for e in E_det:
                assert e.ndim == 1 and len(e) == 9

    def test_error_set_sizes_may_differ(self):
        """Different coupling → different products → possibly different set sizes."""
        gate_pairs = [(0, 1, 0, 1)]
        E_p, _ = build_correlated_error_set(
            n_qudits=2, d=5, gate_pairs=gate_pairs, eta=0.95,
            n_max=2, noise_model="physical"
        )
        E_s, _ = build_correlated_error_set(
            n_qudits=2, d=5, gate_pairs=gate_pairs, eta=0.95,
            n_max=2, noise_model="simplified"
        )
        # Both should be non-trivial; sizes may or may not differ
        assert len(E_p) > 1
        assert len(E_s) > 1
        print(f"Error set sizes: physical={len(E_p)}, simplified={len(E_s)}")


class TestCombinedGateNoisModelParam:
    """Test the noise_model parameter validation."""

    def test_invalid_noise_model_raises(self):
        import pytest
        with pytest.raises(ValueError, match="noise_model"):
            combined_gate_kraus(d=3, control_level=0, target_level=1,
                              eta_dephasing=0.95, p_depolarizing=0.05,
                              noise_model="invalid")

    def test_invalid_noise_model_in_error_set(self):
        import pytest
        with pytest.raises(ValueError, match="noise_model"):
            build_correlated_error_set(
                n_qudits=2, d=3, gate_pairs=[(0, 1, 0, 1)],
                eta=0.95, noise_model="invalid"
            )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
