"""
Verify trapped-ion noise model against arXiv:2310.12110v3, Appendix J.

The paper describes TWO things:
1. Eq. J3/J4: Physical phase errors (uniform per tier)
   - Control: level i unaffected, all others get phase 2*Phi
   - Target: level i gets Phi, level j gets 2*Phi, spectators get 3*Phi

2. Kraus operators: Gaussian-averaged phase model (matches Meth ground truth)
   E_n[k,k] = (f_k * sqrt(sigma_p^2))^n * eta^{f_k^2 / 2} / sqrt(n!)
   where f_k is the coupling strength per level. Coherence between levels with
   coupling difference Delta_f decays as e^{-1/2 sigma_p^2 Delta_f^2}.

The coupling strengths determine which physical model we're using:
- f_k = k (level-dependent): higher levels decohere faster (standard phase damping)
- f_k = constant per tier: matches the C-ROT physics (uniform within tier)

Current implementation uses f_k = k for control/spectator qudits (level-dependent)
and f_k = 1/2/3 per tier for target qudits. See docstring in control_qudit_kraus.

Both models are physically motivated. We test both and document the difference.
"""
import numpy as np
from math import factorial, sqrt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.correlated_noise import (
    _kraus_diagonal, control_qudit_kraus, target_qudit_kraus,
    spectator_qudit_kraus, subspace_depolarizing_kraus,
    combined_gate_kraus, verify_kraus_completeness,
    verify_matrix_kraus_completeness
)


class TestControlQuditEqJ3:
    """Test control qudit noise against Eq. J3 of arXiv:2310.12110v3."""

    def test_control_level_unaffected(self):
        """Eq. J3: c_i|i> -> c_i|i> (control level unchanged)."""
        for d in [3, 5]:
            for ctrl in range(d):
                diags = control_qudit_kraus(d, ctrl, eta=0.9, n_max=10)
                # E_0[ctrl] should be 1, E_{n>0}[ctrl] should be 0
                assert abs(diags[0][ctrl] - 1.0) < 1e-14
                for n in range(1, len(diags)):
                    assert abs(diags[n][ctrl]) < 1e-14

    def test_non_control_dephasing(self):
        """Non-control levels should decohere. Check E_0 < 1 for j!=i."""
        diags = control_qudit_kraus(d=5, control_level=0, eta=0.9, n_max=5)
        for k in range(1, 5):
            assert diags[0][k] < 1.0, f"Level {k} should have E_0 < 1"
            assert diags[0][k] > 0.0, f"Level {k} should have E_0 > 0"

    def test_higher_levels_decohere_faster(self):
        """With f_k = k coupling, higher levels should decohere faster."""
        diags = control_qudit_kraus(d=5, control_level=0, eta=0.9, n_max=5)
        # E_0[1] > E_0[2] > E_0[3] > E_0[4] (less probability of staying undisturbed)
        for k in range(1, 4):
            assert diags[0][k] > diags[0][k+1], \
                f"E_0[{k}]={diags[0][k]:.4f} should be > E_0[{k+1}]={diags[0][k+1]:.4f}"

    def test_manual_computation(self):
        """Verify E_n against manual calculation of the Kraus formula."""
        d = 3
        eta = 0.95
        ctrl = 0
        sigma_p2 = -np.log(eta)

        diags = control_qudit_kraus(d, ctrl, eta, n_max=3)

        for k in range(d):
            f_k = 0.0 if k == ctrl else float(k)
            base = f_k * np.sqrt(sigma_p2)  # Gaussian-averaged: f_k*sqrt(sigma_p^2)
            decay = eta ** (0.5 * f_k ** 2)  # eta^{f_k^2/2}

            for n in range(4):
                expected = (base ** n) * decay / sqrt(factorial(n))
                actual = float(np.real(diags[n][k]))
                assert abs(actual - expected) < 1e-12, \
                    f"E_{n}[{k}]: expected {expected:.8f}, got {actual:.8f}"


class TestTargetQuditEqJ4:
    """Test target qudit noise against Eq. J4 of arXiv:2310.12110v3."""

    def test_three_coupling_tiers(self):
        """Eq. J4: three distinct coupling strengths f=1, 2, 3."""
        d = 5
        diags = target_qudit_kraus(d, control_level=0, target_level=1, eta=0.9, n_max=5)

        # E_0 should show three tiers
        e0 = np.real(diags[0])
        # Gaussian-averaged E_0[k] = eta^{f_k^2/2}:
        # f=1 (ctrl level 0): eta^0.5
        # f=2 (tgt level 1): eta^2
        # f=3 (spectators 2,3,4): eta^4.5
        assert abs(e0[0] - 0.9**0.5) < 1e-10, "Control level: f=1, eta^0.5"
        assert abs(e0[1] - 0.9**2) < 1e-10, "Target level: f=2, eta^2"
        for k in [2, 3, 4]:
            assert abs(e0[k] - 0.9**4.5) < 1e-10, f"Spectator level {k}: f=3, eta^4.5"

    def test_spectator_levels_equal(self):
        """All spectator levels (k not in {i,j}) should have identical dephasing."""
        d = 7
        diags = target_qudit_kraus(d, control_level=0, target_level=1, eta=0.95, n_max=5)
        for n in range(len(diags)):
            spectator_vals = [diags[n][k] for k in range(2, 7)]
            for v in spectator_vals:
                assert abs(v - spectator_vals[0]) < 1e-14, \
                    f"Spectator levels should be identical in E_{n}"


class TestSubspaceDepolarizing:
    """Test subspace depolarizing from Martin Ringbauer's characterization."""

    def test_completeness_all_dims(self):
        """sum K^dag K = I for d=3,5,7 and various level pairs."""
        for d in [3, 5, 7]:
            for i, j in [(0, 1), (0, d-1), (d//2, d-1)]:
                kraus = subspace_depolarizing_kraus(d, i, j, p=0.1)
                ok, dev = verify_matrix_kraus_completeness(kraus)
                assert ok, f"Completeness failed for d={d}, levels ({i},{j}): dev={dev:.2e}"

    def test_noiseless(self):
        """p=0: K_0 = I, K_{1,2,3} = 0."""
        kraus = subspace_depolarizing_kraus(d=5, level_i=0, level_j=1, p=0.0)
        assert np.allclose(kraus[0], np.eye(5))
        for k in range(1, 4):
            assert np.allclose(kraus[k], 0)

    def test_full_depolarizing(self):
        """p=1: K_0 projects onto complement, Paulis at full strength."""
        d = 5
        kraus = subspace_depolarizing_kraus(d, 0, 1, p=1.0)
        # K_0 should be zero on subspace, identity on complement
        assert abs(kraus[0][0, 0]) < 1e-14, "K_0[0,0] should be 0 at p=1"
        assert abs(kraus[0][1, 1]) < 1e-14, "K_0[1,1] should be 0 at p=1"
        assert abs(kraus[0][2, 2] - 1.0) < 1e-14, "K_0[2,2] should be 1 at p=1"

    def test_only_affects_subspace(self):
        """Error operators K_1,K_2,K_3 should be zero outside {i,j} subspace."""
        d = 5
        kraus = subspace_depolarizing_kraus(d, 1, 3, p=0.5)
        for k in range(1, 4):  # error operators
            for row in range(d):
                for col in range(d):
                    if row not in (1, 3) or col not in (1, 3):
                        assert abs(kraus[k][row, col]) < 1e-14, \
                            f"K_{k}[{row},{col}] should be 0 (outside subspace)"


class TestCombinedGateKraus:
    """Test the combined dephasing + depolarizing model."""

    def test_combined_completeness(self):
        """The composed channel should be trace-preserving."""
        d = 5
        result = combined_gate_kraus(d, 0, 1, eta_dephasing=0.95,
                                      p_depolarizing=0.05, n_max_dephasing=15)

        # Check dephasing completeness
        ok, dev = verify_kraus_completeness(result['control_dephasing'])
        assert ok, f"Control dephasing not complete: {dev:.2e}"

        ok, dev = verify_kraus_completeness(result['target_dephasing'])
        assert ok, f"Target dephasing not complete: {dev:.2e}"

        # Check depolarizing completeness
        ok, dev = verify_matrix_kraus_completeness(result['subspace_depol'])
        assert ok, f"Subspace depol not complete: {dev:.2e}"

    def test_dephasing_only_when_p_zero(self):
        """With p=0, subspace depol should just be identity."""
        result = combined_gate_kraus(d=3, control_level=0, target_level=1,
                                      eta_dephasing=0.9, p_depolarizing=0.0)
        assert np.allclose(result['subspace_depol'][0], np.eye(3))

    def test_no_noise_when_both_zero(self):
        """With eta=1 and p=0, everything should be identity."""
        result = combined_gate_kraus(d=3, control_level=0, target_level=1,
                                      eta_dephasing=1.0, p_depolarizing=0.0)
        # E_0 dephasing should be all ones
        assert np.allclose(result['control_dephasing'][0], np.ones(3))
        assert np.allclose(result['spectator_dephasing'][0], np.ones(3))


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
