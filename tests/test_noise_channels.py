"""Kraus-family channel properties .

Trace preservation of the truncated dephasing families is asserted WITH the
measured truncation deficit made explicit: the Gaussian-dephasing Kraus series
is complete only as n_max -> infinity, and at the production default n_max=5
the deficit is ~3.3e-5 (f_k = k, d=3), ~2.4e-3 (target family, coupling up to
3), and ~3.2e-2 (d=5 spectator, coupling up to 4). These bounds pin the
documented behaviour; shrinking them requires raising n_max, not editing the
test.
"""
import numpy as np
import pytest

from src.correlated_noise import (
    control_qudit_kraus, control_qudit_kraus_simplified, target_qudit_kraus,
    spectator_qudit_kraus, subspace_depolarizing_kraus,
    amplitude_damping_kraus, verify_kraus_completeness,
    verify_matrix_kraus_completeness)

ETA_OP = 0.9296  # calibrated operating point, sigma_p^2 = 0.073


def _deficit(diags):
    _, dev = verify_kraus_completeness(diags)
    return dev


class TestDephasingTracePreservation:
    def test_control_physical_d3_nmax5(self):
        assert _deficit(control_qudit_kraus(3, 0, ETA_OP, 5)) < 1e-4

    def test_control_simplified_d3_nmax5(self):
        assert _deficit(control_qudit_kraus_simplified(3, 0, ETA_OP, 5)) < 1e-3

    def test_target_d3_nmax5_documented_deficit(self):
        # coupling up to 3 -> slowest-converging family at n_max=5
        dev = _deficit(target_qudit_kraus(3, 0, 1, ETA_OP, 5))
        assert dev < 5e-3, "target-family truncation deficit grew"

    def test_spectator_d3_nmax5(self):
        assert _deficit(spectator_qudit_kraus(3, ETA_OP, 5)) < 1e-4

    def test_spectator_d5_nmax5_documented_deficit(self):
        dev = _deficit(spectator_qudit_kraus(5, ETA_OP, 5))
        assert dev < 5e-2, "d=5 spectator truncation deficit grew"

    def test_high_nmax_converges_to_tp(self):
        # the family is complete in the n_max -> infinity limit
        assert _deficit(target_qudit_kraus(3, 0, 1, ETA_OP, 40)) < 1e-12

    def test_eta_one_is_identity(self):
        # eta = 1 => sigma_p^2 = 0: E_0 = I, E_{n>0} = 0
        for fam in (control_qudit_kraus(3, 0, 1.0, 5),
                    target_qudit_kraus(3, 0, 1, 1.0, 5),
                    spectator_qudit_kraus(3, 1.0, 5)):
            assert np.allclose(fam[0], np.ones(3))
            for E in fam[1:]:
                assert np.max(np.abs(E)) < 1e-15


class TestDepolarizing:
    @pytest.mark.parametrize("d", [3, 4, 5])
    @pytest.mark.parametrize("p", [0.0, 0.1, 1.0])
    def test_cptp(self, d, p):
        ops = subspace_depolarizing_kraus(d, 0, 1, p)
        ok, dev = verify_matrix_kraus_completeness(ops)
        assert ok, f"deviation {dev}"

    def test_p_zero_is_identity(self):
        ops = subspace_depolarizing_kraus(3, 0, 1, 0.0)
        assert np.allclose(ops[0], np.eye(3))
        for K in ops[1:]:
            assert np.max(np.abs(K)) < 1e-15


class TestAmplitudeDamping:
    @pytest.mark.parametrize("d", [2, 3, 4, 5])
    @pytest.mark.parametrize("gamma", [0.0, 0.05, 0.5, 1.0])
    def test_exactly_trace_preserving(self, d, gamma):
        ops = amplitude_damping_kraus(d, gamma)
        ok, dev = verify_matrix_kraus_completeness(ops)
        assert ok, f"d={d} gamma={gamma}: deviation {dev}"

    def test_gamma_zero_is_identity(self):
        ops = amplitude_damping_kraus(3, 0.0)
        assert np.allclose(ops[0], np.eye(3))
        for A in ops[1:]:
            assert np.max(np.abs(A)) < 1e-15

    def test_qubit_reduction_matches_standard_ad(self):
        gamma = 0.3
        A0, A1 = amplitude_damping_kraus(2, gamma)
        assert np.allclose(A0, [[1, 0], [0, np.sqrt(1 - gamma)]])
        assert np.allclose(A1, [[0, np.sqrt(gamma)], [0, 0]])

    def test_full_decay_at_gamma_one(self):
        # gamma=1: every level decays; the channel maps everything toward |0>
        d = 3
        ops = amplitude_damping_kraus(d, 1.0)
        rho = np.eye(d, dtype=complex) / d
        out = sum(A @ rho @ A.conj().T for A in ops)
        expected = np.zeros((d, d), complex)
        expected[0, 0] = 1.0
        assert np.allclose(out, expected)
