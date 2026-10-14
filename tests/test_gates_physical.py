"""Gate regressions against primary-source definitions .

References:
  - MS: Ringbauer et al., Nat. Phys. 18, 1053 (2022), Eq. (2) (arXiv:2109.06903,
    label eq:entOps): MS(theta, phi) = expm(-i(theta/4)(sigma_phi (x) I + I (x) sigma_phi)^2)
  - LS: Hrmo et al., Nat. Commun. 14, 2242 (2023), Eq. (3) (arXiv:2206.04104,
    label eq:GateOperator): |jj> -> |jj>, |jk> -> e^{i theta}|jk> for j != k
  - ZZ subspace gate: paper Eq. (4): exp(-i(theta/2) Z^{(jk)} (x) Z^{(jk)})
"""
import itertools

import numpy as np
import pytest
from scipy.linalg import expm

from src.gates import (XY_gate, Z_gate, MS_gate, CSUM_gate, CSUB_gate, CEX_gate,
                       zz_subspace_gate, ls_global_gate)

RNG = np.random.default_rng(42)
DIMS = [3, 4, 5]


def _exact_ms(theta, phi, j, k, d):
    sig = np.zeros((d, d), complex)
    sig[j, k] = np.exp(-1j * phi)
    sig[k, j] = np.exp(1j * phi)
    I = np.eye(d)
    S = np.kron(sig, I) + np.kron(I, sig)
    return expm(-1j * (theta / 4) * (S @ S))


def _assert_unitary(U, tol=1e-12):
    U = np.asarray(U)
    assert np.max(np.abs(U @ U.conj().T - np.eye(U.shape[0]))) < tol


class TestUnitarity:
    @pytest.mark.parametrize("d", DIMS)
    def test_all_gates_unitary_random_params(self, d):
        for _ in range(5):
            phi, alpha, theta = RNG.uniform(0, 2 * np.pi, 3)
            j, k = RNG.choice(d, size=2, replace=False)
            j, k = int(j), int(k)
            _assert_unitary(XY_gate(phi, alpha, j, k, d))
            _assert_unitary(Z_gate(theta, j, k, d))
            _assert_unitary(MS_gate(phi, theta, j, k, d))
            _assert_unitary(zz_subspace_gate(theta, j, k, d))
            _assert_unitary(ls_global_gate(theta, d))
        _assert_unitary(CSUM_gate(d))
        _assert_unitary(CSUB_gate(d))
        _assert_unitary(CEX_gate(d, c=1, t1=0, t2=1))


class TestCEX:
    """Embedded-qubit controlled-exchange gate (Amendment 2)."""

    @pytest.mark.parametrize("d", DIMS)
    def test_unitary_and_involution(self, d):
        U = np.asarray(CEX_gate(d, c=1, t1=0, t2=1))
        _assert_unitary(U)
        # a single basis transposition is its own inverse
        assert np.max(np.abs(U @ U - np.eye(d * d))) < 1e-13

    @pytest.mark.parametrize("d", DIMS)
    def test_exact_permutation(self, d):
        c, t1, t2 = 1, 0, 2 if d >= 3 else 1
        U = np.asarray(CEX_gate(d, c=c, t1=t1, t2=t2))
        i1, i2 = c * d + t1, c * d + t2
        for a in range(d):
            for b in range(d):
                idx = a * d + b
                col = U[:, idx]
                if idx == i1:
                    expected = i2
                elif idx == i2:
                    expected = i1
                else:
                    expected = idx  # identity everywhere else
                assert np.argmax(np.abs(col)) == expected and \
                    abs(col[expected] - 1.0) < 1e-13, (d, a, b)

    def test_reduces_to_cnot_at_d2(self):
        # d=2, c=1, t1=0, t2=1 is exactly the qubit CNOT (control |1>).
        U = np.asarray(CEX_gate(2, c=1, t1=0, t2=1))
        CNOT = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=complex)
        assert np.max(np.abs(U - CNOT)) < 1e-13

    def test_embedded_bell_state(self):
        # CEX(d=3,c=1,t1=0,t2=1) @ (|00>+|10>)/sqrt2 == (|00>+|11>)/sqrt2.
        d = 3
        U = np.asarray(CEX_gate(d, c=1, t1=0, t2=1))
        psi_in = np.zeros(d * d, dtype=complex)
        psi_in[0 * d + 0] = 1 / np.sqrt(2)  # |00>
        psi_in[1 * d + 0] = 1 / np.sqrt(2)  # |10>
        psi_out = U @ psi_in
        bell = np.zeros(d * d, dtype=complex)
        bell[0 * d + 0] = 1 / np.sqrt(2)  # |00>
        bell[1 * d + 1] = 1 / np.sqrt(2)  # |11>
        assert np.max(np.abs(psi_out - bell)) < 1e-13

    def test_identity_outside_control_block(self):
        # every |a,b> with a != c is untouched, and |c,b> with b not in {t1,t2} too
        d, c, t1, t2 = 3, 1, 0, 1
        U = np.asarray(CEX_gate(d, c=c, t1=t1, t2=t2))
        for a in range(d):
            for b in range(d):
                idx = a * d + b
                if (a, b) in {(c, t1), (c, t2)}:
                    continue
                assert abs(U[idx, idx] - 1.0) < 1e-13, (a, b)


class TestMSPhysical:
    @pytest.mark.parametrize("d", DIMS)
    def test_ms_matches_expm_reference(self, d):
        pairs = list(itertools.combinations(range(d), 2))[:4]
        for j, k in pairs:
            for theta in np.linspace(0.0, 2 * np.pi, 7):
                for phi in (0.0, 0.7, np.pi / 2, 4.1):
                    U = np.asarray(MS_gate(phi, theta, j, k, d))
                    E = _exact_ms(theta, phi, j, k, d)
                    assert np.max(np.abs(U - E)) < 1e-13, (d, j, k, theta, phi)

    def test_ms_reduces_to_qubit_ms_at_q2(self):
        # At d=2 the gate equals e^{-i theta/2} * expm(-i(theta/2) X_phi (x) X_phi)
        # (global phase documented in Ringbauer arXiv:2109.06903 main text).
        for theta in (0.3, 0.777, 2.5):
            U = np.asarray(MS_gate(0.0, theta, 0, 1, 2))
            X = np.array([[0, 1], [1, 0]], dtype=complex)
            STD = expm(-1j * (theta / 2) * np.kron(X, X))
            phase = np.exp(-1j * theta / 2)
            assert np.max(np.abs(U - phase * STD)) < 1e-13
            # and full agreement with the exact qudit formula
            assert np.max(np.abs(U - _exact_ms(theta, 0.0, 0, 1, 2))) < 1e-13

    def test_ms_single_active_spectator_phase(self):
        # Basis states with exactly one qudit in {j,k} get phase e^{-i theta/4}.
        d, j, k, theta = 3, 0, 1, 1.234
        U = np.asarray(MS_gate(0.0, theta, j, k, d))
        idx = 0 * d + 2  # |0,2>: first active, second spectator
        assert abs(U[idx, idx] - np.exp(-1j * theta / 4)) < 1e-13


class TestLSGates:
    @pytest.mark.parametrize("d", DIMS)
    def test_ls_global_matches_hrmo_eq3(self, d):
        for theta in (0.0, 0.9, np.pi, 5.0):
            U = np.asarray(ls_global_gate(theta, d))
            diag = np.array([np.exp(1j * theta) if a != b else 1.0
                             for a in range(d) for b in range(d)], complex)
            assert np.max(np.abs(U - np.diag(diag))) < 1e-13

    @pytest.mark.parametrize("d", DIMS)
    def test_zz_subspace_matches_paper_eq4(self, d):
        for theta in (0.4, 1.9):
            for j, k in list(itertools.combinations(range(d), 2))[:3]:
                Zjk = np.zeros((d, d), complex)
                Zjk[j, j] = 1.0
                Zjk[k, k] = -1.0
                E = expm(-1j * (theta / 2) * np.kron(Zjk, Zjk))
                U = np.asarray(zz_subspace_gate(theta, j, k, d))
                assert np.max(np.abs(U - E)) < 1e-13

    def test_zz_subspace_is_not_ms(self):
        # Guard against the naming collision : the diagonal
        # ZZ gate must stay clearly distinct from the physical MS gate.
        theta = 1.1
        U_zz = np.asarray(zz_subspace_gate(theta, 0, 1, 3))
        U_ms = np.asarray(MS_gate(0.0, theta, 0, 1, 3))
        assert np.max(np.abs(U_zz - U_ms)) > 0.5

    def test_zz_subspace_diverges_from_hrmo_at_d3(self):
        theta = 1.1
        U_zz = np.asarray(zz_subspace_gate(theta, 0, 1, 3))
        U_ls = np.asarray(ls_global_gate(theta, 3))
        assert np.max(np.abs(U_zz - U_ls)) > 0.5
