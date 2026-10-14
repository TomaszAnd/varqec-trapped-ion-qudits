"""
Tests for code catalog, benchmark codes, and code analysis.
Run: python3 -m pytest tests/test_code_catalog.py -v
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.catalog import five_qudit_code_states, load_code, list_codes
from src.analysis import (
    compute_kl_residuals, compute_distance,
    compute_weight_enumerators, compute_entanglement_entropy,
)


class TestFiveQuditCode:
    def test_orthonormality_q3(self):
        cs = five_qudit_code_states(q=3)
        gram = cs @ cs.conj().T
        assert np.allclose(gram, np.eye(3), atol=1e-12)

    def test_shape_q3(self):
        cs = five_qudit_code_states(q=3)
        assert cs.shape == (3, 243)

    def test_orthonormality_q5(self):
        cs = five_qudit_code_states(q=5)
        gram = cs @ cs.conj().T
        assert np.allclose(gram, np.eye(5), atol=1e-12)

    def test_detects_weight1_gen_pauli(self):
        """The [[5,1,3]]_3 code detects all weight-1 generalized Pauli errors."""
        q = 3
        cs = five_qudit_code_states(q)
        omega = np.exp(2j * np.pi / q)
        GX = np.zeros((q, q), dtype=complex)
        for j in range(q):
            GX[(j+1) % q, j] = 1.0
        GZ = np.diag([omega**j for j in range(q)])
        Id = np.eye(q, dtype=complex)
        max_ov = 0
        for a in range(q):
            for b in range(q):
                if a == 0 and b == 0:
                    continue
                P = np.linalg.matrix_power(GX, a) @ np.linalg.matrix_power(GZ, b)
                for qudit in range(5):
                    op = 1
                    for i in range(5):
                        op = np.kron(op, P if i == qudit else Id)
                    for i in range(q):
                        for j in range(i+1, q):
                            max_ov = max(max_ov, abs(np.vdot(cs[i], op @ cs[j])))
        assert max_ov < 1e-10, f"Max overlap {max_ov}"

    def test_distance_is_3(self):
        """Verify the code has distance exactly 3 under generalized Paulis."""
        q = 3
        cs = five_qudit_code_states(q)
        omega = np.exp(2j * np.pi / q)
        GX = np.zeros((q, q), dtype=complex)
        for j in range(q):
            GX[(j+1) % q, j] = 1.0
        GZ = np.diag([omega**j for j in range(q)])
        gen_paulis = []
        for a in range(q):
            for b in range(q):
                if a == 0 and b == 0:
                    continue
                gen_paulis.append(np.linalg.matrix_power(GX, a) @ np.linalg.matrix_power(GZ, b))
        d = compute_distance(cs, 5, q, gen_paulis, max_weight=3)
        assert d == 3, f"Expected distance 3, got {d}"


class TestCodeCatalog:
    def test_list_codes(self):
        codes = list_codes()
        assert 'central_553_3' in codes
        assert 'five_qudit_d3' in codes

    def test_load_analytical(self):
        data = load_code('five_qudit_d3')
        assert data['code_states'].shape == (3, 243)
        assert data['metadata']['distance'] == 3
        assert data['params'] is None

    def test_no_duplicate_npz_in_registry(self):
        """Each .npz file is reachable under exactly one code name."""
        from src.catalog import _CODE_REGISTRY
        files = [v[0] for v in _CODE_REGISTRY.values() if v[0] is not None]
        assert len(files) == len(set(files)), (
            f"duplicate npz files in registry: "
            f"{[f for f in files if files.count(f) > 1]}")


class TestCodeAnalysis:
    def test_weight_enumerators_benchmark(self):
        """Weight-0 enumerators for [[5,1,3]]_3."""
        cs = five_qudit_code_states(3)
        omega = np.exp(2j * np.pi / 3)
        GX = np.zeros((3, 3), dtype=complex)
        for j in range(3):
            GX[(j+1) % 3, j] = 1.0
        GZ = np.diag([omega**j for j in range(3)])
        gen_paulis = []
        for a in range(3):
            for b in range(3):
                if a == 0 and b == 0:
                    continue
                gen_paulis.append(np.linalg.matrix_power(GX, a) @ np.linalg.matrix_power(GZ, b))
        A, B = compute_weight_enumerators(cs, 5, 3, gen_paulis, max_weight=1)
        assert abs(A[0] - 1.0) < 1e-10, f"A_0 should be 1, got {A[0]}"
        assert abs(B[0] - 1.0) < 1e-10, f"B_0 should be 1, got {B[0]}"
        # For a distance-3 code: A_1 should be 0 (detects all weight-1)
        assert A[1] < 1e-8, f"A_1 should be ~0 for distance-3, got {A[1]}"

    def test_entanglement_entropy(self):
        cs = five_qudit_code_states(3)
        entropies = compute_entanglement_entropy(cs, [0, 1], 5, 3)
        assert len(entropies) == 3
        for S in entropies:
            assert S >= 0, f"Entropy should be non-negative, got {S}"

    def test_kl_residuals_benchmark(self):
        """[[5,1,3]]_3 should have zero KL residuals for generalized Paulis."""
        cs = five_qudit_code_states(3)
        omega = np.exp(2j * np.pi / 3)
        GX = np.zeros((3, 3), dtype=complex)
        for j in range(3):
            GX[(j+1) % 3, j] = 1.0
        GZ = np.diag([omega**j for j in range(3)])
        Id = np.eye(3, dtype=complex)
        E_corr = [np.eye(243, dtype=complex)]
        for a in range(3):
            for b in range(3):
                if a == 0 and b == 0:
                    continue
                P = np.linalg.matrix_power(GX, a) @ np.linalg.matrix_power(GZ, b)
                for q in range(5):
                    op = 1
                    for i in range(5):
                        op = np.kron(op, P if i == q else Id)
                    E_corr.append(op)
        offdiag, var = compute_kl_residuals(cs, E_corr)
        assert offdiag < 1e-10, f"Off-diagonal should be ~0, got {offdiag}"
        assert var < 1e-10, f"Variance should be ~0, got {var}"
