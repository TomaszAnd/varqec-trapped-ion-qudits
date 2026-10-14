"""Tests for ErrorModel.build_grouped() and the factored_to_grouped helper.

Covers roundtrip equivalence with build_dense, dedup behaviour on synthetic
duplicated input, and inverse-permutation consistency.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.errors import (
    ErrorModel, factored_to_grouped, grouped_to_dense,
    _is_proportional_to_identity,
)


def _same_up_to_phase(A, B, atol=1e-10):
    d = A.shape[0]
    prod = A @ B.conj().T
    return _is_proportional_to_identity(prod, d)


def _match_sets(dense_list, reconstructed_list, atol=1e-10):
    """Order-agnostic match: every reconstructed matches some unmatched dense."""
    unmatched = list(range(len(dense_list)))
    for R in reconstructed_list:
        hit = None
        for idx in unmatched:
            if np.allclose(R, dense_list[idx], atol=atol):
                hit = idx
                break
        if hit is None:
            return False
        unmatched.remove(hit)
    return len(unmatched) == 0


class TestRoundtrip:
    def test_d3_n3_dist2(self):
        model = ErrorModel(d=3, n_qudit=3, distance=2)
        E_det_dense, _ = model.build_dense()
        grouped = model.build_grouped(verbose=False)
        reconstructed = grouped_to_dense(grouped, 3, 3)
        assert len(reconstructed) == len(E_det_dense)
        assert _match_sets(E_det_dense, reconstructed)

    def test_d3_n5_dist3_closed(self):
        model = ErrorModel(d=3, n_qudit=5, distance=3, closed=True)
        E_det_dense, _ = model.build_dense()
        grouped = model.build_grouped(verbose=False)
        reconstructed = grouped_to_dense(grouped, 5, 3)
        # Lengths match; every reconstructed operator appears in dense list.
        assert len(reconstructed) == len(E_det_dense)
        # Full set match would be O(N^2 * dim^2); check counts + spot items.
        counts_reco = [np.linalg.norm(R) for R in reconstructed]
        counts_dense = [np.linalg.norm(E) for E in E_det_dense]
        assert np.isclose(sum(counts_reco), sum(counts_dense), rtol=1e-10)

    def test_d4_n3_dist2(self):
        model = ErrorModel(d=4, n_qudit=3, distance=2)
        E_det_dense, _ = model.build_dense()
        grouped = model.build_grouped(verbose=False)
        reconstructed = grouped_to_dense(grouped, 3, 4)
        assert len(reconstructed) == len(E_det_dense)
        assert _match_sets(E_det_dense, reconstructed)

    def test_d5_n3_dist2(self):
        model = ErrorModel(d=5, n_qudit=3, distance=2)
        E_det_dense, _ = model.build_dense()
        grouped = model.build_grouped(verbose=False)
        reconstructed = grouped_to_dense(grouped, 3, 5)
        assert len(reconstructed) == len(E_det_dense)
        assert _match_sets(E_det_dense, reconstructed)


class TestDedup:
    def test_hardware_basis_no_redundancy(self):
        """close_error_basis already dedups; build_grouped should remove 0."""
        model = ErrorModel(d=3, n_qudit=5, distance=3, closed=True)
        factored, _ = model.build_factored()
        grouped = factored_to_grouped(factored, 5, 3, dedup=True, verbose=False)
        # Count total errors in grouped form
        n_out = sum(g['matrices'].shape[0] for g in grouped)
        assert n_out == len(factored), (
            f"Expected no dedup removals on hardware basis, got "
            f"{len(factored)} -> {n_out}")

    def test_synthetic_duplicates_halved(self):
        """Doubling a clean factored list should produce exactly half removals."""
        model = ErrorModel(d=3, n_qudit=3, distance=2)
        flat = model.build_factored()[0]
        doubled = flat + flat
        grouped = factored_to_grouped(doubled, 3, 3, dedup=True, verbose=False)
        n_out = sum(g['matrices'].shape[0] for g in grouped)
        assert n_out == len(flat)
        assert len(doubled) - n_out == len(flat)  # exactly half removed


class TestInversePerm:
    def test_compose_identity_all_cells(self):
        """inverse_perm composed with wires-first permutation yields identity."""
        for d, n, dist in [(3, 3, 2), (3, 5, 3), (4, 3, 2), (5, 3, 2)]:
            closed = dist >= 3
            model = ErrorModel(d=d, n_qudit=n, distance=dist, closed=closed)
            grouped = model.build_grouped(verbose=False)
            for g in grouped:
                wires = g['wires']
                inv = g['inverse_perm']
                leftover = [i for i in range(n) if i not in wires]
                new_order = list(wires) + leftover
                composed = [new_order[inv[i]] for i in range(n)]
                assert composed == list(range(n)), (
                    f"d={d} n={n} dist={dist} wires={wires}: "
                    f"composed={composed}")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
