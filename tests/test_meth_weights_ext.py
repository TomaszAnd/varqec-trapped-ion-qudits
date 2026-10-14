"""meth_weights extension: weight-3 products and q=4,5.

The earlier limitation (weight-2-only reshape in _factor_per_qudit)
is fixed by a recursive SVD split; these tests pin (i) exact regression of
the committed q=3 weight-2 values, (ii) weight-3 factoring correctness,
(iii) q=4,5 weight distributions.
"""
import numpy as np

import src.jax_backend  # noqa: F401  (x64 first)
from src.meth_weights import (_factor_per_qudit, _kron_chain,
                              compute_meth_group_weights)

# committed-implementation regression pins (flat[:4], eta=0.9296, n_max=2)
PIN_Q3_W2 = [0.03384086, 0.01001360, 0.01124956, 0.00924881]   # (3,5,dist3)
PIN_Q3_D4 = [0.02206522, 0.00652916, 0.00733504, 0.00603049]   # (3,5,dist4)
PIN_Q4 = [0.00797644, 0.00401526, 0.00404098, 0.00359996]      # (4,5,dist3)
PIN_Q5 = [0.00364403, 0.00200580, 0.00202898, 0.00207714]      # (5,5,dist3)


def _rand_unitary(d, rng):
    M = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(M)
    return Q


class TestFactorPerQudit:
    def test_weight3_exact_factoring(self):
        d, n = 3, 5
        rng = np.random.default_rng(0)
        A, B, C = (_rand_unitary(d, rng) for _ in range(3))
        wires = [0, 2, 4]
        M = np.kron(np.kron(A, B), C)
        factors = _factor_per_qudit(M, wires, n, d)
        # off-support factors are identity
        for q in (1, 3):
            assert np.allclose(factors[q], np.eye(d))
        # on-support factors reproduce M exactly (phases may redistribute)
        assert np.allclose(_kron_chain([factors[0], factors[2], factors[4]]),
                           M, atol=1e-10)

    def test_weight4_exact_factoring(self):
        d, n = 3, 6
        rng = np.random.default_rng(1)
        parts = [_rand_unitary(d, rng) for _ in range(4)]
        M = _kron_chain(parts)
        wires = [1, 2, 4, 5]
        factors = _factor_per_qudit(M, wires, n, d)
        assert np.allclose(_kron_chain([factors[w] for w in wires]), M,
                           atol=1e-10)

    def test_weight2_unchanged(self):
        d, n = 3, 5
        rng = np.random.default_rng(2)
        A, B = _rand_unitary(d, rng), _rand_unitary(d, rng)
        M = np.kron(A, B)
        factors = _factor_per_qudit(M, [1, 3], n, d)
        assert np.allclose(np.kron(factors[1], factors[3]), M, atol=1e-10)


class TestGroupWeights:
    def test_q3_weight2_regression(self):
        _, flat = compute_meth_group_weights(3, 5, distance=3)
        assert len(flat) == 221
        assert np.isclose(flat.sum(), 1.0)
        assert np.allclose(flat[:4], PIN_Q3_W2, atol=1e-7)

    def test_q3_distance4_weight3(self):
        gw, flat = compute_meth_group_weights(3, 5, distance=4)
        assert len(flat) == 861
        assert np.isclose(flat.sum(), 1.0)
        assert np.all(flat > 0)          # no rejected (zeroed) ops
        assert np.allclose(flat[:4], PIN_Q3_D4, atol=1e-7)
        # group alignment with the grouped detection set
        from src.errors import ErrorModel
        grouped = ErrorModel(d=3, n_qudit=5, distance=4,
                             closed=True).build_grouped(verbose=False)
        assert len(gw) == len(grouped)
        for g, w in zip(grouped, gw):
            assert g['matrices'].shape[0] == len(w)

    def test_q4_q5_weight2(self):
        for d, npin, pin in ((4, 481, PIN_Q4), (5, 841, PIN_Q5)):
            _, flat = compute_meth_group_weights(d, 5, distance=3)
            assert len(flat) == npin
            assert np.isclose(flat.sum(), 1.0)
            assert np.all(flat > 0)
            assert np.allclose(flat[:4], pin, atol=1e-7)
