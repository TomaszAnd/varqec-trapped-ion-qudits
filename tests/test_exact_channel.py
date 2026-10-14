"""Exact density-matrix channel engine validation (R8.5 T1).

Pins:
  (a) CPTP at n in {2, 3, 5}: trace preservation, Hermiticity, positivity
      of the engine output on random pure inputs, v1 and v2;
  (b) at n = 2 the engine's 'kraus' mode reproduces the explicit
      gate_channel_kraus_v2 Cartesian-product density matrix (the R8
      moment-check reference machinery) to numerical precision, and the
      'exact' (analytic Gaussian) masks agree with the truncated series
      at the moment-check n_max to truncation tolerance;
  (s) sigma_a2 = 0 reproduces the v1 dephasing channel exactly;
  (adj) the adjoint dose is the true Hilbert-Schmidt adjoint;
  (c) the trajectory averages of BOTH make_channel_v2_noise_fn (MODE B)
      and make_meth_mc_noise_fn (MODE A) converge to the engine's rho
      (statistical tolerance; the full n = 5 dose check with timing lives
      in results/campaigns/r8p5_instrument/validate_engine.py).
"""
import numpy as np
import pytest

import src.jax_backend  # noqa: F401  (x64 first)
from src.correlated_noise import gate_channel_kraus_v2
from src.decoders.exact_channel import apply_dose, build_dose_steps
from src.simulation import make_channel_v2_noise_fn, make_meth_mc_noise_fn

SIGMA_A2 = 0.047
SIGMA_P2 = 0.073
ETA = float(np.exp(-SIGMA_P2))


def _a2a_pairs(n):
    return [(i, j, 0, 1) for i in range(n) for j in range(i + 1, n)]


def _rand_state(dim, seed):
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return psi / np.linalg.norm(psi)


# ── (a) CPTP ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [2, 3, 5])
@pytest.mark.parametrize("sigma_a2", [0.0, SIGMA_A2])
def test_cptp_exact_mode(n, sigma_a2):
    d = 3
    steps = build_dose_steps(n, d, _a2a_pairs(n), ETA, sigma_a2=sigma_a2)
    psi = _rand_state(d ** n, seed=n)
    rho = apply_dose(np.outer(psi, psi.conj()), steps, d, n)
    assert abs(np.trace(rho) - 1.0) < 1e-12
    assert np.max(np.abs(rho - rho.conj().T)) < 1e-12
    evals = np.linalg.eigvalsh(rho)
    assert evals.min() > -1e-12


# ── (b) n = 2 moment-check reference ───────────────────────────────────

def test_engine_matches_explicit_kraus_product_n2():
    """Engine 'kraus' mode == explicit Cartesian-product Kraus application
    of gate_channel_kraus_v2 at the moment-check truncations (n_max 8/10)."""
    d = 3
    ch = gate_channel_kraus_v2(d, 0, 1, ETA, SIGMA_A2,
                               n_max_dephasing=8, n_max_jitter=10)
    two_deph = [np.kron(np.diag(a), np.diag(b))
                for a in ch['control_dephasing'] for b in ch['target_dephasing']]
    KV2 = [J @ Dd for J in ch['pair_jitter'] for Dd in two_deph]
    psi = _rand_state(d * d, seed=3)
    rho_ref = np.zeros((d * d, d * d), dtype=complex)
    for K in KV2:
        v = K @ psi
        rho_ref += np.outer(v, v.conj())

    steps = build_dose_steps(2, d, [(0, 1, 0, 1)], ETA, sigma_a2=SIGMA_A2,
                             mode="kraus", n_max_dephasing=8, n_max_jitter=10)
    rho_eng = apply_dose(np.outer(psi, psi.conj()), steps, d, 2)
    assert np.max(np.abs(rho_eng - rho_ref)) < 1e-12


def test_exact_masks_match_truncated_series():
    """Analytic Gaussian masks == truncated Kraus-series masks at the
    moment-check n_max, to series-truncation tolerance."""
    d = 3
    psi = _rand_state(d * d, seed=7)
    rho0 = np.outer(psi, psi.conj())
    s_ex = build_dose_steps(2, d, [(0, 1, 0, 1)], ETA, sigma_a2=SIGMA_A2,
                            mode="exact")
    s_kr = build_dose_steps(2, d, [(0, 1, 0, 1)], ETA, sigma_a2=SIGMA_A2,
                            mode="kraus", n_max_dephasing=8, n_max_jitter=10)
    dev = np.max(np.abs(apply_dose(rho0, s_ex, d, 2)
                        - apply_dose(rho0, s_kr, d, 2)))
    assert dev < 1e-7


# ── (s) sigma_a2 = 0 reduction ─────────────────────────────────────────

def test_sigma_a2_zero_is_v1():
    d, n = 3, 3
    psi = _rand_state(d ** n, seed=11)
    rho0 = np.outer(psi, psi.conj())
    s_v2 = build_dose_steps(n, d, _a2a_pairs(n), ETA, sigma_a2=0.0)
    for st in s_v2:
        assert st['V1'] is None and st['jmask'] is None
    s_v1 = build_dose_steps(n, d, _a2a_pairs(n), ETA)
    assert np.max(np.abs(apply_dose(rho0, s_v2, d, n)
                         - apply_dose(rho0, s_v1, d, n))) == 0.0


# ── (adj) adjoint dose ─────────────────────────────────────────────────

@pytest.mark.parametrize("n", [2, 3])
def test_adjoint_is_hs_adjoint(n):
    d = 3
    dim = d ** n
    rng = np.random.default_rng(n)
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    B = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    steps = build_dose_steps(n, d, _a2a_pairs(n), ETA, sigma_a2=SIGMA_A2)
    lhs = np.trace(A.conj().T @ apply_dose(B, steps, d, n))
    rhs = np.trace(apply_dose(A, steps, d, n, adjoint=True).conj().T @ B)
    assert abs(lhs - rhs) < 1e-10 * max(1.0, abs(lhs))


def test_adjoint_is_unital():
    """Both channel factors are unital (doubly stochastic masks), so the
    adjoint dose fixes the identity -- quick self-consistency pin."""
    d, n = 3, 2
    steps = build_dose_steps(n, d, _a2a_pairs(n), ETA, sigma_a2=SIGMA_A2)
    I = np.eye(d ** n, dtype=complex)
    assert np.max(np.abs(apply_dose(I, steps, d, n, adjoint=True) - I)) < 1e-12


# ── (c) trajectory-average convergence, one gate at n = 2 ──────────────

@pytest.mark.parametrize("maker,mode,n_traj,tol_mult", [
    (make_channel_v2_noise_fn, "kraus", 30_000, 5.0),
    (make_meth_mc_noise_fn, "exact", 30_000, 5.0),
])
def test_trajectory_average_converges(maker, mode, n_traj, tol_mult):
    d = 3
    gp = [(0, 1, 0, 1)]
    kw = {"n_max": 5} if maker is make_channel_v2_noise_fn else {}
    nf = maker(2, d, gp, ETA, sigma_a2=SIGMA_A2, **kw)
    steps = build_dose_steps(2, d, gp, ETA, sigma_a2=SIGMA_A2, mode=mode,
                             n_max_dephasing=5, n_max_jitter=10)
    psi = _rand_state(d * d, seed=3)
    rho_exact = apply_dose(np.outer(psi, psi.conj()), steps, d, 2)
    rng = np.random.default_rng(12345)
    acc = np.zeros((d * d, d * d), dtype=complex)
    for _ in range(n_traj):
        v = nf(psi, rng)
        acc += np.outer(v, v.conj())
    acc /= n_traj
    dev = np.max(np.abs(acc - rho_exact))
    # R8 moment check observed max-dev ~ 0.36 / sqrt(N); allow tol_mult x.
    assert dev < tol_mult * 0.36 / np.sqrt(n_traj), dev


# ── (T0, R9) front-axes jitter fast path is bit-identical ──────────────

def test_fast_jitter_bit_identical():
    """The front-axes jitter path (_jitter_pair_front) must be
    BIT-identical to the reference _rotate_pair/_schur_pair sequence:
    tensordot canonicalizes operand layout before the gemm, so only data
    movement may change. Pinned at n in {3, 5}, forward and adjoint.
    (The fully fused 81x81 pair-superoperator matmul is NOT bit-identical
    -- it reorders products -- and was rejected for that reason.)"""
    from src.decoders.exact_channel import (_jitter_pair_front, _rotate_pair,
                                            _schur_pair)
    d = 3
    for n in (3, 5):
        steps = build_dose_steps(n, d, _a2a_pairs(n), ETA, sigma_a2=SIGMA_A2)
        rng = np.random.default_rng(n)
        dim = d ** n
        A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal(
            (dim, dim))
        rho = (A @ A.conj().T) / dim
        for step in steps[:3]:
            V1, jm = step['V1'], step['jmask']
            qa, qb = step['qa'], step['qb']
            for adj in (False, True):
                m = jm.conj() if adj else jm
                T_ref = rho.reshape((d,) * (2 * n)).copy()
                T_ref = _rotate_pair(T_ref, V1.conj().T, qa, qb, n)
                _schur_pair(T_ref, m, qa, qb, n, d)
                T_ref = _rotate_pair(T_ref, V1, qa, qb, n)
                T_new = _jitter_pair_front(
                    rho.reshape((d,) * (2 * n)).copy(), V1, m, qa, qb, n, d)
                assert np.array_equal(np.ascontiguousarray(T_ref),
                                      np.ascontiguousarray(T_new))
