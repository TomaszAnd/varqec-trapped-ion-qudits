"""Projective MAP instrument + exact Petz + sampled estimator (R8.5 T2).

Pins:
  (a) sector construction: orthonormal claimed basis, identity correction
      claims the code space first, per-sector image coordinates consistent;
      the instrument is exactly CPTP (unitary closure corrections +
      orthogonal completion);
  (fom) the fast score-matrix evaluator equals the dense instrument
      applied to exact channel blocks;
  (b) exact-Petz weak-noise limit: a distance-3 code under a weak dose
      recovers to F_avg ~ 1 under both instruments;
  (d) the sampled sector-averaged estimator agrees with the exact engine
      AND between the two unravelings (the R8 12-46x pathology is gone) --
      statistical-tolerance version at n = 5; the tight n = 5/7 runs live
      in results/campaigns/r8p5_instrument/.
"""
import numpy as np
import pytest

import src.jax_backend  # noqa: F401  (x64 first)
from src.catalog import five_qudit_code_states
from src.errors import ErrorModel
from src.meth_weights import compute_meth_group_weights
from src.decoders.weighted_map import build_weighted_correction_set
from src.decoders.exact_channel import (apply_dose, build_dose_steps,
                                        evolve_code_blocks)
from src.decoders.projective_decode import (
    apply_instrument_dense, build_petz_exact_dose,
    build_projective_instrument, build_syndrome_sectors,
    entanglement_fidelity_petz, entanglement_fidelity_projective,
    projective_map_score_matrix, sampled_entanglement_fidelity,
    sector_outcome_probs,
)
from src.simulation import make_channel_v2_noise_fn, make_meth_mc_noise_fn

SIGMA_A2 = 0.047
ETA = float(np.exp(-0.073))


def _a2a(n):
    return [(i, j, 0, 1) for i in range(n) for j in range(i + 1, n)]


def _rand_code(n, d, K, seed):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((d ** n, K)) + 1j * rng.standard_normal((d ** n, K))
    Q, _ = np.linalg.qr(M)
    return np.ascontiguousarray(Q[:, :K].T)


@pytest.fixture(scope="module")
def small_inst():
    d, n, K = 3, 2, 2
    cs = _rand_code(n, d, K, seed=5)
    E_full = ErrorModel(d=d, n_qudit=n, distance=3,
                        closed=True).build_grouped(verbose=False)
    _, prior = compute_meth_group_weights(d, n, distance=3, eta=ETA, n_max=2)
    corr = build_weighted_correction_set(E_full, prior, n, d, max_weight=2)
    return build_syndrome_sectors(cs, corr, d, n)


# ── (a) sector construction + CPTP ─────────────────────────────────────

def test_sectors_orthonormal_identity_first(small_inst):
    Ball = small_inst['Ball']
    R = Ball.shape[1]
    assert np.max(np.abs(Ball.conj().T @ Ball - np.eye(R))) < 1e-10
    s0 = small_inst['sectors'][0]
    # highest-prior correction is the identity -> claims the code space
    assert s0['wires'] == ()
    assert s0['dim'] == small_inst['code_states'].shape[0]
    B0 = Ball[:, :s0['dim']]
    cs = small_inst['code_states']
    # code space inside sector 0
    proj = B0 @ (B0.conj().T @ cs.T)
    assert np.max(np.abs(proj - cs.T)) < 1e-10
    rep = small_inst['report']
    assert rep['claimed_dim_R'] == R
    assert rep['rest_dim'] == 3 ** 2 * 1 - R + (9 - 9)  # dim - R
    assert rep['rest_dim'] >= 0


def test_instrument_cptp_dense(small_inst):
    d, n = 3, 2
    rng = np.random.default_rng(0)
    A = rng.standard_normal((d ** n, d ** n)) \
        + 1j * rng.standard_normal((d ** n, d ** n))
    sigma = A @ A.conj().T
    sigma /= np.trace(sigma)
    out = apply_instrument_dense(small_inst, sigma)
    assert abs(np.trace(out) - 1.0) < 1e-10
    assert np.max(np.abs(out - out.conj().T)) < 1e-10
    assert np.linalg.eigvalsh(out).min() > -1e-10
    probs = sector_outcome_probs(small_inst, sigma)
    assert abs(probs.sum() - 1.0) < 1e-10
    assert probs.min() > -1e-12


# ── (fom) fast evaluator == dense instrument ───────────────────────────

def test_score_matrix_matches_dense(small_inst):
    d, n = 3, 2
    cs = small_inst['code_states']
    K = cs.shape[0]
    steps = build_dose_steps(n, d, _a2a(n), ETA, sigma_a2=SIGMA_A2)
    blocks = evolve_code_blocks(cs, steps, d, n)
    f_fast = projective_map_score_matrix(small_inst, blocks)
    for i in range(K):
        for j in range(i, K):
            dense = apply_instrument_dense(small_inst, blocks[(i, j)])
            ref = cs[i].conj() @ (dense @ cs[j])
            assert abs(f_fast[i, j] - ref) < 1e-10


# ── (b) weak-noise limit, both instruments ─────────────────────────────

def test_weak_noise_recovers():
    d, n = 3, 5
    cs = five_qudit_code_states(d)
    eta_weak = 0.999
    inst = build_projective_instrument(cs, d, n, eta_weak, sigma_a2=0.0)
    steps = build_dose_steps(n, d, _a2a(n), eta_weak, sigma_a2=0.0)
    blocks = evolve_code_blocks(cs, steps, d, n)
    proj = entanglement_fidelity_projective(inst, blocks)
    petz = build_petz_exact_dose(cs, steps, d, n)
    pz = entanglement_fidelity_petz(petz, cs, blocks)
    # Petz (near-optimal) recovers to O(p^2): the weight-<=2 error mass of
    # the weak dose is fully correctable on a distance-3 stabilizer code.
    assert pz['rate_1mFavg'] < 1e-5
    assert max(abs(l) for l in pz['recovery_leak_per_codeword']) < 1e-6
    # The projective-MAP instrument keeps an O(p) residual on this code:
    # the hardware-basis corrections (Z_1 = diag(1,-1,1), ...) are non-Weyl,
    # so each ordered sector mixes Weyl syndrome spaces and its unitary
    # correction only partially inverts the collapsed error -- a real
    # property of the instrument, pinned here (measured 4.27e-3 at
    # eta = 0.999, i.e. ~18% of the dose's error mass).
    assert pz['rate_1mFavg'] < proj['rate_1mFavg'] < 1e-2


# ── (d) sampled estimator: exact agreement + unraveling independence ───

def test_sampled_estimator_unraveling_independent():
    d, n = 3, 5
    cs = five_qudit_code_states(d)
    gp = _a2a(n)
    inst = build_projective_instrument(cs, d, n, ETA, sigma_a2=SIGMA_A2)
    steps = build_dose_steps(n, d, gp, ETA, sigma_a2=SIGMA_A2)
    blocks = evolve_code_blocks(cs, steps, d, n)
    exact = entanglement_fidelity_projective(inst, blocks)

    n_traj = 400
    nf_b = make_channel_v2_noise_fn(n, d, gp, ETA, sigma_a2=SIGMA_A2, n_max=5)
    nf_a = make_meth_mc_noise_fn(n, d, gp, ETA, sigma_a2=SIGMA_A2)
    est_b = sampled_entanglement_fidelity(inst, nf_b, n_traj, seed=1)
    est_a = sampled_entanglement_fidelity(inst, nf_a, n_traj, seed=2)

    for est in (est_a, est_b):
        tol = 5.0 * est['F_e_se'] + 2e-3
        assert abs(est['F_e'] - exact['F_e']) < tol, (est['F_e'],
                                                      exact['F_e'], tol)
    joint = np.hypot(est_a['F_e_se'], est_b['F_e_se'])
    assert abs(est_a['F_e'] - est_b['F_e']) < 5.0 * joint + 2e-3
    # the R8 pathology was a 12-46x LER split; on the linear instrument the
    # two unravelings must agree to statistical precision, and both rates
    # must sit at the exact value's scale, not 10-100x above it
    assert est_a['rate_1mFavg'] < 5.0 * max(exact['rate_1mFavg'], 5e-3)


# ── dense channel application matches apply_instrument_dense ───

def test_channel_ops_match_dense(small_inst):
    """The gemm-based CPTP application (instrument_channel_ops +
    apply_instrument_channel) must equal the
    per-column apply_instrument_dense reference on a random density
    matrix AND on a non-Hermitian codeword block."""
    from src.decoders.projective_decode import (apply_instrument_channel,
                                                apply_instrument_dense,
                                                instrument_channel_ops)
    inst = small_inst
    d, n = inst['d'], inst['n_qudit']
    dim = d ** n
    Ball, W, mask = instrument_channel_ops(inst)
    rng = np.random.default_rng(11)
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    rho = (A @ A.conj().T)
    rho /= np.trace(rho).real
    cs = inst['code_states']
    block01 = np.outer(cs[0], cs[1].conj())        # non-Hermitian
    for sigma in (rho, block01):
        ref = apply_instrument_dense(inst, sigma)
        new = apply_instrument_channel(sigma, Ball, W, mask)
        assert np.max(np.abs(ref - new)) < 1e-12
