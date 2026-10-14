"""Validation battery for the calibrated channel with the angle-fluctuation term.

Pins, per the approved decisions:
  (a) CPTP of ms_jitter_kraus and the v2 composition at d = 3, 4, 5,
      three strengths, n_max in {10, 14};
  (b) the active-subspace Pauli twirl of ms_jitter_kraus equals the SINGLE
      sigma_phi (x) sigma_phi term at p = (1 - e^{-sigma_a2/2})/2 ;
  (d) v2 reduces to the dephasing channel at sigma_a2 = 0 through the FULL
      noise-fn path (bit-identical states, same rng stream), and
      gate_channel_kraus_v2 reproduces the v1 Kraus builders bit-identically;
  (w) compute_meth_group_weights_v2(sigma_a2=0) == v1 weights exactly.

MODE A <-> MODE B agreement (c) and the phi-sensitivity probe (e) are
script-level measurements (results/campaigns/r8_campaign/validation),
not unit tests.
"""
import numpy as np
import pytest

import src.jax_backend  # noqa: F401  (x64 first)
from src.correlated_noise import (
    ms_jitter_kraus, gate_channel_kraus_v2, _ms_jitter_eigensystem,
    control_qudit_kraus, target_qudit_kraus, spectator_qudit_kraus,
    verify_matrix_kraus_completeness,
)
from src.meth_weights import (
    jitter_two_body_rate, compute_meth_group_weights,
    compute_meth_group_weights_v2,
)

SIGMA_A2 = 0.047     # Meth-fitted entangling-gate jitter variance
SIGMA_P2 = 0.073     # Meth-fitted phase variance
ETA = float(np.exp(-SIGMA_P2))


# ── (a) CPTP ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("d", [3, 4, 5])
@pytest.mark.parametrize("sigma_a2", [0.01, SIGMA_A2, 0.2])
@pytest.mark.parametrize("n_max", [10, 14])
def test_ms_jitter_kraus_cptp(d, sigma_a2, n_max):
    ops = ms_jitter_kraus(d, sigma_a2, phi=0.3, n_max=n_max)
    ok, dev = verify_matrix_kraus_completeness(ops, atol=5e-9)
    assert ok, f"jitter CPTP dev={dev:.2e} (d={d}, s2={sigma_a2}, n_max={n_max})"


@pytest.mark.parametrize("d", [3, 4, 5])
@pytest.mark.parametrize("sigma_a2", [0.01, SIGMA_A2, 0.2])
def test_v2_composition_cptp(d, sigma_a2):
    """Pair jitter composed with control (x) target dephasing is CPTP
    (dephasing truncated at n_max = 10 per the v2 spec; jitter n_max = 10)."""
    ch = gate_channel_kraus_v2(d, 0, 1, ETA, sigma_a2, phi=0.0,
                               n_max_dephasing=10, n_max_jitter=10)
    two_deph = [np.kron(np.diag(a), np.diag(b))
                for a in ch['control_dephasing'] for b in ch['target_dephasing']]
    comp = [J @ D for J in ch['pair_jitter'] for D in two_deph]
    tot = sum(K.conj().T @ K for K in comp)
    dev = float(np.max(np.abs(tot - np.eye(d * d))))
    assert dev < 5e-8, f"v2 composition CPTP dev={dev:.2e} (d={d}, s2={sigma_a2})"


def test_jitter_channel_action_matches_multipliers():
    """Kraus action == dephasing by e^{-1/2 s2 (g_m - g_n)^2} in the G basis."""
    d = 3
    V1, gvals = _ms_jitter_eigensystem(d, 0.7, 0, 1)
    V = np.kron(V1, V1)
    M = np.exp(-0.5 * SIGMA_A2 * (gvals[:, None] - gvals[None, :]) ** 2)
    ops = ms_jitter_kraus(d, SIGMA_A2, phi=0.7, n_max=12)
    rng = np.random.default_rng(7)
    A = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
    rho = A @ A.conj().T
    rho /= np.trace(rho).real
    rho_out = sum(K @ rho @ K.conj().T for K in ops)
    rho_e = V.conj().T @ rho @ V
    rho_expect = V @ (M * rho_e) @ V.conj().T
    assert np.max(np.abs(rho_out - rho_expect)) < 1e-10


# ── (b) twirl == single sigma_phi x sigma_phi term ─────────────────────

@pytest.mark.parametrize("phi", [0.0, np.pi / 5, 1.9])
def test_active_subspace_twirl_is_single_two_body_term(phi):
    """Pinned: twirling the active-subspace restriction over the 16
    two-qubit Paulis gives p(sigma_phi x sigma_phi) = (1-e^{-s2/2})/2 and ZERO
    for every other non-identity Pauli."""
    d = 3
    J, K = 0, 1
    act = [J * d + J, J * d + K, K * d + J, K * d + K]
    ops = ms_jitter_kraus(d, SIGMA_A2, phi=phi, n_max=14)
    # exact channel restricted to the active 4-dim space
    P1 = [np.eye(2), np.array([[0, 1], [1, 0]], complex),
          np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex)]
    P2 = [np.kron(a, b) for a in P1 for b in P1]

    def N(r4):
        r9 = np.zeros((9, 9), complex)
        r9[np.ix_(act, act)] = r4
        out = sum(Kk @ r9 @ Kk.conj().T for Kk in ops)
        return out[np.ix_(act, act)]

    ptm = np.array([np.trace(Q @ N(Q)).real / 4.0 for Q in P2])
    chi = np.array([[np.trace(Pi @ Qj @ Pi.conj().T @ Qj.conj().T).real / 4.0
                     for Qj in P2] for Pi in P2])
    p = chi.T @ ptm / 16.0

    p_expect = jitter_two_body_rate(SIGMA_A2)
    # sigma_phi x sigma_phi in the qubit Pauli basis: (cos X + sin Y)^{x2}
    sp = np.cos(phi) * P1[1] + np.sin(phi) * P1[2]
    spp = np.kron(sp, sp)
    proj = np.array([abs(np.trace(Q.conj().T @ spp)) ** 2 / 16.0 for Q in P2])
    expected = np.zeros(16)
    expected[0] = 1.0 - p_expect
    expected += p_expect * proj
    assert np.max(np.abs(p - expected)) < 1e-10
    assert abs((1.0 - p[0]) - p_expect) < 1e-10


def test_two_body_rate_value():
    assert abs(jitter_two_body_rate(SIGMA_A2) - 0.011613) < 5e-7
    assert jitter_two_body_rate(0.0) == 0.0


# ── (d) sigma_a2 = 0 reductions ────────────────────────────────────────

def test_kraus_builder_bit_identical_at_zero_jitter():
    d = 3
    ch = gate_channel_kraus_v2(d, 0, 1, ETA, 0.0, n_max_dephasing=5)
    ref_c = control_qudit_kraus(d, 0, ETA, 5)
    ref_t = target_qudit_kraus(d, 0, 1, ETA, 5)
    ref_s = spectator_qudit_kraus(d, ETA, 5)
    for got, ref in [(ch['control_dephasing'], ref_c),
                     (ch['target_dephasing'], ref_t),
                     (ch['spectator_dephasing'], ref_s)]:
        assert len(got) == len(ref)
        for a, b in zip(got, ref):
            assert np.array_equal(a, b)  # bit-identical
    assert len(ch['pair_jitter']) == 1
    assert np.array_equal(ch['pair_jitter'][0], np.eye(d * d, dtype=complex))


def test_noise_fn_path_bit_identical_at_zero_jitter():
    """v2 noise fn at sigma_a2=0 == v1 noise fn, bitwise, same rng stream."""
    from src.simulation import (make_correlated_dephasing_noise_fn,
                                make_channel_v2_noise_fn)
    n, d = 4, 3
    gp = [(a, b, 0, 1) for a in range(n) for b in range(a + 1, n)]
    nf1 = make_correlated_dephasing_noise_fn(n, d, gp, ETA, n_max=5)
    nf2 = make_channel_v2_noise_fn(n, d, gp, ETA, sigma_a2=0.0, n_max=5)
    rng_state = np.random.default_rng(11)
    for trial in range(3):
        psi = rng_state.normal(size=d ** n) + 1j * rng_state.normal(size=d ** n)
        psi /= np.linalg.norm(psi)
        out1 = nf1(psi, np.random.default_rng(100 + trial))
        out2 = nf2(psi, np.random.default_rng(100 + trial))
        assert np.array_equal(out1, out2)


def test_mode_a_unitary_and_zero_jitter():
    """MODE A trajectories are norm-preserving; sigma_a2=0 leaves phases only
    (diagonal in the computational basis)."""
    from src.simulation import make_meth_mc_noise_fn
    n, d = 3, 3
    gp = [(a, b, 0, 1) for a in range(n) for b in range(a + 1, n)]
    nf = make_meth_mc_noise_fn(n, d, gp, ETA, sigma_a2=SIGMA_A2)
    rng = np.random.default_rng(5)
    psi = rng.normal(size=d ** n) + 1j * rng.normal(size=d ** n)
    psi /= np.linalg.norm(psi)
    out = nf(psi, np.random.default_rng(1))
    assert abs(np.linalg.norm(out) - 1.0) < 1e-12
    nf0 = make_meth_mc_noise_fn(n, d, gp, ETA, sigma_a2=0.0)
    out0 = nf0(psi, np.random.default_rng(2))
    assert np.max(np.abs(np.abs(out0) - np.abs(psi))) < 1e-12  # phases only


# ── (w) v2 weights pins ────────────────────────────────────────────────

def test_v2_weights_reduce_to_v1_at_zero_jitter():
    _, flat1 = compute_meth_group_weights(3, 4, distance=3, eta=ETA, n_max=2)
    _, flat2 = compute_meth_group_weights_v2(3, 4, distance=3, eta=ETA,
                                             sigma_a2=0.0, n_max=2)
    assert np.array_equal(flat1, flat2)  # exact, pinned


def test_v2_weights_normalized_and_shift_to_pair_products():
    d, n = 3, 4
    g1, flat1 = compute_meth_group_weights(d, n, distance=3, eta=ETA, n_max=2)
    g2, flat2 = compute_meth_group_weights_v2(d, n, distance=3, eta=ETA,
                                              sigma_a2=SIGMA_A2, n_max=2)
    assert len(g1) == len(g2)
    assert all(a.shape == b.shape for a, b in zip(g1, g2))
    assert abs(flat2.sum() - 1.0) < 1e-12
    assert np.all(flat2 >= 0)
    # the jitter projection adds mass on weight-2 products: the max relative
    # gain must sit in a weight-2 group (wires of length 2)
    from src.errors import ErrorModel
    grouped = ErrorModel(d=d, n_qudit=n, distance=3, closed=True
                         ).build_grouped(verbose=False)
    sizes = [g['matrices'].shape[0] for g in grouped]
    offs = np.concatenate([[0], np.cumsum(sizes)])
    ratio = flat2 / np.where(flat1 > 0, flat1, 1.0)
    top = int(np.argmax(ratio))
    gi = int(np.searchsorted(offs, top, side='right') - 1)
    assert len(grouped[gi]['wires']) == 2
