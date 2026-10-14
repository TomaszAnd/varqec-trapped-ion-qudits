#!/usr/bin/env python3
"""Angle-fluctuation channel derivation: active-subspace error from MS
gate-angle jitter, numeric checks.

Derivation being verified:
  U_MS(phi,theta) = exp[-i(theta/4) S_phi^2],  S_phi = sigma_phi x I + I x sigma_phi,
  sigma_phi = cos(phi) X_jk + sin(phi) Y_jk  (eigenvalues +1,-1 on {j,k}, 0 else).
  Jitter theta -> theta + dtheta, dtheta ~ N(0, sigma_a^2):
    V(dtheta) = U(theta+dtheta) U(theta)^dag = exp[-i(dtheta/4) S_phi^2]   [theta-FREE]
  Gaussian average = dephasing in the eigenbasis of G = S_phi^2/4,
  eigenvalues g in {1 (4-dim active: 2 states), 1/4 (single-active), 0 (spectator)};
  coherence (m,n) multiplies by e^{-1/2 sigma_a^2 (g_m - g_n)^2}.
  Same Kraus family as _kraus_diagonal with couplings g.

Checks: (1) theta-independence of V; (2) Kraus family completeness + coherence
multipliers; (3) Pauli twirl on the active 2x2 subspace -> 16 Pauli probs,
phi-dependence scan, p_depol; (4) compare subspace_depolarizing_kraus product
channel vs derived twirl; CPTP checks; (5) MODE B composition CPTP;
(6) central-code trained-angle distribution; (7) damping arithmetic;
(8) MODE A timing pilot (1e4 jittered encoder evals at n=5, n=7).

Writes r7_t2_results.json. Runtime target < 10 min (pilot dominates).
Feeds the angle-fluctuation channel derivation of the paper.
"""
import os, sys, json, time
from math import factorial, sqrt, pi

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)
import src.jax_backend  # x64 first
import numpy as np
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'r7_t2_results.json')
res = {}

SIGMA_A2 = 0.047
SIGMA_P2 = 0.073
D = 3
J, K = 0, 1   # active pair


def sigma_phi(phi, d=D, j=J, k=K):
    m = np.zeros((d, d), complex)
    m[j, k] = np.exp(-1j * phi)
    m[k, j] = np.exp(1j * phi)
    return m


def S_op(phi):
    s = sigma_phi(phi)
    I = np.eye(D)
    return np.kron(s, I) + np.kron(I, s)


def U_ms_closed(phi, theta):
    from src.gates import MS_gate
    return np.asarray(MS_gate(phi, theta, J, K, D), complex)


# ── (1) gate form + theta-independence ─────────────────────────────────
phi0, th0, dth = 0.7, 2.31, 0.13
U_gen = expm(-1j * (th0 / 4) * S_op(phi0) @ S_op(phi0))
U_cl = U_ms_closed(phi0, th0)
dev_gate = float(np.max(np.abs(U_gen - U_cl)))
V1 = U_ms_closed(phi0, th0 + dth) @ U_cl.conj().T
V2 = U_ms_closed(phi0, 5.9 + dth) @ U_ms_closed(phi0, 5.9).conj().T
Vgen = expm(-1j * (dth / 4) * S_op(phi0) @ S_op(phi0))
res['gate_form'] = {
    'max_dev_closed_vs_exp': dev_gate,
    'V_theta_indep_dev': float(np.max(np.abs(V1 - V2))),
    'V_vs_exp_dev': float(np.max(np.abs(V1 - Vgen)))}
assert dev_gate < 1e-12 and res['gate_form']['V_theta_indep_dev'] < 1e-12

# ── (2) Gaussian-averaged channel + Kraus family ───────────────────────
G = S_op(phi0) @ S_op(phi0) / 4.0
gvals, gvecs = np.linalg.eigh(G)
res['G_eigenvalues'] = sorted(set(np.round(gvals, 10)))
# coherence multiplier matrix in G eigenbasis
M = np.exp(-0.5 * SIGMA_A2 * (gvals[:, None] - gvals[None, :]) ** 2)
res['coherence_multipliers'] = sorted(set(np.round(M.flatten(), 6)))


def kraus_G(sigma2, n_max=10):
    """_kraus_diagonal-form family with couplings g_m in the G eigenbasis."""
    s = sqrt(sigma2)
    ops = []
    for n in range(n_max + 1):
        diag = (gvals * s) ** n * np.exp(-0.5 * sigma2 * gvals ** 2) / sqrt(factorial(n))
        ops.append(gvecs @ np.diag(diag) @ gvecs.conj().T)
    return ops


KG = kraus_G(SIGMA_A2)
comp = sum(Kk.conj().T @ Kk for Kk in KG)
res['kraus_completeness_dev'] = float(np.max(np.abs(comp - np.eye(9))))
# channel action reproduces M: check on a random dm
rng = np.random.default_rng(7)
A = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
rho = A @ A.conj().T
rho /= np.trace(rho).real
rho_out = sum(Kk @ rho @ Kk.conj().T for Kk in KG)
rho_e = gvecs.conj().T @ rho @ gvecs
rho_expect = gvecs @ (M * rho_e) @ gvecs.conj().T
res['channel_vs_multiplier_dev'] = float(np.max(np.abs(rho_out - rho_expect)))

# average gate infidelity of the jitter channel on the full 9-dim space
Dfull = 9
sum_tr = float(np.sum(M))
res['jitter_avg_infid_full'] = 1 - (sum_tr + Dfull) / (Dfull ** 2 + Dfull)

# ── (3) Pauli twirl on the active 2-qubit subspace ─────────────────────
# embed: qubit basis {|j>,|k>} per qudit; active space indices in 9-dim
act_idx = [J * D + J, J * D + K, K * D + J, K * D + K]  # |jj>,|jk>,|kj>,|kk>
P1 = [np.eye(2), np.array([[0, 1], [1, 0]], complex),
      np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex)]
PAULI_NAMES = ['I', 'X', 'Y', 'Z']
P2 = [(a, b, np.kron(P1[a], P1[b])) for a in range(4) for b in range(4)]


def active_channel(phi, sigma2):
    """Exact jitter channel restricted to the 4-dim active space, as a function."""
    Sm = S_op(phi)
    Gm = (Sm @ Sm / 4.0)[np.ix_(act_idx, act_idx)]
    gv, gU = np.linalg.eigh(Gm)
    Mm = np.exp(-0.5 * sigma2 * (gv[:, None] - gv[None, :]) ** 2)

    def N(r):
        re = gU.conj().T @ r @ gU
        return gU @ (Mm * re) @ gU.conj().T
    return N


def twirl_probs(phi, sigma2):
    """Pauli-channel probabilities of the twirled channel (exact via PTM diag)."""
    N = active_channel(phi, sigma2)
    ptm = np.array([np.trace(Q @ N(Q)).real / 4.0 for _, _, Q in P2])
    chi = np.zeros((16, 16))
    for i, (_, _, Pi) in enumerate(P2):
        for j2, (_, _, Qj) in enumerate(P2):
            chi[i, j2] = np.trace(Pi @ Qj @ Pi.conj().T @ Qj.conj().T).real / 4.0
    p = chi.T @ ptm / 16.0
    return p


p0 = twirl_probs(0.0, SIGMA_A2)
res['twirl_phi0'] = {f'{PAULI_NAMES[a]}{PAULI_NAMES[b]}': float(p0[i])
                     for i, (a, b, _) in enumerate(P2) if abs(p0[i]) > 1e-12}
res['twirl_p_notI_phi0'] = float(1 - p0[0])
# phi scan (theta plays no role: V is theta-free)
scan = {}
for phi in [0.0, 0.4, pi / 3, 1.9, pi / 2]:
    pv = twirl_probs(phi, SIGMA_A2)
    scan[round(phi, 4)] = {'p_notI': float(1 - pv[0]),
                           'max_abs_diff_vs_phi0': float(np.max(np.abs(pv - p0)))}
res['twirl_phi_scan'] = scan
# negativity check (twirled probs must be >= 0 for a CP channel)
res['twirl_min_prob'] = float(np.min(p0))

# ── (4) compare with subspace_depolarizing_kraus product ───────────────
from src.correlated_noise import (subspace_depolarizing_kraus,
                                  verify_matrix_kraus_completeness,
                                  target_qudit_kraus, control_qudit_kraus,
                                  _compose_kraus_single_qudit,
                                  amplitude_damping_kraus)
# CPTP of the library op
for ptest in [0.0, 0.01, 0.3, 1.0]:
    ok, dev = verify_matrix_kraus_completeness(
        subspace_depolarizing_kraus(D, J, K, ptest))
    assert ok, (ptest, dev)
res['subspace_depol_cptp'] = True
# per-qudit product twirl vector: independent (1-p, p/3, p/3, p/3) each qudit
W = res['twirl_p_notI_phi0']
# match total non-identity weight: (1-p)^2 = p0[0] -> p_match
p_match = 1 - sqrt(p0[0])
prod = np.array([(1 - p_match if a == 0 else p_match / 3) *
                 (1 - p_match if b == 0 else p_match / 3)
                 for a, b, _ in P2])
res['per_qudit_p_match'] = float(p_match)
res['product_vs_derived_L1'] = float(np.sum(np.abs(prod - p0)))
res['derived_two_body_weight'] = float(sum(p0[i] for i, (a, b, _) in enumerate(P2)
                                           if a > 0 and b > 0))
res['product_two_body_weight'] = float(sum(prod[i] for i, (a, b, _) in enumerate(P2)
                                           if a > 0 and b > 0))

# ── (5) MODE B composition CPTP ────────────────────────────────────────
# per-qudit: subspace depol (at p_match) o target dephasing (sigma_p^2)
eta = float(np.exp(-SIGMA_P2))
deph_t = target_qudit_kraus(D, J, K, eta, n_max=6)
composed = _compose_kraus_single_qudit(deph_t, subspace_depolarizing_kraus(D, J, K, p_match))
ok_b, dev_b = verify_matrix_kraus_completeness(composed, atol=1e-8)
res['modeB_perqudit_cptp'] = {'ok': bool(ok_b), 'dev': float(dev_b)}
# two-qudit: derived twirled Pauli Kraus (embedded) o (control x target) dephasing
KP = []
for i, (a, b, Q) in enumerate(P2):
    if p0[i] <= 0:
        continue
    Qe = np.eye(9, dtype=complex)
    Qe[np.ix_(act_idx, act_idx)] = Q
    KP.append(sqrt(p0[i]) * Qe)
deph_c = control_qudit_kraus(D, J, eta, n_max=6)
two_deph = [np.kron(np.diag(a_), np.diag(b_)) for a_ in deph_c for b_ in deph_t]
comp2 = [Kp @ Dd for Kp in KP for Dd in two_deph]
tot = sum(Kk.conj().T @ Kk for Kk in comp2)
res['modeB_twoqudit_cptp_dev'] = float(np.max(np.abs(tot - np.eye(9))))

# ── (6) central-code trained angles ────────────────────────────────────────
npz = np.load(os.path.join(
    REPO, 'results/campaigns/r1_campaign/t1_codes/A_L2_s100.npz'),
    allow_pickle=True)
params = np.asarray(npz['params'])
n_q, d_ = int(npz['n_qudit']), int(npz['d'])
n_conn = n_q * (n_q - 1) // 2
nt = d_ - 1
ms_start = 4 * n_q * nt
ms_count = 2 * n_conn * nt
thetas = params[:, ms_start:ms_start + ms_count][:, 1::2].ravel()
phis = params[:, ms_start:ms_start + ms_count][:, 0::2].ravel()
res['headline_angles'] = {
    'npz': 'r1_campaign/t1_codes/A_L2_s100.npz', 'n_ms': int(thetas.size),
    'theta_raw_min_max': [float(thetas.min()), float(thetas.max())],
    'theta_mod4pi_min_max': [float(np.min(thetas % (4 * pi))),
                             float(np.max(thetas % (4 * pi)))],
    'phi_raw_min_max': [float(phis.min()), float(phis.max())]}
# p_depol over the trained set: theta-independent (exact); phi affects only the
# X/Y split within the twirl -> report range of p_notI over trained phis (subsample)
pn = [float(1 - twirl_probs(float(ph), SIGMA_A2)[0]) for ph in phis[:10]]
res['p_notI_over_trained_phis_first10'] = {'min': min(pn), 'max': max(pn)}

# ── (7) damping arithmetic ─────────────────────────────────────────────
T1 = 1.1
OMEGA = 2 * pi * 4e3            # BSB Rabi frequency quoted by Meth
t_pi = pi / OMEGA               # = 125 us
for name, t_gate in [('t_pi_125us', t_pi), ('crot_5pulses_625us', 5 * t_pi)]:
    gamma = 1 - np.exp(-t_gate / T1)
    Kd = amplitude_damping_kraus(D, float(gamma))
    str_sum = sum(abs(np.trace(Kk)) ** 2 for Kk in Kd)
    infid = 1 - (str_sum + D) / (D ** 2 + D)
    res.setdefault('damping', {})[name] = {
        't_gate_s': float(t_gate), 'gamma': float(gamma),
        'avg_gate_infid': float(infid),
        'ratio_dephasing_3.46e-2_over_this': float(3.46e-2 / infid)}
# reproduce the 3.46e-2 reference number (target-role dephasing, d=3, sigma_p^2)
f = np.array([1.0, 2.0, 3.0])
Mt = np.exp(-0.5 * SIGMA_P2 * (f[:, None] - f[None, :]) ** 2)
res['target_deph_avg_infid'] = float(1 - (np.sum(Mt) + D) / (D ** 2 + D))

json.dump(res, open(OUT, 'w'), indent=2, default=float)
print(json.dumps(res, indent=2, default=float))

# ── (8) MODE A timing pilot ────────────────────────────────────────────
import jax
import jax.numpy as jnp
from src.jax_backend import create_jax_encoder

pilot = {}
for tag, npz_path, n_pilot in [
        ('n5', 'results/campaigns/r1_campaign/t1_codes/A_L2_s100.npz', 10_000),
        ('n7', 'results/campaigns/r3_campaign/t2c_codes/d3_L2_s100.npz', 10_000)]:
    dat = np.load(os.path.join(REPO, npz_path), allow_pickle=True)
    pr = jnp.asarray(dat['params'])
    nq, dd = int(dat['n_qudit']), int(dat['d'])
    conns = [[i, j] for i in range(nq) for j in range(i + 1, nq)]
    enc, _, _ = create_jax_encoder(nq, dd, connections=conns, use_scan=True)
    ntr = dd - 1
    nc = len(conns)
    mss = 4 * nq * ntr
    msc = 2 * nc * ntr
    # jitter mask: sigma_a on MS thetas, sigma_a/sqrt(10) on single-qudit
    # rotation angles (XY alpha, Z theta), 0 on phases
    sig = np.zeros(pr.shape[1])
    sig[mss + 1:mss + msc:2] = sqrt(SIGMA_A2)
    for q in range(nq):                       # XY alphas (2 rounds)
        base = 4 * q * ntr
        sig[base + 1:base + 4 * ntr:2] = sqrt(SIGMA_A2 / 10)
    zs = mss + msc
    sig[zs:zs + nq * ntr] = sqrt(SIGMA_A2 / 10)
    sig = jnp.asarray(sig)

    @jax.jit
    def one_shot(key):
        dp = jax.random.normal(key, pr.shape) * sig[None, :]
        return enc(pr + dp, 0)

    batch = jax.jit(jax.vmap(one_shot))
    keys = jax.random.split(jax.random.PRNGKey(0), 500)
    _ = batch(keys).block_until_ready()      # compile
    t0 = time.time()
    done = 0
    key0 = jax.random.PRNGKey(1)
    while done < n_pilot:
        key0, sub = jax.random.split(key0)
        _ = batch(jax.random.split(sub, 500)).block_until_ready()
        done += 500
    wall = time.time() - t0
    pilot[tag] = {'npz': npz_path, 'shots': done, 'wall_s': round(wall, 2),
                  'us_per_encoder_eval': wall / done * 1e6,
                  'est_s_per_1e6_evals': wall / done * 1e6,
                  'note': 'one logical-state prep per eval; x3 for all codewords'}
    print(tag, pilot[tag], flush=True)

res['modeA_pilot'] = pilot
json.dump(res, open(OUT, 'w'), indent=2, default=float)
print('[t2] DONE')
