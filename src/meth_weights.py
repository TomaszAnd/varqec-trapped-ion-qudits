"""Channel weights of the detection operators (paper Sec. IV C / IV D).

The channel-weighted objective of the paper (Sec. IV D, "Optimization
procedure and certification") multiplies each operator's term of the
Knill-Laflamme loss by that operator's probability under the calibrated
channel of Eq. (gate-channel). This module computes those probabilities
LIVE from the Kraus operators of `src.correlated_noise` (paper App. B), so
the weights follow the channel definition automatically.

For each operator E_i of `ErrorModel(d, n_qudit, distance, closed=True)`
(the completed detection set of Sec. IV C: channel error basis
{Z_k, X_{k-1,k}} plus products and same-qudit closure) the weight is

    lambda_i  =  Prod_q  Sum_P  |<P, E_i^q>|^2 / d^2  *  p_spec(P)       (1)

where E_i^q is the q-th single-qudit tensor factor of E_i, P runs over the
single-qudit operator basis of `build_single_qudit_pauli_basis` and

    p_spec(P) = (1/d^2) Sum_a |Tr(P^dag K_a)|^2                           (2)

is the Pauli-twirl probability of P under the SPECTATOR dephasing channel
{K_a} = spectator_qudit_kraus(d, eta, n_max). Two approximations are built
into (1)-(2) and are not spelled out in the manuscript:

  * role approximation: every qudit is weighted with the spectator channel
    (f_k = k), although in one gate step the control and target qudits see
    the control/target couplings (build_per_role_weights computes all three
    role twirls; only "spectator" is lifted);
  * twirl: (2) is the diagonal of the chi-matrix of the dephasing channel in
    the basis P, i.e. the weight of P in the Pauli-twirled channel. Because
    the dephasing Kraus operators are diagonal, this is exact for the Z_k
    (they span all diagonal operators, Sec. IV C) and the X-type swaps pick
    up only their diagonal (identity-shadow) overlap.

Weights are normalised to Sum_i lambda_i = 1 per register. Only the
dephasing part of the channel carries weights (Sec. IV C: the angle
fluctuation term "contributes its operators to the basis but no weights of
its own"). The `_v2` functions at the end add the first-order Pauli-twirl
rate of the MS angle fluctuation on active-pair weight-2 products; that is
the variant of the Sec. IV C footnote, used only when a caller passes
sigma_a2 > 0 (src/seed_race.py defaults to sigma_a2 = 0).

Consumers
---------
  src/seed_race.py                 training prior (compute_meth_group_weights_v2,
                                   sigma_a2 = 0 -> dephasing-only v1 weights)
  src/decoders/projective_decode.py correction ordering of the projective
                                   syndrome decoder (compute_meth_group_weights[_v2])
  src/errors.py                    weighted i.i.d. benchmark draw
                                   (hardware_benchmark_weights)
  scripts/meth_pauli_weights.py    standalone generator / inspection CLI
  tests/test_meth_weights_ext.py, test_weighted_benchmark.py,
  test_projective_decode.py, test_channel_v2.py, test_stratified_importance.py
"""
from __future__ import annotations
import numpy as np


def pauli_twirl_single_qudit(kraus_diags_or_mats, pauli_ops, d):
    """Pauli-twirl probabilities lambda_P = (1/d^2) Sum_a |Tr(P^dag K_a)|^2.

    Args:
        kraus_diags_or_mats: list of Kraus operators, each either a (d,)
            diagonal or a (d, d) matrix
        pauli_ops: list of (d, d) operators P (need not be unitary)
        d: qudit dimension

    Returns:
        (len(pauli_ops),) float array of twirl weights (Eq. (2) of the module
        docstring); they sum to 1 when pauli_ops is an orthogonal operator
        basis of norm sqrt(d).
    """
    weights = np.zeros(len(pauli_ops), dtype=float)
    for i, P in enumerate(pauli_ops):
        s = 0.0
        for K in kraus_diags_or_mats:
            if K.ndim == 1:
                t = np.sum(P.conj().diagonal() * K)
            else:
                t = np.trace(P.conj().T @ K)
            s += abs(t) ** 2
        weights[i] = s / (d ** 2)
    return weights


def build_single_qudit_pauli_basis(d):
    """Single-qudit operator set the weights are projected onto.

    [I] + channel error basis (Z_1..Z_{d-1}, X_01..X_{d-2,d-1}; paper
    Sec. IV C) + the same-qudit closure products of that basis
    (src.errors.close_error_basis). 13 operators for d=3.

    Returns:
        list of (label, (d, d) complex array) pairs; labels "I", "Z_k",
        "X_{k}{k+1}", "closure_i".
    """
    from src.errors import qudit_hardware_error_basis, close_error_basis
    Id = np.eye(d, dtype=complex)
    base = qudit_hardware_error_basis(d)
    cross = close_error_basis(base)
    ops = ([Id] + [np.asarray(o, dtype=complex) for o in base]
           + [np.asarray(o, dtype=complex) for o in cross])
    labels = (["I"] + [f"Z_{k}" for k in range(1, d)]
              + [f"X_{k}{k+1}" for k in range(d - 1)]
              + [f"closure_{i}" for i in range(len(cross))])
    return list(zip(labels, ops))


def build_per_role_weights(d, eta, n_max=2):
    """Single-qudit Pauli-twirl probabilities per gate role.

    Twirls the control (control level 0), target (control 0, target 1) and
    spectator dephasing Kraus families of paper App. B over the basis of
    build_single_qudit_pauli_basis. Only the "spectator" entry is used by the
    lift (role approximation, see module docstring); the other two are kept
    for inspection (scripts/meth_pauli_weights.py).

    Args:
        d: qudit dimension
        eta: dephasing parameter eta = exp(-sigma_p^2)
        n_max: Kraus truncation order of the dephasing series

    Returns:
        dict with "labels" (list of str), "ops" (list of (d, d) arrays) and
        "control", "target", "spectator": float arrays of twirl weights,
        aligned with "ops".
    """
    from src.correlated_noise import (control_qudit_kraus, target_qudit_kraus,
                                       spectator_qudit_kraus)
    pl = build_single_qudit_pauli_basis(d)
    pauli_ops = [op for _, op in pl]
    k_ctrl = control_qudit_kraus(d, control_level=0, eta=eta, n_max=n_max)
    k_tgt = target_qudit_kraus(d, control_level=0, target_level=1, eta=eta, n_max=n_max)
    k_spec = spectator_qudit_kraus(d, eta=eta, n_max=n_max)
    return {"labels": [lbl for lbl, _ in pl], "ops": pauli_ops,
            "control": pauli_twirl_single_qudit(k_ctrl, pauli_ops, d),
            "target": pauli_twirl_single_qudit(k_tgt, pauli_ops, d),
            "spectator": pauli_twirl_single_qudit(k_spec, pauli_ops, d)}


def _split_first_factor(M, d, r):
    """Split M (d^{r+1} x d^{r+1}) = A (d x d) kron Rest (d^r x d^r) via the
    rank-1 SVD of the factor-permuted reshape; exact for tensor-product M
    (the phase split between A and Rest is arbitrary, their product is not)."""
    dr = d ** r
    M_re = M.reshape(d, dr, d, dr).transpose(0, 2, 1, 3).reshape(d * d, dr * dr)
    U, S, Vh = np.linalg.svd(M_re)
    A = (np.sqrt(S[0]) * U[:, 0]).reshape(d, d)
    R = (np.sqrt(S[0]) * Vh[0, :]).reshape(dr, dr)
    return A, R


def _kron_chain(parts):
    """kron(parts[0], parts[1], ...)."""
    out = parts[0]
    for Ai in parts[1:]:
        out = np.kron(out, Ai)
    return out


def _factor_per_qudit(grouped_op_matrix, wires, n_qudit, d):
    """Factor a wire-grouped tensor-product operator into its n_qudit single-qudit
    d x d factors (identity off its support).

    `grouped_op_matrix` is the d^k x d^k matrix of a weight-k operator acting
    on `wires` (the wire-grouped storage of src.errors.ErrorModel, paper
    App. D); it is split by recursive rank-1 SVD (_split_first_factor), which
    is exact for any k >= 2 because every detection operator is a tensor
    product of single-qudit operators. The result is validated against the
    input in the given wire order and, failing that, in reversed order (the
    convention accepted by an earlier weight-2-only implementation).

    Returns:
        list of n_qudit (d, d) complex arrays.
    """
    Id = np.eye(d, dtype=complex)
    factors = [Id.copy() for _ in range(n_qudit)]
    if len(wires) == 0:
        return factors
    M = np.asarray(grouped_op_matrix).astype(complex)
    if len(wires) == 1:
        factors[wires[0]] = M
        return factors
    k = len(wires)
    parts, R = [], M
    for i in range(k - 1):
        A, R = _split_first_factor(R, d, k - 1 - i)
        parts.append(A)
    parts.append(R)
    # wire-order check; fallback = reversed convention (kept from the
    # weight-2 implementation, which accepted B,A when A,B failed)
    if not np.allclose(M, _kron_chain(parts), atol=1e-8) and \
       np.allclose(M, _kron_chain(parts[::-1]), atol=1e-8):
        parts = parts[::-1]
    for w, A in zip(wires, parts):
        factors[w] = A
    return factors


def _pauli_proj(M, pauli_ops, d):
    """|<P, M>|^2 / d^2 for each P in pauli_ops (Hilbert-Schmidt overlap, (len,) array)."""
    return np.array([abs(np.trace(P.conj().T @ M)) ** 2 / (d ** 2)
                     for P in pauli_ops], dtype=float)


def _lift_unnorm(E_grouped, d, n_qudit, per_role):
    """Unnormalized lambda_i = Prod_q Sum_k |<P_k, E_i^q>|^2 * w_spec[k]
    (Eq. (1) of the module docstring, before normalisation).

    Args:
        E_grouped: wire-grouped detection set (list of dicts with 'wires' and
            'matrices' of shape (n_ops, d^k, d^k)), as built by
            ErrorModel.build_grouped
        d, n_qudit: register geometry
        per_role: output of build_per_role_weights (uses "ops" and "spectator")

    Returns:
        (N,) float array in the flat order of E_grouped; an operator whose
        factor has zero projection on every basis element gets weight 0.
    """
    pauli_ops = per_role["ops"]
    w_spec = per_role["spectator"]
    n_total = sum(int(g['matrices'].shape[0]) for g in E_grouped)
    weights = np.zeros(n_total, dtype=float)
    flat = 0
    for g in E_grouped:
        wires = g['wires']
        for e in range(g['matrices'].shape[0]):
            factors = _factor_per_qudit(g['matrices'][e], wires, n_qudit, d)
            logw, valid = 0.0, True
            for q in range(n_qudit):
                p_q = float(np.sum(_pauli_proj(factors[q], pauli_ops, d) * w_spec))
                if p_q <= 0:
                    valid = False
                    break
                logw += np.log(p_q)
            weights[flat] = float(np.exp(logw)) if valid else 0.0
            flat += 1
    return weights


def lift_to_flat_weights(E_grouped, d, n_qudit, per_role):
    """lambda_i = Prod_q Sum_k |<P_k, E_i^q>|^2 * w_spec[k], normalised Sum=1.

    See _lift_unnorm for arguments. Returns the (N,) flat weight vector of
    the detection set, aligned with the concatenation of E_grouped.
    """
    weights = _lift_unnorm(E_grouped, d, n_qudit, per_role)
    s = weights.sum()
    return weights / s if s > 0 else weights


def hardware_benchmark_weights(d, eta=0.9296, kind="meth-spectator", n_max=2):
    """Per-op draw distribution over the 2(d-1) BENCHMARK hardware errors
    (qudit_hardware_error_basis order: Z_1..Z_{d-1}, X_01..X_{d-2,d-1}),
    derived from the spectator dephasing Kraus channel at eta.

    Used by src.errors for a channel-weighted variant of the i.i.d. benchmark
    of paper Sec. V A (the reported benchmark draws uniformly).

    kind="meth-spectator": normalized lambda(P) = (1/d^2) Sum_a |Tr(P^dag K_a)|^2
      -- the SAME projection the channel-weighted training prior uses
      (lift_to_flat_weights), restricted to the benchmark ops. Because the
      hardware ops act as identity on spectator levels, these projections
      carry a large identity-shadow component and are only mildly
      non-uniform at the calibrated point.

    kind="dephasing-exact": normalized lambda_perp(P), the projection of the
      TRACELESS part of each op -- the identity-shadow-free error mass each
      benchmark op captures from the (diagonal) physical channel. Strongly
      non-uniform; the swap ops keep only their spectator-level diagonal
      overlap.

    Calibration note (exact, all d): every benchmark op has Tr = d-2, so all
    ops share the same single-op average infidelity c_d = (4d-4)/(d^2+d)
    (paper Sec. V A) -- reweighting the draw does not change the injected
    error strength at fixed injection probability p.

    Returns:
        (2(d-1),) float array summing to 1.
    """
    from src.errors import qudit_hardware_error_basis
    from src.correlated_noise import spectator_qudit_kraus
    ops = [np.asarray(E, dtype=complex) for E in qudit_hardware_error_basis(d)]
    kraus = spectator_qudit_kraus(d, eta=eta, n_max=n_max)
    Id = np.eye(d)
    w = np.zeros(len(ops), dtype=float)
    for i, P in enumerate(ops):
        if kind == "dephasing-exact":
            P = P - (np.trace(P) / d) * Id
        elif kind != "meth-spectator":
            raise ValueError(f"unknown kind '{kind}'")
        s = 0.0
        for K in kraus:
            Km = np.diag(K) if np.asarray(K).ndim == 1 else np.asarray(K)
            s += abs(np.trace(P.conj().T @ Km)) ** 2
        w[i] = s / d ** 2
    return w / w.sum()


def compute_meth_group_weights(d, n_qudit, distance=3, eta=0.9296, n_max=2):
    """Dephasing-only channel weights of the completed detection set
    (the weights of the paper's primary, channel-weighted protocol).

    Builds ErrorModel(d, n_qudit, distance, closed=(distance >= 3)) and lifts
    the spectator twirl at eta (Eq. (1) of the module docstring).

    Args:
        d, n_qudit, distance: code parameters (distance 2 -> uncompleted basis)
        eta: dephasing parameter (calibrated 0.9296)
        n_max: Kraus truncation order

    Returns:
        (group_weights, flat_weights): group_weights is a list of float
        arrays aligned with the wire groups of ErrorModel.build_grouped, flat
        is their concatenation; both normalised so that flat sums to 1.
    """
    from src.errors import ErrorModel
    grouped = ErrorModel(d=d, n_qudit=n_qudit, distance=distance,
                         closed=(distance >= 3)).build_grouped(verbose=False)
    per_role = build_per_role_weights(d, eta, n_max)
    flat = lift_to_flat_weights(grouped, d, n_qudit, per_role)
    sizes = [g['matrices'].shape[0] for g in grouped]
    offs = np.concatenate([[0], np.cumsum(sizes)])
    group_weights = [np.asarray(flat[offs[i]:offs[i + 1]], dtype=float)
                     for i in range(len(sizes))]
    return group_weights, flat


# ═══════════════════════════════════════════════════════════════════════
# Weights including the MS angle-fluctuation term (paper Sec. IV C footnote)
# ═══════════════════════════════════════════════════════════════════════

def jitter_two_body_rate(sigma_a2):
    """Exact Pauli-twirl rate of the MS gate-angle fluctuation channel:
    p = (1 - e^{-sigma_a2/2}) / 2, the weight of the SINGLE
    sigma_phi (x) sigma_phi term of the twirled channel (exact PTM/chi
    inversion; angle-independent). 0.011613 at the calibrated sigma_a2 = 0.047.

    Derivation sketch: V(dtheta) = exp[-i (dtheta/4) S_phi^2] with
    S_phi^2 = 2 I + 2 sigma_phi (x) sigma_phi on the subspace where both
    qudits occupy the driven levels {j, k}, so there
    V = e^{-i dtheta/2} [cos(dtheta/2) I - i sin(dtheta/2) sigma_phi (x) sigma_phi]
    and the Gaussian average of sin^2(dtheta/2) is (1 - e^{-sigma_a2/2})/2.
    """
    return float((1.0 - np.exp(-0.5 * sigma_a2)) / 2.0)


def build_per_role_weights_v2(d, eta, sigma_a2, phi=0.0, level_j=0, level_k=1,
                              n_max=2):
    """build_per_role_weights extended with the angle-fluctuation twirl data:
    the two-body rate p = (1-e^{-sigma_a2/2})/2 and the sigma_phi operator
    whose product sigma_phi (x) sigma_phi carries 100% of the twirled
    angle-fluctuation error weight. At sigma_a2 = 0 the extension is inert
    (p = 0) and the dephasing entries are the UNCHANGED v1 values.

    Returns:
        the build_per_role_weights dict plus keys "jitter_p" (float),
        "jitter_sigma_phi" ((d, d) array) and "jitter_meta" (dict).
    """
    per_role = build_per_role_weights(d, eta, n_max)
    sig = np.zeros((d, d), dtype=complex)
    sig[level_j, level_k] = np.exp(-1j * phi)
    sig[level_k, level_j] = np.exp(1j * phi)
    per_role["jitter_p"] = jitter_two_body_rate(sigma_a2)
    per_role["jitter_sigma_phi"] = sig
    per_role["jitter_meta"] = {"sigma_a2": float(sigma_a2), "phi": float(phi),
                               "levels": (int(level_j), int(level_k))}
    return per_role


def lift_to_flat_weights_v2(E_grouped, d, n_qudit, per_role_v2, n_gates=None):
    """Weights including the angle-fluctuation term: first-order per-round
    error rates of dephasing + twirled angle fluctuation, normalised Sum=1.

    Every gate step applies dephasing to ALL n qudits (role approximation:
    spectator twirl, as in v1) and the angle-fluctuation term to its ACTIVE
    PAIR only. To first order the rates add over the n_gates all-to-all gate
    steps of one noise round:

        rate(E_i) = n_gates * (1-p) * lambda_v1_unnorm(E_i)
                  + p * proj_{sigma_phi}(A) * proj_{sigma_phi}(B)
                        * Prod_{q not in pair} p_spec(E_i^q)

    where the angle-fluctuation branch is nonzero only for weight-2 products
    on wires (q1, q2) -- their own (always-active, all-to-all) pair -- because
    sigma_phi is traceless, so identity factors project to zero. p is the
    exact twirl rate (angle-independent); proj is the same |<P, F>|^2/d^2
    projection the v1 lift uses. At p = 0 this reproduces v1 EXACTLY (the
    caller short-circuits to lift_to_flat_weights for bit-identity).

    Args:
        E_grouped, d, n_qudit: as _lift_unnorm
        per_role_v2: output of build_per_role_weights_v2
        n_gates: gate steps per round (default: all-to-all, n(n-1)/2)
    """
    p_jit = per_role_v2["jitter_p"]
    sig = per_role_v2["jitter_sigma_phi"]
    pauli_ops = per_role_v2["ops"]
    w_spec = per_role_v2["spectator"]
    if n_gates is None:
        n_gates = n_qudit * (n_qudit - 1) // 2

    lam_deph = _lift_unnorm(E_grouped, d, n_qudit, per_role_v2)
    weights = n_gates * (1.0 - p_jit) * lam_deph

    flat = 0
    for g in E_grouped:
        wires = g['wires']
        for e in range(g['matrices'].shape[0]):
            if len(wires) == 2:
                factors = _factor_per_qudit(g['matrices'][e], wires, n_qudit, d)
                jit = p_jit
                for q in range(n_qudit):
                    F = factors[q]
                    if q in wires:
                        jit *= abs(np.trace(sig.conj().T @ F)) ** 2 / d ** 2
                    else:
                        jit *= float(np.sum(_pauli_proj(F, pauli_ops, d) * w_spec))
                weights[flat] += jit
            flat += 1
    s = weights.sum()
    return weights / s if s > 0 else weights


def compute_meth_group_weights_v2(d, n_qudit, distance=3, eta=0.9296,
                                  sigma_a2=0.047, phi=0.0, n_max=2):
    """Channel weights including the MS angle-fluctuation term: the dephasing
    spectator-twirl weights of compute_meth_group_weights extended with the
    two-body sigma_phi (x) sigma_phi projection (rate (1-e^{-sigma_a2/2})/2
    = 0.011613 at sigma_a2=0.047) on active-pair weight-2 products. This is
    the variant of the paper's Sec. IV C footnote. Same return shape as
    compute_meth_group_weights. sigma_a2 = 0 delegates to the v1 path and
    reproduces its weights bit-identically (pinned test).
    """
    if sigma_a2 == 0.0:
        return compute_meth_group_weights(d, n_qudit, distance=distance,
                                          eta=eta, n_max=n_max)
    from src.errors import ErrorModel
    grouped = ErrorModel(d=d, n_qudit=n_qudit, distance=distance,
                         closed=(distance >= 3)).build_grouped(verbose=False)
    per_role = build_per_role_weights_v2(d, eta, sigma_a2, phi=phi, n_max=n_max)
    flat = lift_to_flat_weights_v2(grouped, d, n_qudit, per_role)
    sizes = [g['matrices'].shape[0] for g in grouped]
    offs = np.concatenate([[0], np.cumsum(sizes)])
    group_weights = [np.asarray(flat[offs[i]:offs[i + 1]], dtype=float)
                     for i in range(len(sizes))]
    return group_weights, flat
