"""
Logical-error-rate (LER) Monte Carlo, trajectory samplers of the calibrated
channel, and the lookup-table decoder.

This module implements the LER protocol of the paper, Sec. V A ("Logical
error rate protocol") and App. "LER benchmarking details and decoder
implementation":

  One shot:  1. draw a Haar-random logical state |L> = sum_k alpha_k |psi_k>
                (complex-Gaussian alpha, normalised = uniform on the code space);
             2. apply one round of noise, |L> -> noise_fn(|L>, rng);
             3. decode: try the identity and every single-qudit inverse
                E^dag, keep the candidate whose output is closest to the code
                space, project it onto the code space (lookup_table_decoder /
                the factored GEMM version);
             4. fidelity F = |<L|decoded>|^2; the shot FAILS if F < 0.5.
  LER = failed shots / shots.  (Wilson intervals are computed by the callers.)

Noise functions all have the signature noise_fn(state, rng) -> state on a
pure state vector of length d^n (qudit 0 = most significant axis, i.e.
state.reshape([d]*n)). Two families:

  * i.i.d. injection (the certification benchmark, Sec. V A):
    make_pauli_depolarizing_noise_fn -- each qudit independently, with
    probability p, suffers one operator drawn uniformly from the supplied
    single-qudit list (the channel error basis of Sec. IV C when called
    through src.errors.make_hardware_noise_fn, which is the production path).
  * trajectory unravelings of the calibrated gate-error channel of Sec. III,
    Eq. (gate-channel), one gate step per coupled pair:
      make_correlated_dephasing_noise_fn  dephasing only (A_{sigma_a^2} off),
      make_channel_v2_noise_fn            dephasing composed with the exact
                                          MS angle-fluctuation Kraus family,
      make_meth_mc_noise_fn               the same channel sampled as random
                                          phases / angles per trajectory
                                          (no Kraus averaging).
    All three agree in expectation with the exact density-matrix engine
    src/decoders/exact_channel.py, which is the arbiter for calibrated-
    channel numbers (Sec. V B); the samplers serve the n >= 8 Monte Carlo
    estimator and the validation tests.

Consumers
---------
  src/central_comparison.py        i.i.d. LER and weight-conditioned probes
                                   (simulate_ler_with_correction_factored)
  src/decoders/exact_channel.py    cross-validation of the samplers
  src/decoders/priors.py           prior matching make_pauli_depolarizing_noise_fn
  scripts/benchmark_ler*.py, scripts/fig6_panels.py, scripts/headline_lsweep.py,
  results/campaigns/**
  tests/test_logical_error_rate.py, test_factored_ler_batched.py,
  test_varqec_decoder.py, test_benchmarks.py, test_central_comparison.py,
  test_channel_v2.py, test_exact_channel.py, test_projective_decode.py
"""
import numpy as np
from typing import List, Callable, Dict, Optional, Tuple


def build_code_projector(code_states: np.ndarray) -> np.ndarray:
    """
    Build projector onto the code space.

    Args:
        code_states: (K, dim) array of orthonormal codeword vectors

    Returns:
        (dim, dim) projector matrix P = sum_k |psi_k><psi_k|
    """
    # P[i,j] = sum_k psi_k[i] * conj(psi_k[j]) = (C^T @ conj(C))[i,j]
    return code_states.T @ np.conj(code_states)


def apply_diagonal_noise(state: np.ndarray, kraus_diags: List[np.ndarray],
                          rng: np.random.Generator) -> np.ndarray:
    """
    Apply a diagonal noise channel to a state vector by sampling one Kraus operator.

    The channel N(rho) = sum_n E_n rho E_n† is unravelled on pure states by:
    1. Computing probability p_n = <psi|E_n†E_n|psi> = sum_k |E_n[k]|^2 |psi[k]|^2
    2. Sampling n with probability p_n
    3. Returning E_n|psi> / ||E_n|psi>||
    The expectation of |out><out| over the draw is exactly N(|psi><psi|).

    Args:
        state: (dim,) complex state vector
        kraus_diags: list of (dim,) diagonal Kraus operators
        rng: random number generator

    Returns:
        (dim,) noisy state vector (normalized)
    """
    probs = np.array([np.sum(np.abs(d)**2 * np.abs(state)**2) for d in kraus_diags])
    probs = probs / probs.sum()  # normalize for numerical safety

    idx = rng.choice(len(kraus_diags), p=probs)
    noisy = kraus_diags[idx] * state  # element-wise for diagonal
    norm = np.linalg.norm(noisy)
    if norm < 1e-15:
        return state  # fallback if collapsed
    return noisy / norm


def apply_pauli_noise(state: np.ndarray, error_ops: List[np.ndarray],
                       p_total: float, rng: np.random.Generator) -> np.ndarray:
    """
    Apply register-level discrete noise: with probability p_total, apply one
    operator drawn uniformly from error_ops; with probability 1-p_total, do nothing.

    Args:
        state: (dim,) state vector
        error_ops: list of (dim, dim) error operators on the full register (excluding identity)
        p_total: total error probability
        rng: random number generator

    Returns:
        (dim,) noisy state vector
    """
    if rng.random() > p_total:
        return state.copy()
    idx = rng.integers(0, len(error_ops))
    noisy = error_ops[idx] @ state
    norm = np.linalg.norm(noisy)
    return noisy / norm if norm > 1e-15 else state


def code_space_fidelity(state: np.ndarray, code_states: np.ndarray,
                         logical_coeffs: np.ndarray) -> float:
    """
    Compute fidelity of a (potentially noisy) state with the intended logical state.

    The logical state is |L> = sum_k alpha_k |psi_k>.
    Fidelity = |<L|state>|^2.

    Args:
        state: (dim,) state vector after noise
        code_states: (K, dim) codeword matrix
        logical_coeffs: (K,) coefficients alpha_k

    Returns:
        fidelity in [0, 1]
    """
    logical_state = logical_coeffs @ code_states  # (dim,)
    return float(np.abs(np.vdot(logical_state, state))**2)


def simulate_logical_error_rate(
    code_states: np.ndarray,
    noise_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    n_shots: int = 1000,
    fidelity_threshold: float = 0.5,
    seed: int = 42
) -> Dict[str, float]:
    """
    Monte Carlo logical error rate WITHOUT decoding (raw survival of the state).

    For each shot:
    1. Pick a Haar-random logical state |L> = sum_k alpha_k |psi_k>
    2. Apply noise: |L'> = noise_fn(|L>)
    3. Compute fidelity F = |<L|L'>|^2
    4. If F < fidelity_threshold, count as logical error

    Args:
        code_states: (K, dim) orthonormal codeword matrix
        noise_fn: function(state, rng) -> noisy_state
        n_shots: number of Monte Carlo samples
        fidelity_threshold: fidelity below this counts as a logical error
        seed: random seed

    Returns:
        dict with 'logical_error_rate', 'mean_fidelity', 'std_fidelity', 'n_shots'
    """
    K = code_states.shape[0]
    rng = np.random.default_rng(seed)
    fidelities = np.zeros(n_shots)

    for shot in range(n_shots):
        # Random logical state (Haar-random coefficients)
        coeffs = rng.standard_normal(K) + 1j * rng.standard_normal(K)
        coeffs /= np.linalg.norm(coeffs)

        logical_state = coeffs @ code_states  # (dim,)

        # Apply noise
        noisy_state = noise_fn(logical_state, rng)

        # Fidelity with original logical state
        fidelities[shot] = float(np.abs(np.vdot(logical_state, noisy_state))**2)

    n_errors = np.sum(fidelities < fidelity_threshold)
    return {
        'logical_error_rate': float(n_errors / n_shots),
        'mean_fidelity': float(np.mean(fidelities)),
        'std_fidelity': float(np.std(fidelities)),
        'n_shots': n_shots,
    }


def sweep_logical_error_rate(
    code_states: np.ndarray,
    noise_fn_factory: Callable[[float], Callable],
    physical_error_rates: List[float],
    n_shots: int = 1000,
    fidelity_threshold: float = 0.5,
    seed: int = 42,
    label: str = ""
) -> Dict[str, any]:
    """
    Sweep the undecoded logical error rate over physical error rates.

    Args:
        code_states: (K, dim) codeword matrix
        noise_fn_factory: function(p) -> noise_fn, where noise_fn(state, rng) -> noisy_state
        physical_error_rates: list of physical error rates to sweep
        n_shots: shots per error rate point
        fidelity_threshold: fidelity threshold for logical error
        seed: random seed
        label: label for this sweep

    Returns:
        dict with 'physical_rates', 'logical_rates', 'mean_fidelities', 'label'
    """
    logical_rates = []
    mean_fids = []

    for p in physical_error_rates:
        noise_fn = noise_fn_factory(p)
        result = simulate_logical_error_rate(
            code_states, noise_fn, n_shots=n_shots,
            fidelity_threshold=fidelity_threshold, seed=seed
        )
        logical_rates.append(result['logical_error_rate'])
        mean_fids.append(result['mean_fidelity'])
        if label:
            print(f"  {label}: p={p:.4f} -> LER={result['logical_error_rate']:.4f}, "
                  f"F_mean={result['mean_fidelity']:.4f}")

    return {
        'physical_rates': physical_error_rates,
        'logical_rates': logical_rates,
        'mean_fidelities': mean_fids,
        'label': label,
    }


# === Noise channel factories ===

def make_pauli_depolarizing_noise_fn(single_qudit_errors: List[np.ndarray],
                                     n_qudits: int, dim_qudit: int, p: float):
    """
    i.i.d. per-qudit discrete error injection (paper Sec. V A, certification benchmark).

    For each qudit independently:
    - With probability (1-p): do nothing
    - With probability p: apply one operator drawn uniformly from
      single_qudit_errors to that qudit

    With single_qudit_errors = the channel error basis of Sec. IV C this is
    the i.i.d. benchmark of the paper (src.errors.make_hardware_noise_fn is
    the equivalent production path; src.decoders.priors builds the matching
    prior). The per-qudit, per-gate error strength is c_d p with
    c_d = (4d-4)/(d^2+d) (Sec. V A).

    Args:
        single_qudit_errors: list of single-qudit (d x d) error matrices
        n_qudits: number of qudits
        dim_qudit: local qudit dimension
        p: per-qudit error probability

    Returns:
        noise_fn(state, rng) -> noisy_state (renormalised)
    """
    d = dim_qudit
    n_errors = len(single_qudit_errors)

    def noise_fn(state, rng):
        s = state.copy()
        for q in range(n_qudits):
            if rng.random() < p:
                idx = rng.integers(0, n_errors)
                E_local = single_qudit_errors[idx]
                shape = [d] * n_qudits
                s_tensor = s.reshape(shape)
                s_tensor = np.tensordot(E_local, s_tensor, axes=([1], [q]))
                s_tensor = np.moveaxis(s_tensor, 0, q)
                s = s_tensor.reshape(-1)
        norm = np.linalg.norm(s)
        if norm > 1e-15:
            s = s / norm
        return s

    return noise_fn


def make_correlated_dephasing_noise_fn(
    n_qudits: int, d: int,
    gate_pairs: List[Tuple[int, int, int, int]],
    eta: float, n_max: int = 5,
    noise_model: str = "physical"
):
    """
    Trajectory sampler of the dephasing part of the calibrated channel
    (paper Eq. (gate-channel) with the angle-fluctuation term A off).

    One call applies one noise round: for every gate step (qudit_a, qudit_b,
    ctrl_level, tgt_level) in gate_pairs, the product channel
        D^(c) on qudit_a (x) D^(t) on qudit_b (x) D^(sp) on every other qudit
    is unravelled on the pure state by sampling one Kraus index per qudit.

    Sampling: because every Kraus operator is diagonal and the channel is a
    tensor product, the joint index distribution
        p(n_1..n_N) = || (E_{n_1} (x) ... (x) E_{n_N}) |psi> ||^2
    factorises by the chain rule. Qudit q is sampled from its marginal
    computed on the state already multiplied by the chosen operators of
    qudits < q (left un-normalised, so the normalised marginal is exactly the
    conditional p(n_q | n_1..n_{q-1})); the state is renormalised once per
    gate step. The expectation of |out><out| is the exact channel output.

    Args:
        n_qudits: number of qudits
        d: qudit dimension
        gate_pairs: list of (qudit_a, qudit_b, ctrl_level, tgt_level);
            the all-to-all memory benchmark uses every pair with
            control = lower index, (ctrl_level, tgt_level) = (0, 1)
        eta: dephasing parameter eta = exp(-sigma_p^2)
        n_max: Kraus truncation order
        noise_model: "physical" (f_k = k) or "simplified" (uniform f = 2) control coupling

    Returns:
        noise_fn(state, rng) -> noisy_state
    """
    from src.correlated_noise import (
        control_qudit_kraus, control_qudit_kraus_simplified,
        target_qudit_kraus, spectator_qudit_kraus
    )

    if noise_model == "simplified":
        ctrl_fn = control_qudit_kraus_simplified
    else:
        ctrl_fn = control_qudit_kraus

    # Pre-build per-gate, per-qudit Kraus diag lists
    gate_kraus = []
    for qudit_a, qudit_b, ctrl_level, tgt_level in gate_pairs:
        per_qudit = []
        for q in range(n_qudits):
            if q == qudit_a:
                per_qudit.append(ctrl_fn(d, ctrl_level, eta, n_max))
            elif q == qudit_b:
                per_qudit.append(target_qudit_kraus(d, ctrl_level, tgt_level, eta, n_max))
            else:
                per_qudit.append(spectator_qudit_kraus(d, eta, n_max))
        gate_kraus.append(per_qudit)

    def noise_fn(state, rng):
        s = state.copy()
        for per_qudit in gate_kraus:
            # For each gate, sample one Kraus index per qudit; the full-system
            # Kraus operator is the kron of the chosen per-qudit diagonals.
            n_ops_per_qudit = [len(kd) for kd in per_qudit]

            # For diagonal Kraus operators of a product channel the qudits can
            # be sampled one after another (chain rule, see docstring).
            full_diag = np.ones(d**n_qudits, dtype=complex)
            for q in range(n_qudits):
                kd_list = per_qudit[q]
                # Marginal |amplitude|^2 of qudit q on the current (partially
                # multiplied, un-normalised) state
                shape = [d] * n_qudits
                s_tensor = np.abs(s.reshape(shape))**2
                # Sum over all other qudits to get marginal probs for qudit q
                axes_to_sum = tuple(i for i in range(n_qudits) if i != q)
                marginal = np.sum(s_tensor, axis=axes_to_sum)  # shape (d,)

                q_probs = np.array([np.sum(np.abs(kd)**2 * marginal) for kd in kd_list])
                q_probs = q_probs / q_probs.sum()

                idx = rng.choice(len(kd_list), p=q_probs)
                chosen_diag = kd_list[idx]

                # Apply to qudit q: multiply state by chosen_diag on qudit axis
                s_reshaped = s.reshape(shape)
                # Expand chosen_diag to broadcast on axis q
                expand_shape = [1] * n_qudits
                expand_shape[q] = d
                s_reshaped = s_reshaped * chosen_diag.reshape(expand_shape)
                s = s_reshaped.reshape(-1)

            norm = np.linalg.norm(s)
            if norm > 1e-15:
                s = s / norm
        return s

    return noise_fn


def _apply_single_qudit_matrix(state_tensor: np.ndarray, M: np.ndarray, q: int) -> np.ndarray:
    """Apply a d x d matrix on qudit axis q of a [d]*n state tensor."""
    out = np.tensordot(M, np.moveaxis(state_tensor, q, 0), axes=([1], [0]))
    return np.moveaxis(out, 0, q)


def make_channel_v2_noise_fn(
    n_qudits: int, d: int,
    gate_pairs: List[Tuple[int, int, int, int]],
    eta: float, sigma_a2: float = 0.0, phi: float = 0.0,
    n_max: int = 5, n_max_jitter: int = 10,
    noise_model: str = "physical"
):
    """Trajectory sampler of the full calibrated channel, paper Eq. (gate-channel):
    per gate step on a coupled pair, the correlated dephasing followed by the
    exact Gaussian-averaged MS angle-fluctuation Kraus family
    A_{sigma_a^2} (src.correlated_noise.ms_jitter_kraus).

    Same interface as make_correlated_dephasing_noise_fn plus (sigma_a2, phi).
    Per gate: the UNCHANGED per-qudit dephasing sampling (the loop body below
    is copied verbatim from make_correlated_dephasing_noise_fn -- keep them in
    sync), then one angle-fluctuation Kraus index sampled on the
    (control, target) pair in the sigma_phi product eigenbasis: rotate the
    pair into the eigenbasis with V1^dag (x) V1^dag, sample a diagonal Kraus
    operator from the pair marginal, multiply, rotate back. At sigma_a2 = 0
    the fluctuation block is skipped entirely, so the noise function
    reproduces the dephasing-only path BIT-IDENTICALLY (same rng stream, same
    arithmetic; pinned test).

    phi convention: the memory benchmark applies noise as a per-pair dose with
    no physical gate instance; DEFAULT phi = 0 (sigma_phi = X_01 on the
    driven levels (0, 1)). See the ms_jitter_kraus docstring.

    Args:
        n_qudits, d, gate_pairs, eta, n_max, noise_model: as
            make_correlated_dephasing_noise_fn
        sigma_a2: gate-angle fluctuation variance (calibrated 0.047; 0 = off)
        phi: MS phase angle (convention 0)
        n_max_jitter: Kraus truncation order of the angle-fluctuation family

    Returns:
        noise_fn(state, rng) -> noisy_state
    """
    from src.correlated_noise import (
        control_qudit_kraus, control_qudit_kraus_simplified,
        target_qudit_kraus, spectator_qudit_kraus,
        _ms_jitter_eigensystem, _kraus_diagonal
    )

    if noise_model == "simplified":
        ctrl_fn = control_qudit_kraus_simplified
    else:
        ctrl_fn = control_qudit_kraus

    # Pre-build per-gate, per-qudit dephasing Kraus diag lists (identical to
    # make_correlated_dephasing_noise_fn)
    gate_kraus = []
    for qudit_a, qudit_b, ctrl_level, tgt_level in gate_pairs:
        per_qudit = []
        for q in range(n_qudits):
            if q == qudit_a:
                per_qudit.append(ctrl_fn(d, ctrl_level, eta, n_max))
            elif q == qudit_b:
                per_qudit.append(target_qudit_kraus(d, ctrl_level, tgt_level, eta, n_max))
            else:
                per_qudit.append(spectator_qudit_kraus(d, eta, n_max))
        gate_kraus.append((per_qudit, qudit_a, qudit_b))

    # Pre-build the angle-fluctuation eigensystem + diagonal Kraus family
    # (shared by all gates: driven levels (0, 1), stated phi). None <=> off.
    jitter = None
    if sigma_a2 > 0.0:
        V1, gvals = _ms_jitter_eigensystem(d, phi, 0, 1)
        eta_a = float(np.exp(-sigma_a2))
        jit_diags = [dg for dg in _kraus_diagonal(d * d, gvals, eta_a, n_max_jitter)
                     if np.max(np.abs(dg)) > 0.0]
        jitter = (V1, V1.conj().T, jit_diags)

    def noise_fn(state, rng):
        s = state.copy()
        for per_qudit, qudit_a, qudit_b in gate_kraus:
            # --- dephasing sampling (verbatim copy, see docstring) ---
            full_diag = np.ones(d**n_qudits, dtype=complex)
            for q in range(n_qudits):
                kd_list = per_qudit[q]
                shape = [d] * n_qudits
                s_tensor = np.abs(s.reshape(shape))**2
                axes_to_sum = tuple(i for i in range(n_qudits) if i != q)
                marginal = np.sum(s_tensor, axis=axes_to_sum)  # shape (d,)

                q_probs = np.array([np.sum(np.abs(kd)**2 * marginal) for kd in kd_list])
                q_probs = q_probs / q_probs.sum()

                idx = rng.choice(len(kd_list), p=q_probs)
                chosen_diag = kd_list[idx]

                s_reshaped = s.reshape(shape)
                expand_shape = [1] * n_qudits
                expand_shape[q] = d
                s_reshaped = s_reshaped * chosen_diag.reshape(expand_shape)
                s = s_reshaped.reshape(-1)

            norm = np.linalg.norm(s)
            if norm > 1e-15:
                s = s / norm

            # --- pair angle fluctuation (skipped entirely at sigma_a2 = 0) ---
            if jitter is not None:
                V1, V1dag, jit_diags = jitter
                shape = [d] * n_qudits
                s_t = s.reshape(shape)
                s_t = _apply_single_qudit_matrix(s_t, V1dag, qudit_a)
                s_t = _apply_single_qudit_matrix(s_t, V1dag, qudit_b)
                # pair marginal in the eigenbasis, flattened (a-axis major,
                # matching the kron(V1, V1) order of the Kraus diagonals)
                probs_t = np.abs(s_t) ** 2
                axes_other = tuple(i for i in range(n_qudits)
                                   if i not in (qudit_a, qudit_b))
                marg = np.sum(probs_t, axis=axes_other)
                if qudit_a > qudit_b:
                    marg = marg.T
                marg = marg.ravel()
                j_probs = np.array([np.sum(np.abs(dg)**2 * marg) for dg in jit_diags])
                j_probs = j_probs / j_probs.sum()
                jdx = rng.choice(len(jit_diags), p=j_probs)
                dg = jit_diags[jdx].reshape(d, d)
                dga = dg if qudit_a < qudit_b else dg.T
                sh = [1] * n_qudits
                sh[qudit_a] = d
                sh[qudit_b] = d
                s_t = s_t * dga.reshape(sh)
                s_t = _apply_single_qudit_matrix(s_t, V1, qudit_a)
                s_t = _apply_single_qudit_matrix(s_t, V1, qudit_b)
                s = s_t.reshape(-1)
                norm = np.linalg.norm(s)
                if norm > 1e-15:
                    s = s / norm
        return s

    return noise_fn


def make_meth_mc_noise_fn(
    n_qudits: int, d: int,
    gate_pairs: List[Tuple[int, int, int, int]],
    eta: float, sigma_a2: float = 0.0, phi: float = 0.0,
    noise_model: str = "physical"
):
    """Monte-Carlo trajectory sampler of the stochastic-phase error model of
    Meth et al. as published: per-gate random phases plus per-instance
    gate-angle fluctuation, no twirl, no Kraus averaging. Its ensemble
    average is the calibrated channel of paper Eq. (gate-channel).

    Per gate step on a coupled pair:
      - each qudit q draws an independent phase Phi_q ~ N(0, sigma_p^2)
        (sigma_p^2 = -ln eta) applied as the deterministic level-coupling
        phase diag e^{i f_k Phi_q} of its role (control / target /
        spectator; same coupling vectors as the Kraus channel of paper
        App. B, whose Gaussian average it therefore reproduces exactly);
      - the pair draws dtheta ~ N(0, sigma_a2) and applies the exact
        fluctuation unitary V(dtheta) = exp[-i (dtheta/4) S_phi^2], i.e.
        eigenphases e^{-i dtheta g_m} in the sigma_phi product eigenbasis
        (theta-independent, paper Sec. III); Gaussian average = ms_jitter_kraus.

    Single-qudit-gate angle fluctuation (variance sigma_a2/10, paper Sec. III)
    is omitted: the memory benchmark contains no single-qudit gate instances.
    All maps are unitary per trajectory, so no renormalization is needed.
    Used to cross-validate the Kraus samplers and the exact engine
    (src/decoders/exact_channel.py).

    Args:
        n_qudits, d, gate_pairs, eta, noise_model: as make_correlated_dephasing_noise_fn
        sigma_a2: gate-angle fluctuation variance (0 = off)
        phi: MS phase angle (convention 0)

    Returns:
        noise_fn(state, rng) -> noisy_state
    """
    from src.correlated_noise import _ms_jitter_eigensystem

    sigma_p2 = -np.log(eta)

    # Per-gate, per-qudit coupling vectors f (match the App. B Kraus roles)
    gate_couplings = []
    for qudit_a, qudit_b, ctrl_level, tgt_level in gate_pairs:
        per_qudit = []
        for q in range(n_qudits):
            if q == qudit_a:
                if noise_model == "simplified":
                    f = np.full(d, 2.0)
                else:
                    f = np.arange(d, dtype=float)
                f[ctrl_level] = 0.0
            elif q == qudit_b:
                f = np.full(d, 3.0)
                f[ctrl_level] = 1.0
                f[tgt_level] = 2.0
            else:
                f = np.arange(d, dtype=float)
            per_qudit.append(f)
        gate_couplings.append((per_qudit, qudit_a, qudit_b))

    V1, gvals = _ms_jitter_eigensystem(d, phi, 0, 1)
    V1dag = V1.conj().T
    gpair = gvals.reshape(d, d)

    def noise_fn(state, rng):
        shape = [d] * n_qudits
        s_t = state.reshape(shape).copy()
        for per_qudit, qudit_a, qudit_b in gate_couplings:
            # stochastic phases, one independent draw per qudit role
            for q in range(n_qudits):
                Phi = rng.normal(0.0, np.sqrt(sigma_p2))
                ph = np.exp(1j * per_qudit[q] * Phi)
                sh = [1] * n_qudits
                sh[q] = d
                s_t = s_t * ph.reshape(sh)
            # gate-angle fluctuation unitary on the pair
            if sigma_a2 > 0.0:
                dtheta = rng.normal(0.0, np.sqrt(sigma_a2))
                s_t = _apply_single_qudit_matrix(s_t, V1dag, qudit_a)
                s_t = _apply_single_qudit_matrix(s_t, V1dag, qudit_b)
                phase = np.exp(-1j * dtheta * gpair)
                dga = phase if qudit_a < qudit_b else phase.T
                sh = [1] * n_qudits
                sh[qudit_a] = d
                sh[qudit_b] = d
                s_t = s_t * dga.reshape(sh)
                s_t = _apply_single_qudit_matrix(s_t, V1, qudit_a)
                s_t = _apply_single_qudit_matrix(s_t, V1, qudit_b)
        return s_t.reshape(-1)

    return noise_fn


# === Detection and correction simulation functions ===

def simulate_ler_with_detection(
    code_states: np.ndarray,
    noise_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    n_shots: int = 1000,
    detection_threshold: float = 0.1,
    seed: int = 42
) -> Dict[str, float]:
    """
    Detection-and-postselection protocol for distance-2 codes.

    Apply noise, project onto the code space; if the projection norm^2 is
    below detection_threshold the error is flagged and the shot discarded.
    For accepted shots the fidelity with the original logical state is
    recorded (a shot fails at fidelity < 0.5). The reported d = 2 panels of
    the paper use the correction protocol instead (App. "LER benchmarking
    details", "Distance-2 codes"); this function serves the postselection
    cross-check.

    Args:
        code_states: (K, dim) orthonormal codeword matrix
        noise_fn: function(state, rng) -> noisy_state
        n_shots: number of Monte Carlo samples
        detection_threshold: projection norm^2 below this = detected
        seed: random seed

    Returns:
        dict with 'detected_fraction', 'undetected_error_rate',
        'post_selected_fidelity', 'mean_raw_fidelity', 'n_shots'
    """
    from src.simulation import detection_decoder

    K = code_states.shape[0]
    rng = np.random.default_rng(seed)

    n_detected = 0
    post_selected_fidelities = []
    raw_fidelities = []

    for shot in range(n_shots):
        coeffs = rng.standard_normal(K) + 1j * rng.standard_normal(K)
        coeffs /= np.linalg.norm(coeffs)
        logical_state = coeffs @ code_states

        noisy_state = noise_fn(logical_state, rng)

        # Raw fidelity (no decoding)
        raw_fid = float(np.abs(np.vdot(logical_state, noisy_state))**2)
        raw_fidelities.append(raw_fid)

        # Detection decoder
        decoded, detected = detection_decoder(noisy_state, code_states, detection_threshold)

        if detected:
            n_detected += 1
        else:
            fid = float(np.abs(np.vdot(logical_state, decoded))**2)
            post_selected_fidelities.append(fid)

    detected_frac = n_detected / n_shots
    if post_selected_fidelities:
        ps_fid = float(np.mean(post_selected_fidelities))
        ps_err = float(np.mean([f < 0.5 for f in post_selected_fidelities]))
    else:
        ps_fid = 0.0
        ps_err = 1.0

    return {
        'detected_fraction': detected_frac,
        'undetected_error_rate': ps_err,
        'post_selected_fidelity': ps_fid,
        'mean_raw_fidelity': float(np.mean(raw_fidelities)),
        'n_shots': n_shots,
    }


def simulate_ler_with_correction(
    code_states: np.ndarray,
    noise_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    error_ops: List[np.ndarray],
    n_shots: int = 1000,
    seed: int = 42
) -> Dict[str, float]:
    """
    LER protocol of paper Sec. V A with the dense lookup-table decoder.

    Per shot: Haar-random logical state, noise, lookup_table_decoder over
    error_ops (dense d^n x d^n correction operators), fidelity with the
    injected state; the shot fails at fidelity < 0.5. For registers where
    dense operators are infeasible use simulate_ler_with_correction_factored
    (same protocol, same statistics).

    Args:
        code_states: (K, dim) orthonormal codeword matrix
        noise_fn: function(state, rng) -> noisy_state
        error_ops: list of candidate correctable error operators (dim x dim);
            the decoder tries E^dag for each (include the identity)
        n_shots: number of Monte Carlo samples
        seed: random seed

    Returns:
        dict with 'logical_error_rate', 'mean_fidelity', 'mean_raw_fidelity', 'n_shots'
    """
    from src.simulation import lookup_table_decoder

    K = code_states.shape[0]
    rng = np.random.default_rng(seed)

    corrected_fidelities = []
    raw_fidelities = []

    for shot in range(n_shots):
        coeffs = rng.standard_normal(K) + 1j * rng.standard_normal(K)
        coeffs /= np.linalg.norm(coeffs)
        logical_state = coeffs @ code_states

        noisy_state = noise_fn(logical_state, rng)

        raw_fid = float(np.abs(np.vdot(logical_state, noisy_state))**2)
        raw_fidelities.append(raw_fid)

        corrected = lookup_table_decoder(noisy_state, code_states, error_ops)
        fid = float(np.abs(np.vdot(logical_state, corrected))**2)
        corrected_fidelities.append(fid)

    return {
        'logical_error_rate': float(np.mean([f < 0.5 for f in corrected_fidelities])),
        'mean_fidelity': float(np.mean(corrected_fidelities)),
        'mean_raw_fidelity': float(np.mean(raw_fidelities)),
        'n_shots': n_shots,
    }


# ═══════════════════════════════════════════════════════════════════════
# Decoders (paper Sec. V A, App. "LER benchmarking details and decoder
# implementation")
# ═══════════════════════════════════════════════════════════════════════

def _build_decoder_projector(code_states: np.ndarray) -> np.ndarray:
    """Build projector P = sum_k |psi_k><psi_k|."""
    return code_states.T @ np.conj(code_states)


def projection_decoder(noisy_state: np.ndarray,
                       code_states: np.ndarray) -> np.ndarray:
    """Project the noisy state onto the code space and renormalise (no correction).

    Returns the input unchanged if the projection vanishes."""
    P = _build_decoder_projector(code_states)
    projected = P @ noisy_state
    norm = np.linalg.norm(projected)
    if norm < 1e-15:
        return noisy_state
    return projected / norm


def detection_decoder(noisy_state: np.ndarray,
                      code_states: np.ndarray,
                      detection_threshold: float = 0.1
                      ) -> Tuple[Optional[np.ndarray], bool]:
    """Distance-2 detection: project onto the code space; flag the shot
    (return (None, True)) if the projection norm^2 is below threshold,
    otherwise return (normalised projection, False)."""
    P = _build_decoder_projector(code_states)
    projected = P @ noisy_state
    proj_norm_sq = np.real(np.vdot(projected, projected))
    if proj_norm_sq < detection_threshold:
        return None, True
    return projected / np.sqrt(proj_norm_sq), False


def nearest_codeword_decoder(noisy_state: np.ndarray,
                             code_states: np.ndarray) -> np.ndarray:
    """Return the single codeword with maximum overlap (hard decision; reference only)."""
    overlaps = np.abs(np.conj(code_states) @ noisy_state)**2
    return code_states[np.argmax(overlaps)].copy()


def lookup_table_decoder(noisy_state: np.ndarray,
                         code_states: np.ndarray,
                         error_ops: List[np.ndarray]) -> np.ndarray:
    """Lookup-table decoder of paper Sec. V A (dense version).

    For every candidate E in error_ops the score is the code-space weight of
    the corrected state,
        score(E) = sum_k |<psi_k| E^dag |noisy>|^2 = || P E^dag |noisy> ||^2,
    computed as sum_k |<E psi_k | noisy>|^2. The candidate with the largest
    score (first maximum on ties) is applied as E^dag and the result is
    projected onto the code space and renormalised. The decoder uses no noise
    prior: every candidate is treated as equally likely.

    Args:
        noisy_state: (dim,) state vector
        code_states: (K, dim) orthonormal codeword matrix
        error_ops: list of (dim, dim) candidate error operators (include the
            identity so that "do nothing" is a candidate)

    Returns:
        (dim,) decoded, normalised state (the input if error_ops is empty)
    """
    K = code_states.shape[0]
    best_overlap = -1.0
    best_correction = None
    for E in error_ops:
        error_states = (E @ code_states.T).T
        overlap = sum(np.abs(np.vdot(error_states[k], noisy_state))**2 for k in range(K))
        if overlap > best_overlap:
            best_overlap = overlap
            best_correction = E
    if best_correction is None:
        return noisy_state
    corrected = best_correction.conj().T @ noisy_state
    return projection_decoder(corrected, code_states)


# Alias kept for callers that use the syndrome-decoder name.
syndrome_based_decoder = lookup_table_decoder


# ═══════════════════════════════════════════════════════════════════════
# Factored LER simulation (for n >= 7 where dense E_corr is infeasible)
# ═══════════════════════════════════════════════════════════════════════

def _apply_factored_op(state, op, q, n_qudit, d):
    """Apply a d×d operator to qudit q of an n-qudit state vector."""
    shape = tuple([d] * n_qudit)
    s = state.reshape(shape)
    s = np.tensordot(op, s, axes=([1], [q]))
    s = np.moveaxis(s, 0, q)
    return s.reshape(-1)


def simulate_ler_with_correction_factored(
    code_states, noise_fn, single_errors, n_qudit, d,
    n_shots=2000, seed=42, batch=1024
):
    """
    LER protocol of paper Sec. V A with the factored single-qudit lookup-table
    decoder (the path every distance-3 benchmark of the paper runs on; App.
    "Decoder execution").

    Avoids building dense d^n x d^n correction operators AND the dense
    projector. Uses overlap with individual codewords instead:
      overlap = sum_k |<psi_k|corrected>|^2

    The correction set is {I} + {E_q for q in [n], E in single_errors}, i.e.
    the identity and every single-qudit inverse of Sec. V A.

    Decoding uses the precomputed-corrected-codeword identity (the same trick
    as src/decoders/weighted_map.py): for correction E (applied as E dagger),
      score(E)  = sum_k |<psi_k|E^dag|noisy>|^2 = sum_k |<E psi_k|noisy>|^2
      coeffs_k  = <psi_k|E^dag|noisy>           = <E psi_k|noisy>
    so CES[i,k] = E_i psi_k is built once per call and the whole per-shot
    correction loop is one GEMM per shot batch; the winner's code-space
    projection coefficients come out of the same GEMM, so no correction is
    ever applied to a state. `batch` is a pure performance knob. RNG draws
    happen shot-by-shot in the original order (alpha, then noise_fn), so the
    random stream is identical to the un-batched implementation; per-shot
    outcomes are pinned against an inline reference by
    tests/test_factored_ler_batched.py. 7-12x faster than the per-shot loop.

    Args:
        code_states: (K, dim) orthonormal codeword matrix
        noise_fn: function(state, rng) -> noisy_state
        single_errors: list of single-qudit (d x d) candidate errors
        n_qudit, d: register geometry (dim = d**n_qudit)
        n_shots: number of Monte Carlo samples
        seed: random seed
        batch: shots per GEMM

    Returns dict with 'logical_error_rate' (fraction of shots with fidelity
    < 0.5), 'mean_fidelity', 'mean_raw_fidelity', 'n_shots'.
    """
    K, dim = code_states.shape

    # Correction stack, reference order: identity, then (q ascending, E in
    # single_errors order); ties in the score argmax resolve to the first
    # maximum, matching the strict-> replacement of the original loop.
    ces = [np.asarray(code_states)]
    for q in range(n_qudit):
        for E in single_errors:
            ces.append(np.stack([
                _apply_factored_op(code_states[k], E, q, n_qudit, d)
                for k in range(K)]))
    CES = np.stack(ces)                        # (n_corr, K, dim)
    n_corr = CES.shape[0]
    M = np.conj(CES.reshape(n_corr * K, dim))  # rows = <E_i psi_k|

    rng = np.random.default_rng(seed)
    logical_errors = 0
    fids_raw = np.empty(n_shots)
    fids_corr = np.empty(n_shots)

    for start in range(0, n_shots, batch):
        b = min(batch, n_shots - start)
        LOGICAL = np.empty((b, dim), dtype=complex)
        NOISY = np.empty((b, dim), dtype=complex)
        for j in range(b):
            alpha = rng.standard_normal(K) + 1j * rng.standard_normal(K)
            alpha /= np.linalg.norm(alpha)
            LOGICAL[j] = alpha @ code_states
            NOISY[j] = noise_fn(LOGICAL[j], rng)

        fids_raw[start:start + b] = np.abs(
            np.einsum('jd,jd->j', np.conj(LOGICAL), NOISY)) ** 2

        inner = (M @ NOISY.T).reshape(n_corr, K, b)   # <E_i psi_k|noisy_j>
        best = np.argmax(np.sum(np.abs(inner) ** 2, axis=1), axis=0)   # (b,)
        coeffs = inner[best, :, np.arange(b)]         # (b, K)
        projected = coeffs @ code_states              # (b, dim)
        norms = np.linalg.norm(projected, axis=1)
        ok = norms > 1e-10
        projected[ok] /= norms[ok, None]
        fc = np.abs(np.einsum('jd,jd->j', np.conj(LOGICAL), projected)) ** 2
        fids_corr[start:start + b] = fc
        logical_errors += int(np.sum(fc < 0.5))

    return {
        'logical_error_rate': float(logical_errors / n_shots),
        'mean_fidelity': float(np.mean(fids_corr)),
        'mean_raw_fidelity': float(np.mean(fids_raw)),
        'n_shots': n_shots,
    }
