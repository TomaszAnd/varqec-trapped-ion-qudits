"""
Kraus operators of the calibrated trapped-ion gate-error channel.

This module implements the per-gate noise channel of the paper, Sec. III
("Hardware noise model"), Eq. (gate-channel):

    N_gate = A_{sigma_a^2} o ( D^(c) (x) D^(t) (x) [ (x)_{m not in {c,t}} D^(sp) ] )

where D^(c), D^(t), D^(sp) are the Gaussian-averaged stochastic-phase
dephasing channels on the control, target and spectator qudits of an
entangling gate (paper App. B, "Noise-model Kraus operators"; source:
Meth et al. arXiv:2310.12110v3, App. J), and A_{sigma_a^2} is the
entangling-gate angle fluctuation, which for the MS gate is an exact
dephasing channel in the eigenbasis of the collective generator S_phi^2
(paper Sec. III).

Conventions shared by every function below
------------------------------------------
* Stochastic phase Phi ~ N(0, sigma_p^2); the channel is parameterised by
  eta := exp(-sigma_p^2), so sigma_p^2 = -ln(eta). Calibrated value
  eta = 0.9296 (sigma_p^2 = 0.073).
* Each level k of a qudit acquires the phase f_k * Phi with an integer
  coupling f_k set by the qudit's role in the gate:
    control   f_k = k, except f = 0 on the control level ("physical");
              the "simplified" variant uses the uniform f = 2 of the literal
              published phase map of Meth et al. and is used by no reported
              number;
    target    f = 1 on the control level, 2 on the target level, 3 elsewhere;
    spectator f_k = k on every level.
* Gaussian averaging over Phi gives the Kraus family
    E_n = sum_k (f_k sigma_p)^n / sqrt(n!) * exp(-sigma_p^2 f_k^2 / 2) |k><k|,
  n = 0, 1, 2, ...; the coherence between two levels whose couplings differ
  by Delta_f decays as exp(-1/2 sigma_p^2 Delta_f^2). At eta = 0.9296 this is
  1.0 / 0.9642 / 0.8642 for Delta_f = 0, 1, 2. The code and the manuscript
  (App. B) agree on this normalisation.
* Angle fluctuation: dtheta ~ N(0, sigma_a^2), calibrated sigma_a^2 = 0.047;
  benchmark convention: driven levels (0, 1), phi = 0, control = the
  lower-indexed qudit of the pair.
* All dephasing Kraus operators are diagonal and are returned as 1-D arrays
  of their diagonals (length d); the angle-fluctuation and amplitude-damping
  operators are full matrices.

Layout
------
  1. Gaussian-averaged dephasing Kraus diagonals per role (App. B)
  2. Subspace depolarizing (legacy generic knob; not part of Eq. (gate-channel))
  3. Amplitude damping (App. B; idle error, not applied in any benchmark)
  4. MS gate-angle fluctuation channel (Sec. III, A_{sigma_a^2})
  5. Completeness checks
  6. Correlated error-set construction for the detection loss

Consumers
---------
  src/simulation.py             trajectory samplers of the channel
                                (make_correlated_dephasing_noise_fn,
                                make_channel_v2_noise_fn, make_meth_mc_noise_fn)
  src/decoders/exact_channel.py exact density-matrix evolution of the channel
  src/meth_weights.py           Pauli-twirl weights of the channel operators
  scripts/benchmark_ler_meth_kraus.py, notebooks/central_comparison.ipynb
  tests/test_trapped_ion_noise.py, test_noise_channels.py,
  test_noise_model_paper.py, test_channel_v2.py, test_exact_channel.py,
  test_dual_noise_models.py, test_benchmarks.py, test_detection_loss.py
"""
import numpy as np
from itertools import product as cartesian_product
from typing import List, Tuple
from math import factorial, sqrt


def _kraus_diagonal(d: int, coupling_strengths: np.ndarray, eta: float, n_max: int) -> List[np.ndarray]:
    """
    Kraus-operator diagonals of Gaussian phase noise with level couplings f_k.

    For each Kraus index n = 0, 1, ..., n_max (paper App. B):
        E_n[k,k] = (f_k * sqrt(-ln(eta)))^n * eta^{f_k^2 / 2} / sqrt(n!)
                 = (f_k sigma_p)^n / sqrt(n!) * exp(-sigma_p^2 f_k^2 / 2)

    where f_k is the coupling strength for level k. This is the Gaussian
    average of the random phase diag(e^{i f_k Phi}), Phi ~ N(0, sigma_p^2):
    the coherence between two levels with coupling difference Delta_f decays as
        E[e^{i Delta_f Phi}] = e^{-1/2 sigma_p^2 Delta_f^2}
    (the Gaussian characteristic function), which these operators reproduce
    exactly as n_max -> infinity, with Sum_n E_n^dag E_n = I (complete channel).
    A level with f_k = 0 appears only in E_0 (0^n / sqrt(n!) = delta_{n0}).

    Args:
        d: qudit dimension
        coupling_strengths: array of shape (d,) with coupling strength f_k for each level
        eta: dephasing parameter, eta = exp(-sigma_p^2), range (0, 1]
        n_max: maximum Kraus index (truncation order of the series in n)

    Returns:
        List of n_max + 1 complex arrays of shape (d,) (diagonals of Kraus operators)
    """
    assert 0 < eta <= 1, f"eta must be in (0, 1], got {eta}"
    assert len(coupling_strengths) == d

    sigma_p2 = -np.log(eta)  # sigma_p^2 = -ln(eta)
    # Gaussian-averaged dephasing, normalised as in paper App. B (and Meth
    # arXiv:2310.12110v3 App. J): the coherence between levels with coupling
    # difference Delta_f decays as e^{-1/2 sigma_p^2 Delta_f^2}. The two
    # factors below are the eta-form of that appendix,
    #   base^n / sqrt(n!) * decay = [f sqrt(-ln eta)]^n eta^{f^2/2} / sqrt(n!).
    base = coupling_strengths * np.sqrt(sigma_p2)  # f_k * sqrt(sigma_p^2)
    decay = eta ** (0.5 * coupling_strengths ** 2)  # eta^{f_k^2 / 2} = e^{-sigma_p^2 f_k^2 / 2}

    diags = []
    for n in range(n_max + 1):
        diag = (base ** n) * decay / sqrt(factorial(n))
        diags.append(diag.astype(complex))

    return diags


def control_qudit_kraus(d: int, control_level: int, eta: float, n_max: int = 5) -> List[np.ndarray]:
    """
    Kraus operator diagonals for dephasing on the CONTROL qudit of a gate.

    "Physical" (level-resolved) coupling, paper App. B "Noise on the control
    qudit": f_k = k for k != control_level, f = 0 on the control level, i.e.
        E_n^(c) = |i><i| delta_{n0} + sum_{j != i} (j sigma_p)^n / sqrt(n!)
                  e^{-sigma_p^2 j^2 / 2} |j><j|.
    For d=5, ctrl=0: f = {0, 1, 2, 3, 4}. This is the control channel D^(c)
    of Eq. (gate-channel) and the one every reported number uses.

    See also: control_qudit_kraus_simplified() for the literal published
    phase map (uniform f = 2).

    Args:
        d: qudit dimension
        control_level: level i (unaffected by noise)
        eta: dephasing parameter eta = exp(-sigma_p^2)
        n_max: truncation order

    Returns:
        List of n_max + 1 complex arrays of shape (d,) (Kraus operator diagonals)
    """
    coupling = np.arange(d, dtype=float)
    coupling[control_level] = 0.0
    return _kraus_diagonal(d, coupling, eta, n_max)


def control_qudit_kraus_simplified(d: int, control_level: int, eta: float, n_max: int = 5) -> List[np.ndarray]:
    """
    Kraus operator diagonals for dephasing on the CONTROL qudit, literal variant.

    "Simplified" coupling: the literal published phase map of Meth et al.
    (arXiv:2310.12110v3 App. J, quoted in paper App. B),
        c_i|i> + sum_{j!=i} c_j|j>  ->  c_i|i> + e^{i 2 Phi} sum_{j!=i} c_j|j>,
    in which ALL non-control levels pick up the SAME phase 2 Phi, i.e. uniform
    coupling f = 2 off the control level and f = 0 on it.
    For d=5, ctrl=0: f = {0, 2, 2, 2, 2}.

    Not used by any reported number: kept as the `noise_model="simplified"`
    comparison variant. The level-resolved control_qudit_kraus() is the
    production model.

    Args:
        d: qudit dimension
        control_level: level i (unaffected by noise)
        eta: dephasing parameter eta = exp(-sigma_p^2)
        n_max: truncation order

    Returns:
        List of n_max + 1 complex arrays of shape (d,) (Kraus operator diagonals)
    """
    coupling = np.full(d, 2.0)
    coupling[control_level] = 0.0
    return _kraus_diagonal(d, coupling, eta, n_max)


def target_qudit_kraus(d: int, control_level: int, target_level: int,
                       eta: float, n_max: int = 5) -> List[np.ndarray]:
    """
    Kraus operator diagonals for dephasing on the TARGET qudit of a gate.

    Paper App. B "Noise on the target qudit" (Meth et al. App. J phase map
    e^{i Phi} c_i|i> + e^{i 2 Phi} c_j|j> + e^{i 3 Phi} sum_{k!=i,j} c_k|k>),
    three tiers of coupling:
        f(k) = 1 if k = control_level (i)
        f(k) = 2 if k = target_level (j)
        f(k) = 3 if k not in {i, j} (spectator levels)
    This is the target channel D^(t) of Eq. (gate-channel).

    Args:
        d: qudit dimension
        control_level: level i
        target_level: level j
        eta: dephasing parameter eta = exp(-sigma_p^2)
        n_max: truncation order

    Returns:
        List of n_max + 1 complex arrays of shape (d,)
    """
    coupling = np.full(d, 3.0)  # default: spectator coupling = 3
    coupling[control_level] = 1.0
    coupling[target_level] = 2.0

    return _kraus_diagonal(d, coupling, eta, n_max)


def spectator_qudit_kraus(d: int, eta: float, n_max: int = 5) -> List[np.ndarray]:
    """
    Kraus operator diagonals for dephasing on SPECTATOR qudits (not addressed by the gate).

    Paper App. B "Spectator dephasing": level-resolved coupling f_k = k on
    every level of the qudit,
        D_n = sum_k [k sqrt(-ln eta)]^n eta^{k^2 / 2} / sqrt(n!) |k><k|.
    This is the spectator channel D^(sp) of Eq. (gate-channel), applied to
    every qudit of the register that is not the control or target of the step.

    Args:
        d: qudit dimension
        eta: dephasing parameter eta = exp(-sigma_p^2)
        n_max: truncation order

    Returns:
        List of n_max + 1 complex arrays of shape (d,)
    """
    coupling = np.arange(d, dtype=float)
    return _kraus_diagonal(d, coupling, eta, n_max)


def subspace_depolarizing_kraus(d: int, level_i: int, level_j: int,
                                 p: float) -> List[np.ndarray]:
    """
    Kraus operators for depolarizing noise restricted to the 2D subspace {|i>, |j>}.

    Generic model of a gate error acting within the driven two-level subspace
    (paper Sec. III: "within the driven two-level subspace the error acts as
    depolarizing noise"):
        N(rho) = (1-p)*rho + (p/3)*(X_{ij} rho X_{ij} + Y_{ij} rho Y_{ij} + Z_{ij} rho Z_{ij})
    with identity on the spectator levels.

    NOT part of the calibrated channel of Eq. (gate-channel): the calibrated
    model represents the driven-subspace error by the exact MS angle
    fluctuation channel (ms_jitter_kraus) instead. This function is a legacy
    generic robustness knob kept for combined_gate_kraus, the scripts and the
    tests; no reported number uses it.

    Args:
        d: qudit dimension
        level_i: first level of gate subspace
        level_j: second level of gate subspace
        p: depolarizing error probability (0 = no error, 1 = full depolarizing)

    Returns:
        List of 4 d x d complex matrices (K_0 = no-error, K_1,2,3 = Pauli errors in subspace)
    """
    assert 0 <= p <= 1, f"p must be in [0, 1], got {p}"
    assert 0 <= level_i < d and 0 <= level_j < d and level_i != level_j

    # Build projectors
    P_sub = np.zeros((d, d), dtype=complex)
    P_sub[level_i, level_i] = 1.0
    P_sub[level_j, level_j] = 1.0
    P_perp = np.eye(d, dtype=complex) - P_sub

    # No-error Kraus: sqrt(1-p) on subspace, identity on complement
    K0 = np.sqrt(1 - p) * P_sub + P_perp

    # Subspace Pauli X: |i><j| + |j><i|
    X_sub = np.zeros((d, d), dtype=complex)
    X_sub[level_i, level_j] = 1.0
    X_sub[level_j, level_i] = 1.0
    K1 = np.sqrt(p / 3) * X_sub

    # Subspace Pauli Y: -i|i><j| + i|j><i|
    Y_sub = np.zeros((d, d), dtype=complex)
    Y_sub[level_i, level_j] = -1j
    Y_sub[level_j, level_i] = 1j
    K2 = np.sqrt(p / 3) * Y_sub

    # Subspace Pauli Z: |i><i| - |j><j|
    Z_sub = np.zeros((d, d), dtype=complex)
    Z_sub[level_i, level_i] = 1.0
    Z_sub[level_j, level_j] = -1.0
    K3 = np.sqrt(p / 3) * Z_sub

    return [K0, K1, K2, K3]


def combined_gate_kraus(d: int, control_level: int, target_level: int,
                        eta_dephasing: float, p_depolarizing: float,
                        n_max_dephasing: int = 5, noise_model: str = "physical") -> dict:
    """
    Per-role Kraus operators for a gate: dephasing + generic subspace depolarizing.

    Legacy bundle (dephasing composed with subspace_depolarizing_kraus).
    The calibrated channel of the paper, Eq. (gate-channel), is instead
    gate_channel_kraus_v2 (dephasing composed with the exact MS angle
    fluctuation); this function is kept for the scripts and tests only and
    no reported number uses it.

    Returns Kraus operators for each qudit role:
    - control_qudit: dephasing Kraus ops (diagonal)
    - target_qudit: dephasing Kraus ops (diagonal)
    - gate_subspace: subspace depolarizing Kraus ops (NOT diagonal, full d x d)
    - spectator_qudit: dephasing Kraus ops (diagonal)

    The full channel per gate is the composition:
        N_gate = N_subspace_depol o N_dephasing

    Args:
        d: qudit dimension
        control_level: level i (control of the gate)
        target_level: level j (target of the gate)
        eta_dephasing: dephasing parameter eta = exp(-sigma_p^2)
        p_depolarizing: depolarizing probability in gate subspace
        n_max_dephasing: truncation order for dephasing Kraus series
        noise_model: "physical" (f_k=k, level-dependent) or "simplified" (uniform f=2)

    Returns:
        dict with keys:
          'control_dephasing': list of diagonal arrays (n_max+1 operators)
          'target_dephasing': list of diagonal arrays (n_max+1 operators)
          'subspace_depol': list of d x d matrices (4 operators: I, X, Y, Z in subspace)
          'spectator_dephasing': list of diagonal arrays (n_max+1 operators)
    """
    if noise_model == "simplified":
        ctrl_kraus = control_qudit_kraus_simplified(d, control_level, eta_dephasing, n_max_dephasing)
    elif noise_model == "physical":
        ctrl_kraus = control_qudit_kraus(d, control_level, eta_dephasing, n_max_dephasing)
    else:
        raise ValueError(f"noise_model must be 'physical' or 'simplified', got '{noise_model}'")

    return {
        'control_dephasing': ctrl_kraus,
        'target_dephasing': target_qudit_kraus(d, control_level, target_level, eta_dephasing, n_max_dephasing),
        'subspace_depol': subspace_depolarizing_kraus(d, control_level, target_level, p_depolarizing),
        'spectator_dephasing': spectator_qudit_kraus(d, eta_dephasing, n_max_dephasing),
    }


def amplitude_damping_kraus(d: int, gamma: float) -> List[np.ndarray]:
    """Qudit amplitude-damping Kraus operators (paper App. B).

        A_k = sum_{r=k}^{d-1} sqrt(C(r,k)) sqrt((1-gamma)^(r-k)) gamma^(k/2) |r-k><r|

    for k = 0..d-1. Idle decay toward |0> with per-event probability gamma
    (set by the experimental T1/T2 of Meth et al.). Exactly trace-preserving
    for every gamma in [0, 1]: column r of sum_k A_k^dag A_k is the binomial
    sum ((1-gamma) + gamma)^r = 1. At gamma=0 the channel is the identity.

    Per paper Sec. III the idle errors are too slow to compete with the gate
    errors over an encoder and the damping channel is applied in no benchmark;
    only its first-order lowering operators enter the extended detection set
    of the unweighted protocol variants (built in src/errors.py).

    Args:
        d: qudit dimension
        gamma: single-event damping probability in [0, 1]

    Returns:
        List of d full d x d complex matrices.
    """
    assert 0.0 <= gamma <= 1.0, f"gamma must be in [0, 1], got {gamma}"
    from math import comb
    ops = []
    for k in range(d):
        A = np.zeros((d, d), dtype=complex)
        for r in range(k, d):
            A[r - k, r] = np.sqrt(comb(r, k)) * np.sqrt((1.0 - gamma) ** (r - k)) \
                * gamma ** (k / 2.0)
        ops.append(A)
    return ops


# ═══════════════════════════════════════════════════════════════════════
# MS gate-angle fluctuation channel A_{sigma_a^2} (paper Sec. III)
# ═══════════════════════════════════════════════════════════════════════

def _ms_jitter_eigensystem(d: int, phi: float, level_j: int, level_k: int
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Eigensystem of the MS angle-fluctuation generator G = S_phi^2 / 4 on two qudits.

    sigma_phi = cos(phi) X_jk + sin(phi) Y_jk, i.e. the d x d matrix with
    [j,k] = e^{-i phi}, [k,j] = e^{+i phi}, zero elsewhere (eigenvalues
    +1, -1 on span{|j>,|k>}, 0 on the d-2 spectator levels). Because
    S_phi = sigma_phi (x) I + I (x) sigma_phi, the eigenbasis of G is the
    PRODUCT basis V1 (x) V1 of single-qudit sigma_phi eigenvectors, with
    eigenvalues g = (s1 + s2)^2 / 4 in {1, 1/4, 0}: the three tiers are
    both qudits in the driven subspace with equal sign (|++>, |-->: g = 1),
    exactly one qudit in the driven subspace (g = 1/4), and the rest
    (|+->, |-+> and any spectator level: g = 0).

    Args:
        d: qudit dimension
        phi: MS phase angle (benchmark convention phi = 0, sigma_phi = X_jk)
        level_j, level_k: the two levels the gate drives (convention (0, 1))

    Returns:
        V1: (d, d) unitary of sigma_phi eigenvectors (columns)
        gvals: (d*d,) G eigenvalues in the kron(V1, V1) column order
    """
    assert 0 <= level_j < d and 0 <= level_k < d and level_j != level_k
    sig = np.zeros((d, d), dtype=complex)
    sig[level_j, level_k] = np.exp(-1j * phi)
    sig[level_k, level_j] = np.exp(1j * phi)
    svals, V1 = np.linalg.eigh(sig)
    gvals = ((svals[:, None] + svals[None, :]) ** 2 / 4.0).ravel()
    return V1, gvals


def ms_jitter_kraus(d: int, sigma_a2: float, phi: float = 0.0,
                    level_j: int = 0, level_k: int = 1,
                    n_max: int = 10, drop_zero: bool = True) -> List[np.ndarray]:
    """Exact Gaussian-averaged MS gate-angle fluctuation channel A_{sigma_a^2} (two qudits).

    Paper Sec. III: an MS gate of angle theta executes as theta + dtheta with
    dtheta ~ N(0, sigma_a2), and because dtheta enters through the gate's own
    generator,
        U_MS(theta + dtheta) U_MS(theta)^dag = exp[-i (dtheta/4) S_phi^2]
    for every theta. The Gaussian average is therefore dephasing in the
    eigenbasis of G = S_phi^2/4 with coupling tiers g in {1, 1/4, 0}
    (_ms_jitter_eigensystem): the coherence between eigenspaces with gap
    Delta_g decays by e^{-1/2 sigma_a2 Delta_g^2}.

    Built from the SAME `_kraus_diagonal` construction as the dephasing
    channel (with eta_a = e^{-sigma_a2} and couplings g_m in the G eigenbasis),
    rotated back to the computational basis as V diag(E_n) V^dag with
    V = V1 (x) V1. With n_max >= 10 the truncation error of the completeness
    relation is <= 1.4e-10 at the calibrated sigma_a2 = 0.047.

    Args:
        d: qudit dimension
        sigma_a2: gate-angle fluctuation variance (calibrated sigma_a^2 = 0.047)
        phi: MS phase angle. The memory benchmark applies the channel as a
            per-pair noise dose with no physical gate instance, so phi is a
            stated convention: DEFAULT phi = 0 (sigma_phi = X_jk). The total
            error weight of the channel is exactly phi-independent; phi only
            rotates the error operator within the X/Y plane of the {j,k}
            subspace.
        level_j, level_k: the two levels the MS gate drives (convention (0,1))
        n_max: Kraus truncation order
        drop_zero: drop exactly-zero operators (at sigma_a2=0 this leaves
            the single identity operator)

    Returns:
        List of d^2 x d^2 complex matrices (Kraus operators on the pair,
        ordered as kron(qudit_a, qudit_b)).
    """
    assert sigma_a2 >= 0.0, f"sigma_a2 must be >= 0, got {sigma_a2}"
    if sigma_a2 == 0.0:
        # exact identity channel (V diag(1) V^dag would leave ~1e-32 dust)
        return [np.eye(d * d, dtype=complex)]
    V1, gvals = _ms_jitter_eigensystem(d, phi, level_j, level_k)
    eta_a = float(np.exp(-sigma_a2))
    diags = _kraus_diagonal(d * d, gvals, eta_a, n_max)
    V = np.kron(V1, V1)
    ops = []
    for diag in diags:
        if drop_zero and np.max(np.abs(diag)) == 0.0:
            continue
        ops.append((V * diag[None, :]) @ V.conj().T)
    return ops


def gate_channel_kraus_v2(d: int, control_level: int, target_level: int,
                          eta_dephasing: float, sigma_a2: float,
                          phi: float = 0.0,
                          jitter_level_j: int = 0, jitter_level_k: int = 1,
                          n_max_dephasing: int = 5, n_max_jitter: int = 10,
                          noise_model: str = "physical") -> dict:
    """Kraus operators of one calibrated gate step, paper Eq. (gate-channel).

    Per gate step on a coupled pair (control c, target t):
        N_gate = A_{sigma_a^2}(pair c,t)  o  [ D^(c) (x) D^(t) ]
    with every other qudit receiving the spectator dephasing D^(sp).
    The composition order (angle fluctuation after dephasing) follows the
    paper's equation; both factors are Gaussian averages of the SAME gate
    step and the order is a labelling of the per-step Kraus product
    (verified CPTP in tests/test_channel_v2.py).

    This is the full calibrated channel. It is NOT the sigma_phi (x) sigma_phi
    Pauli twirl of the angle fluctuation (that is only the projection used by
    the channel-weighting prior, src/meth_weights.py) and NOT
    subspace_depolarizing_kraus (legacy generic knob).

    At sigma_a2 = 0 the dephasing entries are BIT-IDENTICAL to
    control/target/spectator_qudit_kraus (same functions, same arguments)
    and 'pair_jitter' degenerates to the single identity operator.

    Args:
        d: qudit dimension
        control_level, target_level: levels i, j of the dephasing phase maps
        eta_dephasing: eta = exp(-sigma_p^2)
        sigma_a2: gate-angle fluctuation variance
        phi: MS phase angle (convention 0)
        jitter_level_j, jitter_level_k: levels the MS gate drives (convention (0, 1))
        n_max_dephasing, n_max_jitter: Kraus truncation orders
        noise_model: "physical" or "simplified" control coupling

    Returns:
        dict with keys:
          'control_dephasing', 'target_dephasing', 'spectator_dephasing':
              lists of diagonal arrays (the App. B dephasing families)
          'pair_jitter': list of d^2 x d^2 matrices on the (control, target)
              pair (identity-only list at sigma_a2 = 0)
    """
    if noise_model == "simplified":
        ctrl_kraus = control_qudit_kraus_simplified(d, control_level, eta_dephasing, n_max_dephasing)
    elif noise_model == "physical":
        ctrl_kraus = control_qudit_kraus(d, control_level, eta_dephasing, n_max_dephasing)
    else:
        raise ValueError(f"noise_model must be 'physical' or 'simplified', got '{noise_model}'")

    return {
        'control_dephasing': ctrl_kraus,
        'target_dephasing': target_qudit_kraus(d, control_level, target_level, eta_dephasing, n_max_dephasing),
        'spectator_dephasing': spectator_qudit_kraus(d, eta_dephasing, n_max_dephasing),
        'pair_jitter': ms_jitter_kraus(d, sigma_a2, phi=phi,
                                       level_j=jitter_level_j, level_k=jitter_level_k,
                                       n_max=n_max_jitter),
    }


def verify_kraus_completeness(diags: List[np.ndarray], atol: float = 1e-10) -> Tuple[bool, float]:
    """
    Check that sum_n E_n^dag E_n ~ I for diagonal Kraus operators.

    Since E_n is diagonal with diagonal d_n, E_n^dag E_n has diagonal |d_n|^2.
    Completeness means sum_n |d_n[k]|^2 = 1 for all k. For the truncated
    Gaussian series the deviation is the dropped tail of the Poisson-like
    weights (f_k sigma_p)^{2n} e^{-sigma_p^2 f_k^2} / n!, n > n_max.

    Args:
        diags: list of 1D arrays (Kraus operator diagonals)
        atol: absolute tolerance

    Returns:
        (is_complete, max_deviation): tuple of (bool, float)
    """
    sum_sq = np.zeros_like(diags[0], dtype=float)
    for d in diags:
        sum_sq += np.abs(d) ** 2

    deviation = np.max(np.abs(sum_sq - 1.0))
    return deviation < atol, float(deviation)


def verify_matrix_kraus_completeness(kraus_ops: List[np.ndarray], atol: float = 1e-10) -> Tuple[bool, float]:
    """
    Check that sum_k K_k^dag K_k = I for full matrix Kraus operators.

    Args:
        kraus_ops: list of d x d complex matrices
        atol: absolute tolerance

    Returns:
        (is_complete, max_deviation): tuple of (bool, float)
    """
    d = kraus_ops[0].shape[0]
    total = np.zeros((d, d), dtype=complex)
    for K in kraus_ops:
        total += K.conj().T @ K
    deviation = np.max(np.abs(total - np.eye(d)))
    return deviation < atol, float(deviation)


# ═══════════════════════════════════════════════════════════════════════
# Correlated error-set construction (Kraus products for the detection loss)
# ═══════════════════════════════════════════════════════════════════════

def _single_qudit_error_products(kraus_diags: List[np.ndarray]) -> List[np.ndarray]:
    """
    Compute all error products E_alpha^dag E_beta for a single qudit's Kraus operators.

    These are the operators the Knill-Laflamme conditions (paper Sec. IV B)
    are stated against. Since operators are diagonal:
        (E_alpha^dag E_beta)[k,k] = conj(E_alpha[k,k]) * E_beta[k,k]
                                  = conj(diag_alpha[k]) * diag_beta[k]

    Returns unique products (deduplicated by value, rounded to 12 decimals).
    """
    products = []
    seen = set()
    for da in kraus_diags:
        for db in kraus_diags:
            prod = np.conj(da) * db
            key = tuple(np.round(prod, decimals=12))
            if key not in seen:
                seen.add(key)
                products.append(prod)
    return products


def build_correlated_error_set(
    n_qudits: int,
    d: int,
    gate_pairs: List[Tuple[int, int, int, int]],
    eta: float,
    n_max: int = 2,
    truncation_threshold: float = 1e-8,
    noise_model: str = "physical"
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Build a detection set directly from the dephasing Kraus products of the channel.

    This is the "Kraus-product" alternative to the discrete channel error
    basis of paper Sec. IV C: instead of the Z_k / X_{k-1,k} operators it
    enumerates the register-wide products E_alpha^dag E_beta of the
    Gaussian-averaged dephasing Kraus operators of every gate step. It is
    used by the tests and the legacy Kraus benchmark script; the reported
    codes are optimised against the channel error basis (src/errors.py).

    For each gate acting on (qudit_a, qudit_b) at levels (ctrl_level, tgt_level):
    - qudit_a gets control_qudit_kraus (physical) or control_qudit_kraus_simplified
    - qudit_b gets target_qudit_kraus (same in both models)
    - all other qudits get spectator_qudit_kraus (same in both models)

    All operators are diagonal -> stored as 1D vectors of length d^n_qudits.

    Args:
        n_qudits: number of physical qudits
        d: qudit dimension (3, 5, or 7)
        gate_pairs: list of (qudit_a, qudit_b, ctrl_level, tgt_level)
        eta: noise parameter eta = exp(-sigma_p^2)
        n_max: Kraus truncation order (2-3 is usually sufficient)
        truncation_threshold: drop operators with max|diag| below this
        noise_model: "physical" (f_k=k) or "simplified" (uniform f=2)

    Returns:
        E_detect: list of 1D arrays of length d^n_qudits (error operator diagonals),
                  identity first
        E_correct: same list (for this noise model, detect = correct)
    """
    if noise_model == "simplified":
        _control_kraus_fn = control_qudit_kraus_simplified
    elif noise_model == "physical":
        _control_kraus_fn = control_qudit_kraus
    else:
        raise ValueError(f"noise_model must be 'physical' or 'simplified', got '{noise_model}'")

    dim_full = d ** n_qudits
    all_error_diags = set()

    # Identity is always in the error set
    identity_diag = np.ones(dim_full, dtype=complex)
    all_error_diags.add(tuple(np.round(identity_diag, decimals=12)))

    for qudit_a, qudit_b, ctrl_level, tgt_level in gate_pairs:
        # Get single-qudit Kraus diagonals for each role
        per_qudit_products = []
        for q in range(n_qudits):
            if q == qudit_a:
                kraus = _control_kraus_fn(d, ctrl_level, eta, n_max)
            elif q == qudit_b:
                kraus = target_qudit_kraus(d, ctrl_level, tgt_level, eta, n_max)
            else:
                kraus = spectator_qudit_kraus(d, eta, n_max)
            per_qudit_products.append(_single_qudit_error_products(kraus))

        # Tensor product of single-qudit error products
        for combo in cartesian_product(*per_qudit_products):
            # Tensor product of diagonals = kronecker product of 1D vectors
            full_diag = combo[0]
            for q_diag in combo[1:]:
                full_diag = np.kron(full_diag, q_diag)

            # Truncation: skip near-identity or near-zero operators
            if np.max(np.abs(full_diag - identity_diag)) < truncation_threshold:
                continue  # too close to identity, skip
            if np.max(np.abs(full_diag)) < truncation_threshold:
                continue  # negligible operator

            key = tuple(np.round(full_diag, decimals=12))
            all_error_diags.add(key)

    # Convert back to arrays, add identity
    error_list = [identity_diag]
    for key in all_error_diags:
        diag = np.array(key, dtype=complex)
        if not np.allclose(diag, identity_diag):
            error_list.append(diag)

    print(f"Correlated error set ({noise_model}): {len(error_list)} operators for {n_qudits} qudits (d={d})")

    # For channel-adaptive codes, E_detect = E_correct
    return error_list, error_list


def _compose_kraus_single_qudit(dephasing_diags: List[np.ndarray],
                                 depol_matrices: List[np.ndarray]) -> List[np.ndarray]:
    """
    Compose dephasing and subspace depolarizing Kraus ops for one qudit.

    The composed channel is N_depol o N_dephasing, so:
        K_{n,k} = S_k @ diag(D_n)
    for dephasing index n and depolarizing index k. (Legacy helper of the
    subspace-depolarizing model; kept for the derivation scripts.)

    Returns list of d x d matrices.
    """
    composed = []
    for D_n in dephasing_diags:
        D_matrix = np.diag(D_n)
        for S_k in depol_matrices:
            composed.append(S_k @ D_matrix)
    return composed
