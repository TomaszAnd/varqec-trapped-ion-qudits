"""
Channel error basis, completed detection set and i.i.d. noise injection
for trapped-ion qudits of any dimension d.

Implements paper Sec. IV C ("Channel error basis and completion"):

  * the single-qudit CHANNEL ERROR BASIS of 2(d-1) unitary operators
    (qudit_hardware_error_basis): d-1 phase flips Z_k (level k = 1..d-1,
    representing the dephasing part of the calibrated channel) and d-1
    adjacent-level swaps X_{k-1,k} (representing the entangling-gate angle
    fluctuation). At d = 3 the basis is [Z_1, Z_2, X_01, X_12];
  * two optional enlargements used by the unweighted protocol variants:
    "extended" adds the quadrature swaps Y_{k-1,k}; "full" further adds the
    lowering/raising operators L, L^dag of each transition (amplitude
    damping, non-unitary);
  * the COMPLETED DETECTION SET (build_native_error_set,
    build_native_error_set_factored, ErrorModel): all products of basis
    operators of weight <= distance-1 across the register, plus, for
    distance >= 3, the same-qudit products E_a E_b that complete the set
    (close_error_basis), with entries proportional to the identity or
    already present dropped; and the correction set of weight
    <= floor((distance-1)/2);
  * the wire-grouped representation of the detection set consumed by the
    vectorized loss kernels (factored_to_grouped / ErrorModel.build_grouped;
    paper App. "Wire-grouped implementation");
  * the discrete-basis i.i.d. noise injection of the LER benchmark
    (make_hardware_noise_fn, make_weighted_hardware_noise_fn; paper
    Sec. "Logical error rate protocol"): one basis error per qudit with
    probability p.

Index conventions: single-qudit operators are (d, d) matrices in the
computational basis; an n-qudit dense operator is the Kronecker product
with qudit 0 as the most significant factor (flat index
sum_q k_q d**(n-1-q)), the same convention as src/encoder.py.

Consumers: src/loss.py, src/jax_backend.py, src/meth_weights.py (channel
weights over this basis), src/seed_race.py, src/central_comparison.py,
src/decoders/*, scripts/train.py, scripts/benchmark_ler*.py and the
campaign scripts under results/campaigns,
src/__init__.py (re-exports) and the tests listed in each docstring.

Paper cross-check: basis operators and their order, the completion rule
and the distance thresholds match Sec. IV C; see the "Paper check" notes
in the individual docstrings. One wording issue in the manuscript is
recorded at close_error_basis.
"""
import itertools
from pennylane import numpy as np


def qudit_hardware_error_basis(d: int):
    """
    The channel error basis of paper Sec. IV C: 2(d-1) unitary single-qudit
    operators, in the fixed order used by every weight vector in the code.

      - Z_k = I - 2|k><k|, k = 1..d-1: phase flip on level k (sign of level
        k relative to all others). These span the diagonal operators and
        represent the dephasing part of the calibrated channel.
      - X_{k-1,k} = |k-1><k| + |k><k-1| + P_rest, k = 1..d-1: swap of the
        adjacent levels k-1 and k, identity elsewhere. Represents the
        angle fluctuation of a gate driving the (k-1,k) transition.

    Order: [Z_1, ..., Z_{d-1}, X_01, ..., X_{d-2,d-1}].
    For d=3: [Z_1, Z_2, X_01, X_12] (4 errors);
    for d=4: [Z_1, Z_2, Z_3, X_01, X_12, X_23] (6 errors).
    All operators are Hermitian and unitary, so E^dag = E.

    Paper check: identical operators, order and d=3 example as Sec. IV C.

    Args:
        d: qudit dimension (>= 2).

    Returns:
        list of 2(d-1) complex (d, d) arrays.
    """
    errors = []

    # Z-type: phase flip on level k (dephasing / AC-Stark phase)
    for k in range(1, d):
        Z = np.eye(d, dtype=complex)
        Z[k, k] = -1.0
        errors.append(Z)

    # X-type: swap adjacent levels (bitflip in the driven two-level block)
    for k in range(d - 1):
        X = np.eye(d, dtype=complex)
        X[k, k] = 0.0
        X[k + 1, k + 1] = 0.0
        X[k, k + 1] = 1.0
        X[k + 1, k] = 1.0
        errors.append(X)

    return errors


def qudit_extended_hardware_error_basis(d: int):
    """
    "extended" basis: the channel error basis plus the quadrature swaps.

    Appends Y_{k-1,k} = -i|k-1><k| + i|k><k-1| + P_rest (k = 1..d-1) to
    qudit_hardware_error_basis(d), giving 3(d-1) operators in the order
    [Z_1..Z_{d-1}, X_01..X_{d-2,d-1}, Y_01..Y_{d-2,d-1}]. With Z, X, Y on
    each adjacent block this is the subspace Pauli triple. Used only by the
    unweighted protocol variants; the paper's channel error basis is the
    2(d-1)-operator "hardware" basis (the Y swaps are not in Sec. IV C).

    Args:
        d: qudit dimension (>= 2).

    Returns:
        list of 3(d-1) complex (d, d) unitary (Hermitian) arrays.
    """
    errors = qudit_hardware_error_basis(d)

    # Y-type: quadrature component of laser fluctuation
    for k in range(d - 1):
        Y = np.eye(d, dtype=complex)
        Y[k, k] = 0.0
        Y[k + 1, k + 1] = 0.0
        Y[k, k + 1] = -1j
        Y[k + 1, k] = 1j
        errors.append(Y)

    return errors


def qudit_full_hardware_error_basis(d: int):
    """
    "full" basis: extended basis plus amplitude-damping ladder operators.

    Returns 5(d-1) operators in the order
    [Z_1..Z_{d-1}, X_01.., Y_01.., L_01..L_{d-2,d-1}, Ld_01..Ld_{d-2,d-1}]:
      - Z_k, X_{k-1,k}, Y_{k-1,k}: the extended basis (subspace Pauli
        triple on each adjacent transition);
      - L_{k-1,k} = |k-1><k|: lowering operator per transition. The
        amplitude-damping first-order Kraus operator
        A_1 = sum_r sqrt(r gamma (1-gamma)^{r-1}) |r-1><r| lies in
        span{L_r} for every gamma, so detecting each L_r detects A_1
        exactly, gamma-independently;
      - L^dag_{k-1,k} = |k><k-1|: raising operator (heating), included so
        that the basis is closed under the adjoint: close_error_basis()
        forms plain products E_a E_b, while the Knill-Laflamme conditions
        need E_a^dag E_b, and E_a^dag is itself a basis element only if
        both L and L^dag are present.

    The L and L^dag operators are NOT unitary; see the caveat on
    factored_to_grouped about deduplication with this basis. Used only by
    the unweighted "full"-noise protocol variants (scripts/train.py
    --noise full, src/seed_race.py); the paper's channel error basis is
    the "hardware" basis.

    Args:
        d: qudit dimension (>= 2).

    Returns:
        list of 5(d-1) complex (d, d) arrays.
    """
    errors = qudit_extended_hardware_error_basis(d)

    # L-type: amplitude damping (lowering), one per adjacent transition
    for k in range(1, d):
        L = np.zeros((d, d), dtype=complex)
        L[k - 1, k] = 1.0
        errors.append(L)

    # L^dag-type: raising (heating; dagger closure of the basis)
    for k in range(1, d):
        Ld = np.zeros((d, d), dtype=complex)
        Ld[k, k - 1] = 1.0
        errors.append(Ld)

    return errors


_BASIS_BUILDERS = {
    "hardware": qudit_hardware_error_basis,
    "extended": qudit_extended_hardware_error_basis,
    "full": qudit_full_hardware_error_basis,
}


def _single_qudit_error_basis(d: int, basis=None, extended: bool = False):
    """Resolve the single-qudit basis by name.

    basis=None keeps the older boolean interface exactly: extended=False
    -> "hardware", extended=True -> "extended". A given basis name
    overrides `extended`.
    """
    if basis is None:
        basis = "extended" if extended else "hardware"
    if basis not in _BASIS_BUILDERS:
        raise ValueError(f"basis must be one of {sorted(_BASIS_BUILDERS)}, got '{basis}'")
    return _BASIS_BUILDERS[basis](d)


def _is_proportional_to_identity(M, d):
    """Check if M = c * I for some scalar c with |c| = 1."""
    if not np.allclose(M - np.diag(np.diag(M)), 0, atol=1e-12):
        return False
    diag = np.diag(M)
    return np.allclose(diag, diag[0] * np.ones(d), atol=1e-12)


def _is_proportional(M, E):
    """Check M = c * E for some nonzero scalar c. Valid for arbitrary
    (including non-unitary, non-invertible) operators."""
    idx = np.unravel_index(np.argmax(np.abs(E)), E.shape)
    if np.abs(E[idx]) < 1e-12:
        return False
    c = M[idx] / E[idx]
    if np.abs(c) < 1e-12:
        return False
    return np.allclose(M, c * E, atol=1e-12)


def _is_in_set(M, op_set, d):
    """Check if M is proportional (up to a nonzero scalar) to any operator
    in op_set. `d` is unused (kept for signature stability).

    Uses the direct proportionality test, which is equivalent to the
    earlier M @ E^dag proportional-to-identity test for unitary E (E
    invertible: M @ E^dag = cI iff M = cE) and remains correct for the
    non-unitary lowering/raising operators of the full basis, where
    M @ E^dag is never proportional to identity and zero products would
    false-positive the identity test. (factored_to_grouped still uses the
    identity test for its dedup; see the caveat there.)
    """
    for E in op_set:
        if _is_proportional(M, E):
            return True
    return False


def close_error_basis(single_errors):
    """
    Same-qudit completion of a single-qudit error basis (paper Sec. IV C).

    Correcting every weight-1 error requires the Knill-Laflamme condition
    <psi_i| E_a^dag E_b |psi_j> = delta_ij alpha_ab for every PAIR of
    weight-1 errors, and the products E_a^dag E_b of two errors on the same
    qudit generally leave the basis (the channel error basis is not a
    group). This function forms all pairwise products M = E_a E_b of the
    given operators and returns the NEW operators: products proportional
    to the identity, proportional to an input operator, or proportional
    to an already accepted product are dropped (proportionality up to any
    nonzero scalar, i.e. up to a global phase).

    The products are plain E_a E_b, not E_a^dag E_b: this is the same set
    whenever the basis is closed under the adjoint, which holds for the
    "hardware" and "extended" bases (every operator Hermitian) and for the
    "full" basis (L and L^dag both present).

    Paper check: matches the completion rule of Sec. IV C ("same-qudit
    products ... entries proportional to identity, or already present,
    are dropped"). The manuscript's illustrative example there, written
    as Z_1 X_01 = -|1><0| at q = 3, does not match the operators as
    defined: with Z_1 = I - 2|1><1| and X_01 the full swap, the product
    is |0><1| - |1><0| + |2><2|. The conclusion drawn in the paper (the
    product is not a basis operator) holds either way; the code is
    unchanged. For the d=3 hardware basis the completion has 8 operators.

    Args:
        single_errors: list of (d, d) arrays (any of the bases above).

    Returns:
        list of (d, d) arrays: the deduplicated new operators only
        (originals and identity-proportional products excluded).
    """
    d = single_errors[0].shape[0]
    cross_products = []

    for Ea in single_errors:
        for Eb in single_errors:
            M = Ea @ Eb
            if _is_proportional_to_identity(M, d):
                continue
            if _is_in_set(M, single_errors, d):
                continue
            if _is_in_set(M, cross_products, d):
                continue
            cross_products.append(M)

    return cross_products


def _embed_single_qudit(E, q, n_qudit, d):
    """Embed the (d, d) operator E on qudit q as a dense d**n x d**n operator
    (Kronecker product with identities, qudit 0 most significant)."""
    Id = np.eye(d, dtype=complex)
    op = 1
    for i in range(n_qudit):
        op = np.kron(op, E if i == q else Id)
    return op


def build_native_error_set(n_qudit: int, d: int, distance: int,
                           extended: bool = False, closed: bool = False,
                           basis=None):
    """
    Dense completed detection set and correction set (paper Sec. IV C).

    E_det contains, as dense d**n x d**n matrices:
      - the identity;
      - every tensor product of basis operators of weight w = 1 ..
        distance-1 (all w-subsets of qudits, all basis choices per qudit,
        ordered by weight, then qudit subset, then basis choice);
      - if closed=True and distance >= 3: every same-qudit completion
        operator of close_error_basis(basis) embedded on each qudit
        (weight 1). These are the products E_a E_b the Knill-Laflamme
        correction conditions need; at distance 2 nothing is added, which
        is the paper's "at d = 2 only the weight-1 errors are needed".
    E_corr contains the identity and the weight <= floor((distance-1)/2)
    products (the errors the code is meant to CORRECT); it never contains
    the completion operators.

    Paper check: weight thresholds and completion rule agree with
    Sec. IV C. Memory grows as d**(2n) per operator; use
    build_native_error_set_factored / ErrorModel.build_grouped for n >= 7.

    Args:
        n_qudit: number of physical qudits n.
        d: qudit dimension.
        distance: target code distance (2, 3, 4).
        extended: if True and basis is None, use the "extended" basis.
        closed: if True, add the same-qudit completion to E_det.
        basis: "hardware" | "extended" | "full"; None (default) derives
            the basis from `extended`.

    Returns:
        (E_det, E_corr): two lists of complex (d**n, d**n) arrays.
    """
    Id = np.eye(d, dtype=complex)

    id_full = Id
    for _ in range(1, n_qudit):
        id_full = np.kron(id_full, Id)

    E_det = [id_full]
    E_corr = [id_full]

    max_det = distance - 1
    max_corr = (distance - 1) // 2

    single_errs = _single_qudit_error_basis(d, basis=basis, extended=extended)

    # Standard weight-1 through weight-max_det construction
    for w in range(1, max_det + 1):
        for qudit_subset in itertools.combinations(range(n_qudit), w):
            for err_choice in itertools.product(single_errs, repeat=w):
                op = 1
                idx_choice = 0
                for q in range(n_qudit):
                    if q in qudit_subset:
                        op = np.kron(op, err_choice[idx_choice])
                        idx_choice += 1
                    else:
                        op = np.kron(op, Id)
                E_det.append(op)
                if w <= max_corr:
                    E_corr.append(op)

    # Completion: add the same-qudit products (weight 1) to E_det only.
    # max_corr >= 1 <=> distance >= 3, i.e. only when weight-1 errors are
    # to be corrected and the pair products E_a E_b must be detected.
    if closed and max_corr >= 1:
        cross_products = close_error_basis(single_errs)
        for q in range(n_qudit):
            for E in cross_products:
                op = _embed_single_qudit(E, q, n_qudit, d)
                E_det.append(op)

    return E_det, E_corr


def build_native_error_set_factored(n_qudit: int, d: int, distance: int,
                                     extended: bool = False, closed: bool = False,
                                     basis=None):
    """
    Same sets as build_native_error_set, in factored (sparse) form.

    Each error is a list of (qudit_idx, (d, d) matrix) tuples, in the
    same order as the dense builder:
      weight 0 (identity): []
      weight 1:            [(q, E)]
      weight 2:            [(q1, E1), (q2, E2)]   (q1 < q2)
    The completion operators (closed=True, distance >= 3) appear as
    weight-1 factors [(q, E)] appended after the product errors. Memory is
    O(w d^2) per error instead of O(d**(2n)); this is the representation
    the wire-grouped kernels start from (paper App. "Wire-grouped
    implementation").

    Args: as build_native_error_set.

    Returns:
        (E_det_factors, E_corr_factors): two lists of factor lists.
    """
    single_errs = _single_qudit_error_basis(d, basis=basis, extended=extended)

    E_det_factors = [[]]  # identity = empty factor list
    E_corr_factors = [[]]

    max_det = distance - 1
    max_corr = (distance - 1) // 2

    for w in range(1, max_det + 1):
        for qudit_subset in itertools.combinations(range(n_qudit), w):
            for err_choice in itertools.product(single_errs, repeat=w):
                factors = [(qudit_subset[i], err_choice[i]) for i in range(w)]
                E_det_factors.append(factors)
                if w <= max_corr:
                    E_corr_factors.append(factors)

    if closed and max_corr >= 1:
        cross_products = close_error_basis(single_errs)
        for q in range(n_qudit):
            for E in cross_products:
                E_det_factors.append([(q, E)])

    return E_det_factors, E_corr_factors


# ── Discrete-basis i.i.d. noise injection (LER benchmark) ─────────────

def make_hardware_noise_fn(d: int, n_qudit: int, p: float):
    """
    i.i.d. discrete-basis noise of the LER benchmark (paper Sec. "Logical
    error rate protocol"): for each qudit independently, with probability
    1-p do nothing, with probability p apply ONE operator drawn uniformly
    from the 2(d-1)-element channel error basis qudit_hardware_error_basis.

    The basis operators are unitary, so the state stays normalized; the
    final renormalization is only a numerical guard. Uses plain numpy
    (no autograd) since this runs inside Monte-Carlo loops.

    Args:
        d: qudit dimension.
        n_qudit: number of qudits.
        p: per-qudit error probability (the benchmark's "p").

    Returns:
        noise_fn(state, rng) -> noisy_state, where state is a flat complex
        vector of length d**n_qudit (qudit 0 most significant) and rng a
        numpy.random.Generator. The input state is not modified.
    """
    import numpy
    single_errors = [numpy.array(E) for E in qudit_hardware_error_basis(d)]
    n_errors = len(single_errors)
    shape = [d] * n_qudit

    def noise_fn(state, rng):
        s = state.copy()
        for q in range(n_qudit):
            if rng.random() < p:
                idx = rng.integers(n_errors)
                E = single_errors[idx]
                s_tensor = s.reshape(shape)
                s_tensor = numpy.tensordot(E, s_tensor, axes=([1], [q]))
                s_tensor = numpy.moveaxis(s_tensor, 0, q)
                s = s_tensor.reshape(-1)
        norm = numpy.linalg.norm(s)
        if norm > 1e-15:
            s = s / norm
        return s

    return noise_fn


def make_weighted_hardware_noise_fn(d: int, n_qudit: int, p: float, weights):
    """
    Channel-weighted variant of make_hardware_noise_fn.

    Identical protocol: for each qudit independently, with probability
    1-p do nothing, with probability p apply one basis error; but the
    error is drawn with probability weights[i] instead of uniformly. The
    per-qudit injection probability p is unchanged, so the injected error
    strength matches the uniform benchmark at every quoted p (every
    hardware basis operator has Tr = d-2, hence the same single-operator
    average infidelity; see src/meth_weights.hardware_benchmark_weights
    for the channel-derived weights).

    Args:
        d, n_qudit, p: as make_hardware_noise_fn.
        weights: length-2(d-1) probability vector indexed in the order of
            qudit_hardware_error_basis(d).

    Returns:
        noise_fn(state, rng) -> noisy_state, as make_hardware_noise_fn.
    """
    import numpy
    single_errors = [numpy.array(E) for E in qudit_hardware_error_basis(d)]
    w = numpy.asarray(weights, dtype=float)
    if w.shape != (len(single_errors),):
        raise ValueError(f"weights must have shape ({len(single_errors)},), "
                         f"got {w.shape}")
    if numpy.any(w < 0) or not numpy.isclose(w.sum(), 1.0):
        raise ValueError("weights must be a probability vector")
    cumw = numpy.cumsum(w)
    cumw[-1] = 1.0
    shape = [d] * n_qudit

    def noise_fn(state, rng):
        s = state.copy()
        for q in range(n_qudit):
            if rng.random() < p:
                idx = int(numpy.searchsorted(cumw, rng.random(),
                                             side='right'))
                E = single_errors[idx]
                s_tensor = s.reshape(shape)
                s_tensor = numpy.tensordot(E, s_tensor, axes=([1], [q]))
                s_tensor = numpy.moveaxis(s_tensor, 0, q)
                s = s_tensor.reshape(-1)
        norm = numpy.linalg.norm(s)
        if norm > 1e-15:
            s = s / norm
        return s

    return noise_fn


# ── Backward-compatibility aliases (d = 3) ────────────────────────────
# Kept because src/__init__.py re-exports them and tests/test_qutrit.py,
# tests/test_native.py use them. Prefer the qudit_* functions.

def hardware_error_basis():
    """Single-qutrit (d=3) channel error basis. Backward-compat alias."""
    return qudit_hardware_error_basis(d=3)


def extended_hardware_error_basis():
    """Extended qutrit (d=3) basis. Backward-compat alias."""
    return qudit_extended_hardware_error_basis(d=3)


def build_qutrit_error_set(n_qutrit: int, distance: int, extended: bool = False):
    """Dense error sets for qutrits (d=3, closed=False). Backward-compat alias."""
    return build_native_error_set(n_qutrit, d=3, distance=distance, extended=extended)


# ── Wire-grouped representation (vectorized loss kernels) ─────────────

def factored_to_grouped(factored_list, n_qudit: int, d: int,
                        dedup: bool = True, verbose: bool = True):
    """
    Convert factored errors [(q, op), ...] into wire-grouped form (paper
    App. "Wire-grouped implementation").

    Errors acting on the same sorted set of qudits ("wire signature") are
    stacked along a leading axis so that one vectorized contraction
    (jax.vmap over src.jax_backend._apply_local_matrix, or the numpy
    equivalent src.loss._apply_local_dense) applies all of them. For each
    error the local matrix is the Kronecker product of its factors in
    increasing qudit order, i.e. a (d**w, d**w) matrix whose row/column
    index is k_{w0} d**(w-1) + ... + k_{w_{w-1}} for wires (w0 < w1 < ...).

    Ported from U. Holzer's qutrit prototype and generalized to any d and
    n_qudit with explicit shape parameters.

    Deduplication (dedup=True): an error is dropped when, for an already
    accepted error M2 on the same wires, M @ M2^dag is proportional to the
    identity, i.e. M = c M2 for unitary M2. CAVEAT for the non-unitary
    "full" basis: the test also returns True when M @ M2^dag = 0 (the zero
    matrix passes the proportional-to-identity check), so pairs of ladder
    operators with vanishing product (e.g. L_12 and L_01 on one qudit) are
    treated as duplicates and the later one is silently dropped. For
    d=3, n=3, distance 3, closed, the "full" detection set shrinks from
    415 to 289 operators while an exact proportionality test keeps all
    415. The "hardware" and "extended" bases are unaffected (unitary).
    This behaviour is kept as is; callers using basis="full" with the
    grouped path should be aware of it.

    Args:
        factored_list: list of factor lists [(q, (d, d) op), ...]; the
            empty list denotes the identity.
        n_qudit: total number of qudits.
        d: qudit dimension.
        dedup: if True, drop errors equal (up to a global phase) to a
            previously accepted error on the same wires (see caveat).
        verbose: if True, print a one-line dedup summary.

    Returns:
        list of dicts, one per distinct wire signature, with keys
          'wires':        tuple of sorted qudit indices (w of them);
          'matrices':     complex128 array (n_in_group, d**w, d**w);
          'inverse_perm': tuple of length n_qudit, see the comment at its
                          construction below.
        The identity, if present, forms the group wires=() with matrices
        of shape (1, 1, 1) holding [[1.0]]; the loss kernels skip it.
    """
    tagged = []
    for factors in factored_list:
        if not factors:
            wires = ()
            small = np.array([[1.0 + 0.0j]], dtype=complex)
        else:
            sorted_factors = sorted(factors, key=lambda qe: qe[0])
            wires = tuple(q for q, _ in sorted_factors)
            ops = [np.asarray(op, dtype=complex) for _, op in sorted_factors]
            small = ops[0]
            for op in ops[1:]:
                small = np.kron(small, op)
        tagged.append((wires, small))

    n_before = len(tagged)
    if dedup:
        accepted = []
        for wires, M in tagged:
            dup = False
            for wires2, M2 in accepted:
                if wires2 != wires:
                    continue
                prod = M @ M2.conj().T
                if _is_proportional_to_identity(prod, prod.shape[0]):
                    dup = True
                    break
            if not dup:
                accepted.append((wires, M))
        tagged = accepted
    n_after = len(tagged)

    if verbose:
        print(f"dedup: kept {n_after}/{n_before} errors "
              f"(removed {n_before - n_after})")

    grouped_by_wires = {}
    for wires, M in tagged:
        grouped_by_wires.setdefault(wires, []).append(M)

    result = []
    for wires, mats in grouped_by_wires.items():
        stacked = np.stack(mats).astype(np.complex128)
        # Applying a group matrix by tensordot over the `wires` axes of the
        # rank-n state tensor leaves the result with axes ordered
        # new_order = [wires..., every other qudit in increasing order].
        # inverse_perm[i] = position of original qudit i in new_order, so
        # transpose(result, inverse_perm) restores the original qudit order
        # (it is the inverse of new_order as a permutation).
        leftover = [i for i in range(n_qudit) if i not in wires]
        new_order = list(wires) + leftover
        inverse_perm = tuple(new_order.index(i) for i in range(n_qudit))
        result.append({
            'wires': wires,
            'matrices': stacked,
            'inverse_perm': inverse_perm,
        })
    return result


def grouped_to_dense(grouped, n_qudit: int, d: int):
    """
    Reconstruct dense d**n x d**n operators from the wire-grouped form.

    Test utility (tests/test_errors.py) for round-trip validation against
    build_native_error_set: each group matrix is applied to every
    computational basis vector with the same contraction + inverse_perm
    recipe the loss kernels use, and the resulting columns are assembled
    into the dense operator.

    Args:
        grouped: output of factored_to_grouped / ErrorModel.build_grouped.
        n_qudit, d: register size and local dimension.

    Returns:
        list of complex (d**n, d**n) arrays, one per error across all
        groups in group order (the identity group, if present, yields
        identity matrices).
    """
    dim = d ** n_qudit
    Id_qudit = np.eye(d, dtype=complex)
    Id_full = np.eye(dim, dtype=complex)

    flat = []
    for group in grouped:
        wires = group['wires']
        mats = group['matrices']
        if len(wires) == 0:
            for _ in range(mats.shape[0]):
                flat.append(Id_full.copy())
            continue

        w = len(wires)
        for e_idx in range(mats.shape[0]):
            M_small = mats[e_idx]
            factors_by_q = {}
            # M_small has shape (d**w, d**w) and lives on the `wires` axes.
            # Rather than factorizing it, embed it directly: apply it to
            # each computational basis vector with the same tensordot +
            # inverse_perm recipe as the loss kernels and read off the
            # resulting column of the dense operator.
            shape = (d,) * n_qudit
            full = np.zeros((dim, dim), dtype=complex)
            M_tensor = M_small.reshape((d,) * (2 * w))
            inverse_perm = group['inverse_perm']
            for col in range(dim):
                v = np.zeros(dim, dtype=complex)
                v[col] = 1.0
                v_t = v.reshape(shape)
                contracted = np.tensordot(
                    M_tensor, v_t,
                    axes=(list(range(w, 2 * w)), list(wires)))
                v_t = np.transpose(contracted, inverse_perm)
                full[:, col] = v_t.reshape(dim)
            flat.append(full)
    return flat


# ── ErrorModel class ─────────────────────────────────────────────────

class ErrorModel:
    """
    Discrete error model of a register: basis + completed detection set.

    Bundles qudit dimension d, register size n_qudit, target distance and
    the single-qudit basis choice into one object and exposes the three
    representations of the detection/correction sets (dense, factored,
    wire-grouped) and the i.i.d. noise injector. This is the object the
    training scripts and the LER benchmark construct first
    (e.g. ErrorModel(d=3, n_qudit=5, distance=3, closed=True)).

    Attributes:
        d, n_qudit, distance, extended, closed, basis: the constructor
            arguments (see build_native_error_set for their meaning).
        single_errors: the resolved single-qudit basis (list of (d, d)).
        cross_products: close_error_basis(single_errors) if closed, else [].
        dim: d ** n_qudit.  K: returns d (the K = d convention of
            scripts/train.py; no other consumer).  n_single_errors: basis
            size.

    Usage:
        model = ErrorModel(d=3, n_qudit=5, distance=3, closed=True)
        E_det, E_corr = model.build_dense()
        E_det_f, E_corr_f = model.build_factored()
        E_det_g = model.build_grouped()
        noise_fn = model.make_noise_fn(p=0.05)
    """

    def __init__(self, d: int, n_qudit: int, distance: int = 2,
                 extended: bool = False, closed: bool = False,
                 basis=None):
        self.d = d
        self.n_qudit = n_qudit
        self.distance = distance
        self.extended = extended
        self.closed = closed
        self.basis = basis

        self.single_errors = _single_qudit_error_basis(
            d, basis=basis, extended=extended)

        self.cross_products = close_error_basis(self.single_errors) if closed else []

    @property
    def dim(self) -> int:
        return self.d ** self.n_qudit

    @property
    def K(self) -> int:
        return self.d

    @property
    def n_single_errors(self) -> int:
        return len(self.single_errors)

    def build_dense(self):
        """Build dense (dim x dim matrix) error sets. Returns (E_det, E_corr)."""
        return build_native_error_set(
            self.n_qudit, self.d, self.distance,
            extended=self.extended, closed=self.closed, basis=self.basis)

    def build_factored(self):
        """Build factored error sets. Returns (E_det_factors, E_corr_factors)."""
        return build_native_error_set_factored(
            self.n_qudit, self.d, self.distance,
            extended=self.extended, closed=self.closed, basis=self.basis)

    def build_grouped(self, dedup: bool = True, verbose: bool = True,
                      factored=None):
        """
        Wire-grouped DETECTION set for the vectorized loss kernels
        (src.jax_backend.create_jax_loss_vmap*, src.loss grouped path).

        Builds the factored detection set (or takes `factored` if given)
        and converts it with factored_to_grouped; see that function for
        the shape contract and the dedup caveat with basis="full".

        Args:
            dedup, verbose: passed to factored_to_grouped.
            factored: optional precomputed factored detection list; None
                builds it from this model.

        Returns:
            list of dicts, one per distinct wire signature, with keys
              wires:        tuple of sorted qudit indices (w of them)
              matrices:     complex128 array (n_in_group, d**w, d**w)
              inverse_perm: tuple of length n_qudit restoring the original
                            qudit order after the `wires` axes have been
                            contracted to the front (src.jax_backend.
                            _apply_local_matrix convention).
            The identity is kept as the group wires=() with matrices of
            shape (1, 1, 1) holding [[1.0]]; the loss skips it.
        """
        if factored is None:
            factored, _ = self.build_factored()
        return factored_to_grouped(factored, self.n_qudit, self.d,
                                   dedup=dedup, verbose=verbose)

    def make_noise_fn(self, p: float):
        """Create independent per-qudit hardware noise function."""
        return make_hardware_noise_fn(self.d, self.n_qudit, p)

    def __repr__(self):
        cross = f"+{len(self.cross_products)} cross" if self.cross_products else ""
        return (f"ErrorModel(d={self.d}, n={self.n_qudit}, dist={self.distance}, "
                f"errors={self.n_single_errors}{cross})")
