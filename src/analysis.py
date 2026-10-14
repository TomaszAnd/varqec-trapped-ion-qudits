"""
Code characterization: Knill-Laflamme residuals, distance probe,
Shor-Laflamme weight enumerators and entanglement entropy.

Works on any code given as a (K, dim) array of codeword vectors
(variational or analytical), with the flat-index convention of
src/encoder.py (qudit 0 most significant). Dense numpy only; intended
for post-hoc analysis of n <= 7 codes, not for training.

  compute_kl_residuals        worst-case off-diagonal and diagonal-variance
                              violations of the KL conditions over all
                              pairs E_a^dag E_b of a correction set
  compute_distance            smallest error weight at which the
                              off-diagonal KL condition fails (threshold)
  compute_weight_enumerators  Shor-Laflamme A_j, B_j (paper App. A,
                              Table of weight enumerators)
  compute_entanglement_entropy  von Neumann entropy of each codeword
                              across a bipartition

Consumers:
notebooks/central_comparison.ipynb, tests/test_code_catalog.py,
tests/test_enumerators.py.

Paper cross-check: the enumerator formulas match App. A up to the
normalization of the codeword projector, recorded at
compute_weight_enumerators.
"""
import numpy as np
import itertools


def compute_kl_residuals(code_states, E_corr):
    """
    Worst-case Knill-Laflamme residuals over ALL pairs M = E_a^dag E_b.

    The KL conditions for correcting E_corr read
    <psi_i| E_a^dag E_b |psi_j> = alpha_ab delta_ij. For every ordered
    pair (a, b) this evaluates the two ways they can fail (the same two
    terms as the paper's detection loss, Sec. IV B, here unsummed and
    without the K/4 weighting):
      - off-diagonal: |<psi_i| M |psi_j>| for i < j (must vanish);
      - diagonal: Var_k Re<psi_k| M |psi_k> (must vanish, i.e. the
        diagonal must be codeword-independent).
    Cost is O(|E_corr|^2 K^2 dim^2); use on dense correction sets only.

    Args:
        code_states: (K, dim) complex array, rows = normalized codewords.
        E_corr: list of (dim, dim) correction operators (e.g. the E_corr of
            build_native_error_set, identity included).

    Returns:
        (max_offdiag, max_diag_var): floats, the maxima over all pairs.
    """
    K = code_states.shape[0]
    max_offdiag = 0.0
    max_diag_var = 0.0

    for Ea in E_corr:
        Ea_dag = Ea.conj().T
        for Eb in E_corr:
            M = Ea_dag @ Eb
            # Off-diagonal
            for i in range(K):
                for j in range(i + 1, K):
                    val = abs(np.vdot(code_states[i], M @ code_states[j]))
                    max_offdiag = max(max_offdiag, val)
            # Diagonal variance
            diags = np.array([float(np.real(np.vdot(code_states[k], M @ code_states[k])))
                              for k in range(K)])
            max_diag_var = max(max_diag_var, float(np.var(diags)))

    return max_offdiag, max_diag_var


def compute_distance(code_states, n_qudit, d, single_error_basis, max_weight=3):
    """
    Numerical distance probe: smallest error weight that is NOT detected.

    For w = 1, 2, ..., max_weight builds every weight-w tensor product of
    operators from single_error_basis (all w-subsets of qudits, all basis
    choices) and checks only the OFF-DIAGONAL Knill-Laflamme condition:
    detection at weight w fails if max_{i<j} |<psi_i| E |psi_j>| exceeds
    the fixed threshold 0.01. The diagonal-uniformity condition is not
    tested here (see compute_kl_residuals), so this is a necessary-but-
    not-sufficient certificate of distance; it is a quick sanity probe,
    not the paper's distance claim (which rests on the KL residuals,
    LER and enumerators).

    Args:
        code_states: (K, dim) complex array of codewords.
        n_qudit, d: register size and local dimension.
        single_error_basis: list of (d, d) non-identity single-qudit
            operators (e.g. qudit_hardware_error_basis(d)).
        max_weight: largest weight probed.

    Returns:
        int: the first weight w at which detection fails, or
        max_weight + 1 if all weights up to max_weight are detected.
    """
    K = code_states.shape[0]
    Id = np.eye(d, dtype=complex)
    threshold = 0.01

    for w in range(1, max_weight + 1):
        worst = 0.0
        for qudit_subset in itertools.combinations(range(n_qudit), w):
            for err_choice in itertools.product(single_error_basis, repeat=w):
                op = 1
                idx = 0
                for q in range(n_qudit):
                    if q in qudit_subset:
                        op = np.kron(op, err_choice[idx])
                        idx += 1
                    else:
                        op = np.kron(op, Id)
                for i in range(K):
                    for j in range(i + 1, K):
                        val = abs(np.vdot(code_states[i], op @ code_states[j]))
                        worst = max(worst, val)
        if worst > threshold:
            return w
    return max_weight + 1


def compute_weight_enumerators(code_states, n_qudit, d, single_error_basis, max_weight=2):
    """
    Shor-Laflamme weight enumerators A_j, B_j (paper App. A).

        A_j = (1/K^2) sum_{wt(E) = j} |Tr(E P)|^2
        B_j = (1/K)   sum_{wt(E) = j} Tr(E P E^dag P)

    with P = sum_k |psi_k><psi_k| the (trace-K) code-space projector and
    the sum over all weight-j tensor products of operators from
    single_error_basis (all j-subsets of qudits, all basis choices; the
    paper's "single-qudit channel error basis"). With this normalization
    A_0 = B_0 = 1, and A_j = B_j for j < distance is the exact
    Knill-Laflamme detection condition; B_j - A_j >= 0 measures the
    deficit at weight j.

    Paper check: App. A writes the same two formulas but defines
    P = (1/K) sum_k |psi_k><psi_k| (trace 1), which would give
    A_0 = 1/K^2 and scale every A_j, B_j by 1/K^2; the paper's table
    caption states A_0 = B_0 = 1, which is this function's (trace-K
    projector) convention. The code is unchanged; the values in the
    paper's enumerator table are those of this function.

    Args:
        code_states: (K, dim) complex array of codewords.
        n_qudit, d: register size and local dimension.
        single_error_basis: list of (d, d) non-identity single-qudit
            operators (weight-1 generators of the sum).
        max_weight: largest weight j computed (0, 1, 2, ...).

    Returns:
        (A, B): two float arrays of length max_weight + 1, indexed by j.
    """
    K, dim = code_states.shape
    P_c = code_states.T @ np.conj(code_states)  # (dim, dim) projector
    Id = np.eye(d, dtype=complex)

    A = np.zeros(max_weight + 1)
    B = np.zeros(max_weight + 1)

    # Weight 0
    A[0] = np.abs(np.trace(P_c)) ** 2 / K ** 2
    B[0] = np.real(np.trace(P_c @ P_c)) / K

    for w in range(1, max_weight + 1):
        for qudit_subset in itertools.combinations(range(n_qudit), w):
            for err_choice in itertools.product(single_error_basis, repeat=w):
                op = np.array([[1.0]], dtype=complex)
                idx = 0
                for q in range(n_qudit):
                    if q in qudit_subset:
                        op = np.kron(op, err_choice[idx])
                        idx += 1
                    else:
                        op = np.kron(op, Id)
                A[w] += np.abs(np.trace(op @ P_c)) ** 2
                B[w] += np.real(np.trace(op @ P_c @ op.conj().T @ P_c))
        A[w] /= K ** 2
        B[w] /= K

    return A, B


def compute_entanglement_entropy(code_states, bipartition_A, n_qudit, d):
    """
    Von Neumann entanglement entropy S(rho_A) = -Tr rho_A log2 rho_A of
    each codeword across the bipartition A | complement.

    The codeword is reshaped to a rank-n tensor (qudit 0 leading), the A
    qudits are permuted to the front, the tensor is flattened to a
    (d**|A|, d**|B|) matrix psi, and rho_A = psi psi^dag is diagonalized.
    Entropy is in bits (log base 2); maximum is |A| log2 d.

    Args:
        code_states: (K, dim) complex array of codewords.
        bipartition_A: list of qudit indices forming subsystem A.
        n_qudit, d: register size and local dimension.

    Returns:
        list of K floats, S(rho_A) for each codeword in order.
    """
    K = code_states.shape[0]
    n_A = len(bipartition_A)
    n_B = n_qudit - n_A
    d_A = d ** n_A
    d_B = d ** n_B

    bipartition_B = [q for q in range(n_qudit) if q not in bipartition_A]
    perm = list(bipartition_A) + bipartition_B

    entropies = []
    for k in range(K):
        psi = code_states[k].reshape([d] * n_qudit)
        psi = np.transpose(psi, perm)
        psi = psi.reshape(d_A, d_B)
        rho_A = psi @ psi.conj().T
        eigenvalues = np.linalg.eigvalsh(rho_A)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        S = -np.sum(eigenvalues * np.log2(eigenvalues))
        entropies.append(float(S))

    return entropies
