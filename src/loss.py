"""Knill-Laflamme loss functions on the PennyLane / autograd backend.

What it implements
------------------
The detection loss of paper Eq. (12), Sec. IV B (a re-expression of Cao et
al. arXiv:2204.03560 Eq. 16), for a PennyLane QNode encoder, plus the npz
save/load helpers every trainer uses. The JAX backend (src/jax_backend.py)
is the training path of the paper; this module is the backend-independent
reference the JAX losses are checked against
(tests/test_backend_agreement.py) and the path scripts/train.py takes
with --backend pennylane.

Two formulations of the Knill-Laflamme conditions, both zero exactly when
the code detects every operator in the set:

1. Correction-based (kl_loss_fast):
   Term 1: Σ_{E∈E_det} Σ_{i<j} |⟨ψ_i|E|ψ_j⟩|²                (off-diagonal)
   Term 2: Σ_{M∈{E_a†E_b}} (K/4) Var_k(⟨ψ_k|M|ψ_k⟩)         (diagonal uniformity)
   with the products E_a†E_b precomputed by `precompute_error_products`;
   O(|E_corr|²) operators in Term 2.

2. Detection-based (kl_loss_detection_*_minibatch), paper Eq. (12):
   Σ_{E∈E_det} [ Σ_{i<j} |⟨ψ_i|E|ψ_j⟩|² + (K/4) Var_k(⟨ψ_k|E|ψ_k⟩) ]
   a single sum over the completed detection set (Sec. IV C), O(|E_det|).
   Each variant draws a uniform fraction `batch_fraction` of the operators
   per call and scales by |E_det| / n_sample, so the estimate is unbiased
   (batch_fraction = 1 is the full loss).

The (K/4) factor: Var_k is the population variance (1/K) Σ_k |x_k - mean|²,
so (K/4) Var = (1/4) Σ_k |x_k - mean|², the form written in Cao et al.
Eq. 16. Term 2 is only included at distance >= 3; at distance 2 the
detection set is the bare error basis and only Term 1 is enforced.

Error representations:
  - dense (dim, dim) matrices: kl_loss_fast, kl_loss_detection_minibatch
  - factored single-qudit ops, lists of (qudit_idx, d x d):
    kl_loss_detection_factored_minibatch (App. D style application)
  - 1D diagonal vectors: kl_loss_detection_diagonal_minibatch

Consumers: scripts/train.py, scripts/fig6_panel_train.py and
src/seed_race.py (save_varqec_result), src/catalog.py (load_varqec_result),
src/correlated_noise.py (estimate_memory_mb), src/__init__.py re-exports,
tests/test_detection_loss.py, tests/test_qutrit.py, tests/test_benchmarks.py,
tests/test_factored_errors.py, tests/test_backend_agreement.py.
"""
import numpy as np
import pennylane as qml


def estimate_memory_mb(n_operators, dim):
    """Memory (MB) of n_operators dense complex128 matrices of size dim x dim.
    Used by src/correlated_noise.py to cap the dense error-set size."""
    bytes_per_matrix = dim * dim * 16  # complex128 = 16 bytes
    total_bytes = n_operators * bytes_per_matrix
    return total_bytes / (1024 ** 2)


def precompute_error_products(E_corr):
    """Precompute all E_a† @ E_b products for the correction-based loss.

    Call ONCE before the training loop: the products are constant (they do
    not depend on the variational parameters). |E_corr|² matrices, e.g.
    256 for |E_corr| = 16, 5776 for |E_corr| = 76.

    Args:
        E_corr: list of dense (dim, dim) correction-set operators.

    Returns:
        M_products: list of (dim, dim) matrices [Ea†Eb for all ordered pairs]
    """
    M_products = []
    for Ea in E_corr:
        Ea_dag = np.conj(Ea.T)
        for Eb in E_corr:
            M_products.append(Ea_dag @ Eb)
    return M_products


def _kl_overlap_loss(overlaps, K, distance, scale=1.0):
    """One operator's bracket of Eq. (12) from its overlap matrix.

    overlaps[i, j] = <psi_i|E|psi_j>, shape (K, K). Returns
    scale * [ Σ_{i<j} |overlaps[i,j]|² + (K/4) Var_k overlaps[k,k] ], the
    variance term only at distance >= 3; `scale` is the minibatch
    inverse-sampling factor |E_det| / n_sample.
    """
    loss = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            loss = loss + scale * qml.math.abs(overlaps[i, j]) ** 2
    if distance >= 3:
        diag_vals = qml.math.stack([overlaps[k, k] for k in range(K)])
        loss = loss + scale * (K / 4) * qml.math.real(qml.math.var(diag_vals))
    return loss


def kl_loss_fast(params, encoder_fn, E_det, M_products, K, distance):
    """Correction-based KL loss over dense operators (full batch).

    Term 1 runs over E_det, Term 2 over the precomputed products
    M_products = {E_a†E_b}. Passing M_products=[] with a dense completed
    detection set as E_det gives Term 1 only (what scripts/train.py and the
    backend-agreement test do at distance 2).

    Args:
        params: variational parameters, as the encoder expects them.
        encoder_fn: (params, k) -> (dim,) statevector (PennyLane QNode or
            any autograd-compatible callable).
        E_det: list of dense (dim, dim) detection operators.
        M_products: PRECOMPUTED list of E_a†E_b matrices
            (from precompute_error_products), used iff distance >= 3.
        K: number of codewords.
        distance: code distance.

    Returns:
        loss value (PennyLane autograd-compatible scalar)
    """
    # Generate code states and stack into matrix
    code_states = qml.math.stack([encoder_fn(params, k) for k in range(K)])
    # code_states shape: (K, dim)

    loss = 0.0

    # === Term 1: Off-diagonal (orthogonality) ===
    # For each error E, compute <ψ_i|E|ψ_j> for all i<j
    for E in E_det:
        # Apply E to all codewords at once: E @ code_states.T -> (dim, K)
        E_applied = qml.math.tensordot(E, qml.math.transpose(code_states), axes=1)
        # Inner products: code_states.conj() @ E_applied -> (K, K)
        overlaps = qml.math.tensordot(qml.math.conj(code_states), E_applied, axes=[[1], [0]])
        # Sum |overlaps[i,j]|^2 for i < j (only 6 pairs for K=4)
        for i in range(K):
            for j in range(i + 1, K):
                loss = loss + qml.math.abs(overlaps[i, j]) ** 2

    # === Term 2: Diagonal variance (KL condition) ===
    if distance >= 3:
        for M in M_products:
            # Apply M to all codewords: M @ code_states.T -> (dim, K)
            M_applied = qml.math.tensordot(M, qml.math.transpose(code_states), axes=1)
            # Diagonal expectations: <ψ_k|M|ψ_k> = sum over dim of conj(ψ_k) * (M @ ψ_k)
            vals = qml.math.sum(qml.math.conj(code_states) * qml.math.transpose(M_applied), axis=1)
            loss = loss + (K / 4) * qml.math.var(vals)

    return loss



def _apply_factored_qml(state, factors, n_qudits, d):
    """Apply a factored error, a list of (qudit_idx, d x d op) pairs, to a
    flat (d**n_qudits,) autograd statevector by one per-wire tensor
    contraction per factor (App. D): O(d**(n+1)) per factor instead of a
    dense d**n x d**n matmul. An empty factor list is the identity."""
    if not factors:
        return state
    s = state
    shape = tuple([d] * n_qudits)
    for qudit_idx, op in factors:
        s_tensor = qml.math.reshape(s, shape)
        # tensordot with numpy op on autograd state
        op_tensor = qml.math.convert_like(op, s_tensor)
        s_tensor = qml.math.tensordot(op_tensor, s_tensor, axes=([1], [qudit_idx]))
        # moveaxis(src=0, dst=qudit_idx): after tensordot, new axis is at 0
        perm = list(range(1, qudit_idx + 1)) + [0] + list(range(qudit_idx + 1, n_qudits))
        s_tensor = qml.math.transpose(s_tensor, perm)
        s = qml.math.reshape(s_tensor, (-1,))
    return s



def kl_loss_detection_minibatch(params, encoder_fn, E_det, K, distance,
                                batch_fraction=0.2, rng=None):
    """Detection loss of paper Eq. (12) over dense operators, minibatched.

    For each sampled error E in E_det:
      - Off-diagonal: Σ_{i<j} |⟨ψ_i|E|ψ_j⟩|²
      - Diagonal variance (distance >= 3): (K/4) Var_k(⟨ψ_k|E|ψ_k⟩)

    NO E_a†E_b products needed: a single sum over the completed detection
    set (O(|E_det|) instead of O(|E_corr|²)). Equivalent to the
    correction-based loss when E_det is the completed set (Sec. IV C).

    Args:
        params, encoder_fn, K, distance: as in kl_loss_fast.
        E_det: list of dense (dim, dim) operators.
        batch_fraction: fraction of E_det drawn uniformly without
            replacement per call (1.0 = full batch); the sum is scaled by
            |E_det| / n_sample so the estimate is unbiased.
        rng: numpy random Generator (a fresh default_rng() if None).

    Returns the (estimated) loss, autograd-compatible.
    """
    if rng is None:
        rng = np.random.default_rng()

    code_states = qml.math.stack([encoder_fn(params, k) for k in range(K)])
    loss = 0.0

    n_total = len(E_det)
    n_sample = max(1, int(n_total * batch_fraction))
    indices = rng.choice(n_total, n_sample, replace=False)
    scale = n_total / n_sample

    for idx in indices:
        E = E_det[idx]
        E_applied = qml.math.tensordot(E, qml.math.transpose(code_states), axes=1)
        overlaps = qml.math.tensordot(qml.math.conj(code_states), E_applied, axes=[[1], [0]])
        loss = loss + _kl_overlap_loss(overlaps, K, distance, scale)

    return loss


def kl_loss_detection_factored_minibatch(params, encoder_fn, E_det_factors, K, distance,
                                          n_qudits, dim_qudit,
                                          batch_fraction=0.2, rng=None):
    """Detection loss of Eq. (12) over FACTORED errors, minibatched.

    Same as kl_loss_detection_minibatch but each error is a list of
    (qudit_idx, d x d op) pairs applied by per-wire contraction
    (`_apply_factored_qml`, App. D) instead of a dense matrix; memory
    O(|E_det| d²) instead of O(|E_det| d^(2n)).

    Args:
        E_det_factors: list of factored errors.
        n_qudits, dim_qudit: register size n and qudit dimension d.
        others: as in kl_loss_detection_minibatch.
    """
    if rng is None:
        rng = np.random.default_rng()

    code_states = qml.math.stack([encoder_fn(params, k) for k in range(K)])
    loss = 0.0

    n_det = len(E_det_factors)
    n_sample = max(1, int(n_det * batch_fraction))
    idx_det = rng.choice(n_det, n_sample, replace=False)
    scale = n_det / n_sample

    for i in idx_det:
        factors = E_det_factors[i]
        E_codewords = qml.math.stack([
            _apply_factored_qml(code_states[k], factors, n_qudits, dim_qudit)
            for k in range(K)
        ])
        overlaps = qml.math.tensordot(
            qml.math.conj(code_states), qml.math.transpose(E_codewords), axes=[[1], [0]])
        loss = loss + _kl_overlap_loss(overlaps, K, distance, scale)

    return loss


def kl_loss_detection_diagonal_minibatch(params, encoder_fn, E_det_diags, K, distance,
                                          batch_fraction=0.01, rng=None):
    """Detection loss of Eq. (12) over DIAGONAL error operators, minibatched.

    Same as kl_loss_detection_minibatch but each error is its (dim,)
    diagonal (dephasing-type sets), applied as an elementwise product.

    Args:
        E_det_diags: list of (dim,) diagonals.
        others: as in kl_loss_detection_minibatch (batch_fraction default
            0.01 because diagonal sets are typically very large).
    """
    if rng is None:
        rng = np.random.default_rng()

    code_states = qml.math.stack([encoder_fn(params, k) for k in range(K)])
    loss = 0.0
    n_total = len(E_det_diags)

    n_sample = max(1, int(n_total * batch_fraction))
    indices = rng.choice(n_total, n_sample, replace=False)
    scale = n_total / n_sample

    for idx in indices:
        v = E_det_diags[idx]
        v_tensor = qml.math.convert_like(v, code_states)
        weighted_states = v_tensor[None, :] * code_states
        overlaps = qml.math.tensordot(
            qml.math.conj(code_states), qml.math.transpose(weighted_states),
            axes=[[1], [0]]
        )
        loss = loss + _kl_overlap_loss(overlaps, K, distance, scale)

    return loss


def save_varqec_result(filepath, params, losses, noise_type, distance, n_layers, metadata=None):
    """Save trained parameters and the loss trace as an npz.

    Keys: params (n_layers, params_per_layer), losses (per-step trace),
    noise_type, distance, n_layers, final_loss, converged (final loss
    below the 1e-6 certificate), plus every entry of `metadata`
    (src/seed_race.py stores K, n_qudit, d, connections, entangler, the
    certification record, sampler settings and op-EV counts there).
    Readers: `load_varqec_result`, src/catalog.py, src/decoders/_common.py.
    """
    result = {
        'params': np.array(params),
        'losses': np.array(losses),
        'noise_type': noise_type,
        'distance': distance,
        'n_layers': n_layers,
        'final_loss': float(losses[-1]) if losses else None,
        'converged': float(losses[-1]) < 1e-6 if losses else False,
    }
    if metadata:
        result.update(metadata)
    np.savez(filepath, **result)
    print(f"Saved VarQEC result to {filepath} (loss={result['final_loss']:.2e})")


def load_varqec_result(filepath):
    """Load an npz written by `save_varqec_result` as a plain dict
    (0-d arrays for the scalar metadata; allow_pickle for string fields)."""
    data = np.load(filepath, allow_pickle=True)
    return dict(data)
