"""JAX backend: native-gate encoder, wire-grouped detection loss, Adam loop.

What it implements
------------------
- The variational encoder of paper Sec. IV A (`create_jax_encoder`): label
  k on the first qudit, then L layers of [two XY rounds per qudit] ->
  [one MS gate per (edge, transition)] -> [one Z phase per (qudit,
  transition)], (5n + 2|E|)(q-1) parameters per layer. The gate closed forms
  match src/gates.py (XY_gate, Z_gate, MS_gate). `create_jax_encoder_variant`
  swaps the entangling block for the alternatives of Sec. VIII B.
- The Knill-Laflamme detection loss of Eq. (12), Sec. IV B,
      L = sum_{E in E_det} [ sum_{i<j} |<psi_i|E|psi_j>|^2
                             + (K/4) Var_k <psi_k|E|psi_k> ],
  over the wire-grouped detection set of App. D (`create_jax_loss_vmap`,
  `build_varqec_loss`), and its per-operator-weighted form
  (`create_jax_loss_vmap_weighted`, `..._v2`, `build_varqec_loss_weighted`)
  that gives the channel-weighted objective of Sec. IV D and the sampled
  estimates of Sec. IX A when a sampler supplies the weights.
- Two per-step samplers that produce such weight tuples:
  `stratified_weights` (uniform per-wire-group subsampling) and
  `hoeffding_weights` (uniform over the flat detection set). The
  channel-weight importance sampler lives in src/sampling/.
- A plain Adam training loop (`train_jax`) for single runs; the multi-start
  protocol with certified stopping is src/seed_race.py.

Gradients are reverse-mode `jax.grad` (roughly 3-10x faster than
parameter-shift on the cells we train).

Wire-grouped application (App. D). A weight-w error is stored as
(wires, d**w x d**w matrix) and applied to the statevector by a tensor
contraction on those w axes only, cost O(q^(w+1)) instead of O(q^(2n)) for a
dense matrix; errors sharing a wire signature are stacked and applied in one
`jax.vmap` (or, in the v2 loss, one batched einsum per arity class). This
is what makes n = 9, q = 3 tractable. The wire-grouped loss was ported from
U. Holzer's qutrit prototype and generalised to arbitrary d and n_qudit
(shape parameters explicit).

Design decisions worth noting:
- Statevector dtype is complex128 (`jax_enable_x64=True`). complex64
  would break the 1e-6 residual certificate at distance >= 3.
- The scan encoder uses `jax.checkpoint` per layer so memory stays
  O(d**n) rather than O(L * d**n) on the backward pass.
- Errors are grouped by wire signature so a single `jax.vmap` applies
  every error in a group in one tensordot; the Python loop over groups is
  traced away under `@jit` (no `lax.scan`, no `lax.switch`).

Consumers: src/seed_race.py, src/decoders/_common.py (rebuilds encoders
from saved npz), scripts/train.py and the training scripts under scripts/,
the campaign drivers under results/campaigns/, notebooks/, and
tests/test_jax_backend*.py, tests/test_loss_fastpath.py,
tests/test_backend_agreement.py, tests/test_paper_config_repro.py.

Requires: jax, jaxlib, optax.
"""
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import optax
import numpy as np

jax.config.update("jax_enable_x64", True)


# ── JAX gate implementations ──────────────────────────────────────────

def _xy_gate_jax(phi, alpha, level_j, level_k, d):
    """XY rotation on the (level_j, level_k) subspace of one qudit.

    exp(-i alpha/2 (cos(phi) X_jk + sin(phi) Y_jk)) embedded in d x d,
    identity on the other levels; matches src.gates.XY_gate.
    """
    I_jk = jnp.zeros((d, d), dtype=jnp.complex128)
    I_jk = I_jk.at[level_j, level_j].set(1.0)
    I_jk = I_jk.at[level_k, level_k].set(1.0)

    X_jk = jnp.zeros((d, d), dtype=jnp.complex128)
    X_jk = X_jk.at[level_j, level_k].set(1.0)
    X_jk = X_jk.at[level_k, level_j].set(1.0)

    Y_jk = jnp.zeros((d, d), dtype=jnp.complex128)
    Y_jk = Y_jk.at[level_j, level_k].set(-1j)
    Y_jk = Y_jk.at[level_k, level_j].set(1j)

    proj_rest = jnp.eye(d, dtype=jnp.complex128) - I_jk
    c = jnp.cos(alpha / 2) + 0j
    s = jnp.sin(alpha / 2) + 0j
    return proj_rest + c * I_jk + (-1j * s * jnp.cos(phi)) * X_jk + (-1j * s * jnp.sin(phi)) * Y_jk


def _z_gate_jax(theta, level_j, level_k, d):
    """Relative phase exp(+i theta/2) on level_j, exp(-i theta/2) on level_k,
    identity elsewhere; matches src.gates.Z_gate."""
    proj_j = jnp.zeros((d, d), dtype=jnp.complex128).at[level_j, level_j].set(1.0)
    proj_k = jnp.zeros((d, d), dtype=jnp.complex128).at[level_k, level_k].set(1.0)
    proj_rest = jnp.eye(d, dtype=jnp.complex128) - proj_j - proj_k
    return proj_rest + jnp.exp(1j * theta / 2) * proj_j + jnp.exp(-1j * theta / 2) * proj_k


def _ms_gate_jax(phi, theta, level_j, level_k, masks):
    """Two-qudit Mølmer–Sørensen gate on one transition, from precomputed masks.

    `masks` is the 6-tuple from src.gates._build_ms_masks(level_j, level_k, d):
    constant d^2 x d^2 index masks so that the gate is a linear combination
    with theta/phi-dependent scalar coefficients (cheap to re-trace under
    jit). Same closed form as src.gates.MS_gate, verified against Ringbauer
    et al., Nat. Phys. 18, 1053 (2022), Eq. (2).
    """
    M_c, M_p1, M_p0, M_s_minus, M_s_plus, M_s = masks
    c = jnp.exp(-1j * theta / 2) * jnp.cos(theta / 2) + 0j
    s = -1j * jnp.exp(-1j * theta / 2) * jnp.sin(theta / 2) + 0j
    p1 = jnp.exp(-1j * theta / 4) + 0j
    s_minus = s * jnp.exp(-1j * 2 * phi)
    s_plus = s * jnp.exp(1j * 2 * phi)
    return c * M_c + p1 * M_p1 + M_p0 + s_minus * M_s_minus + s_plus * M_s_plus + s * M_s


# ── JAX state-vector operations ───────────────────────────────────────

def _apply_single_gate_jax(state, U, q, n_qudit, d):
    """Apply a d x d gate to qudit q of a flat (d**n_qudit,) statevector.

    Reshapes to (d,)*n, contracts the gate on axis q and moves the new axis
    back to position q: cost O(d**(n+1)), no dense d**n x d**n matrix.
    """
    shape = tuple([d] * n_qudit)
    s = state.reshape(shape)
    s = jnp.tensordot(U, s, axes=[[1], [q]])
    perm = list(range(1, q + 1)) + [0] + list(range(q + 1, n_qudit))
    s = jnp.transpose(s, perm)
    return s.reshape(-1)


def _apply_two_gate_jax(state, U, q1, q2, n_qudit, d):
    """Apply a d^2 x d^2 gate to qudits (q1, q2) of a flat statevector.

    U is reshaped to (d, d, d, d) = (out1, out2, in1, in2) and contracted
    on axes (q1, q2); the inverse permutation restores the qudit order.
    """
    shape = tuple([d] * n_qudit)
    s = state.reshape(shape)
    U_4d = U.reshape(d, d, d, d)
    s = jnp.tensordot(U_4d, s, axes=[[2, 3], [q1, q2]])
    remaining = sorted(set(range(n_qudit)) - {q1, q2})
    perm = [0] * n_qudit
    perm[q1] = 0
    perm[q2] = 1
    for i, r in enumerate(remaining):
        perm[r] = i + 2
    s = jnp.transpose(s, perm)
    return s.reshape(-1)


def _apply_factored_error_jax(state, factors, n_qudit, d):
    """Apply a factored error, a list of (qudit_idx, d x d op) pairs, one
    single-qudit contraction per factor. Used by `create_jax_loss_factored`."""
    s = state
    for q, op in factors:
        s = _apply_single_gate_jax(s, op, q, n_qudit, d)
    return s


def _apply_local_matrix(state, matrix, wires, inverse_perm, d, n_qudit):
    """Apply a (d**w x d**w) matrix to the selected wires of a statevector.

    The wire-grouped error application of paper App. D: a weight-w error is
    stored as (wires, d**w x d**w matrix) and applied by contracting only
    the w state axes it acts on, cost O(d**(n+w)) per application instead
    of O(d**(2n)) for a dense d**n x d**n operator. Ported from U. Holzer's
    qutrit prototype and generalised to any d and n_qudit.

    Args:
        state: flat (d**n_qudit,) complex statevector.
        matrix: (d**w, d**w) complex matrix, row index = output levels of
            the wires in `wires` order, column index = input levels.
        wires: tuple (w0, ..., w_{w-1}) of qudit indices acted on.
        inverse_perm: length-n_qudit permutation restoring the original
            axis order after the contraction has moved the `wires` axes to
            the front (precomputed in ErrorModel.build_grouped).
        d, n_qudit: qudit dimension and count (static under jit).

    Returns the flat (d**n_qudit,) result. For wires=() and
    matrix=[[1.0]] the call is a no-op (scalar broadcast).

    Vmap-friendly over `matrix` with `in_axes=(None, 0, None, None, None, None)`.
    """
    shape = (d,) * n_qudit
    state_tensor = jnp.reshape(state, shape)
    w = len(wires)
    matrix_tensor = jnp.reshape(matrix, (d,) * (2 * w))
    axes = list(range(w, 2 * w))
    contracted = jnp.tensordot(matrix_tensor, state_tensor,
                               axes=(axes, list(wires)))
    state_tensor = jnp.transpose(contracted, inverse_perm)
    return jnp.reshape(state_tensor, (d ** n_qudit,))


def create_jax_loss_vmap(E_det_grouped, K, d, n_qudit, distance):
    """Build the jit-compiled detection loss of Eq. (12) over grouped errors.

    Computes, for codewords psi_0..psi_{K-1},
        L = sum_E [ sum_{i<j} |<psi_i|E|psi_j>|^2
                    + (K/4) Var_k <psi_k|E|psi_k> ]
    (Term 1: off-diagonal Knill-Laflamme condition; Term 2: diagonal
    uniformity). Ported from U. Holzer's qutrit prototype and generalised
    to any d and n_qudit.

    Algorithm (App. D). Errors sharing a wire signature are stacked and
    applied by a single `jax.vmap` of `_apply_local_matrix`. Term 1 vmaps
    over the matrix axis and iterates the K*(K-1)/2 codeword pairs in
    Python (K is small). Term 2 (distance >= 3) uses a double vmap, outer
    over matrices, inner over the K codewords, to compute all
    <psi_k|E|psi_k> in one shot. The whole closure is `@jax.jit`, so the
    Python loop over groups is traced away at compile time; there is no
    `lax.scan` and no `lax.switch`.

    Args:
        E_det_grouped: list of dicts from ErrorModel.build_grouped(), each
            {'wires': tuple, 'matrices': (n_g, d**w, d**w) complex,
            'inverse_perm': tuple}; the identity sentinel group (wires=(),
            matrices [[1.0]]) contributes 0 to both terms.
        K: number of codewords (= d for one logical qudit).
        d: qudit dimension.
        n_qudit: number of physical qudits.
        distance: code distance. Term 2 is included iff distance >= 3
            (at distance 2 the detection set is the bare basis and only the
            off-diagonal condition is enforced).

    Returns:
        loss_fn(code_states) -> float64 scalar, where code_states has shape
        (K, d**n_qudit).
    """
    prepared = []
    for g in E_det_grouped:
        wires = g['wires']
        inverse_perm = g['inverse_perm']
        matrices = jnp.asarray(g['matrices'], dtype=jnp.complex128)
        prepared.append((wires, inverse_perm, matrices))

    vmap_apply_E = jax.vmap(
        _apply_local_matrix,
        in_axes=(None, 0, None, None, None, None))
    vmap_apply_M = jax.vmap(
        jax.vmap(_apply_local_matrix,
                 in_axes=(0, None, None, None, None, None)),
        in_axes=(None, 0, None, None, None, None))

    @jit
    def loss_fn(code_states):
        loss = jnp.array(0.0, dtype=jnp.float64)
        for wires, inverse_perm, matrices in prepared:
            # Term 1: Σ_{E in group} Σ_{i<j} |<psi_i|E|psi_j>|^2
            for i in range(K):
                for j in range(i + 1, K):
                    E_cj = vmap_apply_E(
                        code_states[j], matrices, wires, inverse_perm,
                        d, n_qudit)
                    inner = jnp.sum(jnp.conj(code_states[i]) * E_cj, axis=1)
                    loss = loss + jnp.sum(jnp.abs(inner) ** 2)

            if distance >= 3:
                # Term 2: (K/4) Σ_{E in group} Var_k <psi_k|E|psi_k>
                # Var_k is the population variance (1/K) Σ_k |x_k - mean|^2,
                # so (K/4) Var = (1/4) Σ_k |x_k - mean|^2, the diagonal term
                # of Cao et al. arXiv:2204.03560 Eq. 16 (paper Sec. IV B).
                # The prototype used K/3; both vanish at the KL fixed point
                # but K/4 keeps the paper's gradient balance between the
                # two terms and matches src/loss.py.
                M_psi = vmap_apply_M(
                    code_states, matrices, wires, inverse_perm,
                    d, n_qudit)
                vals = jnp.sum(
                    jnp.conj(code_states)[None, :, :] * M_psi, axis=2)
                loss = loss + (K / 4) * jnp.sum(
                    jnp.real(jnp.var(vals, axis=1)))
        return loss

    return loss_fn


# ── JAX encoder factory ───────────────────────────────────────────────

def create_jax_encoder(n_qudit, d, connections=None, use_scan=True):
    """Create the native-gate encoder of paper Sec. IV A (Fig. ansatz).

    The encoder maps a logical label k to |psi_k> = U(params) |k, 0, ..., 0>.
    Each layer applies, in this order, with its own slice of params:
      1. two rounds of XY rotations on every qudit, one per adjacent
         transition (t, t+1), t = 0..d-2, two parameters (phi, alpha) each
         -> 4 (d-1) parameters per qudit;
      2. one MS gate per (edge, transition) of `connections`, two
         parameters (phi, theta) each -> 2 (d-1) per edge;
      3. one Z phase per (qudit, transition), one parameter each
         -> (d-1) per qudit.
    Total (5 n_qudit + 2 |E|)(d-1) parameters per layer, and L |E| (d-1)
    MS gates per L-layer encoder (the "MS budget" of Eq. ms-budget).

    Args:
        n_qudit: number of physical qudits.
        d: qudit dimension.
        connections: list of [q1, q2] edges the MS gates act on; default
            is the ring. The paper uses all-to-all (see src/seed_race.py).
        use_scan: if True (default), layers run under jax.lax.scan with a
            per-layer jax.checkpoint (fast compile, O(d**n) backward
            memory). False unrolls all layers (easier to debug; the two
            agree to numerical precision, tests/test_jax_backend.py).

    Returns: (encoder_fn, connections, params_per_layer) where
        encoder_fn(params, code_ind) -> (d**n_qudit,) complex128 state for
        params of shape (n_layers, params_per_layer) and code_ind in
        range(d); the initial state puts code_ind on the first qudit.
    """
    if connections is None:
        connections = [[i, (i + 1) % n_qudit] for i in range(n_qudit)]

    n_transitions = d - 1
    n_conn = len(connections)
    params_per_layer = (5 * n_qudit + 2 * n_conn) * n_transitions
    dim = d ** n_qudit

    from src.gates import _build_ms_masks
    ms_masks = {}
    for t in range(n_transitions):
        masks_np = _build_ms_masks(t, t + 1, d)
        ms_masks[(t, t+1)] = tuple(jnp.array(m, dtype=jnp.complex128) for m in masks_np)

    connections_tuple = tuple(tuple(c) for c in connections)

    def _apply_layer(state, layer_p):
        """Apply one ansatz layer (shared between scan and unrolled paths)."""
        pi = 0
        for q in range(n_qudit):
            for t in range(n_transitions):
                U = _xy_gate_jax(layer_p[pi], layer_p[pi+1], t, t+1, d)
                state = _apply_single_gate_jax(state, U, q, n_qudit, d)
                pi += 2
            for t in range(n_transitions):
                U = _xy_gate_jax(layer_p[pi], layer_p[pi+1], t, t+1, d)
                state = _apply_single_gate_jax(state, U, q, n_qudit, d)
                pi += 2
        for q1, q2 in connections_tuple:
            for t in range(n_transitions):
                U = _ms_gate_jax(layer_p[pi], layer_p[pi+1], t, t+1, ms_masks[(t, t+1)])
                state = _apply_two_gate_jax(state, U, q1, q2, n_qudit, d)
                pi += 2
        for q in range(n_qudit):
            for t in range(n_transitions):
                U = _z_gate_jax(layer_p[pi], t, t+1, d)
                state = _apply_single_gate_jax(state, U, q, n_qudit, d)
                pi += 1
        return state

    def _init_state(code_ind):
        state = jnp.zeros(dim, dtype=jnp.complex128)
        return state.at[code_ind * d ** (n_qudit - 1)].set(1.0)

    if use_scan:
        @jax.checkpoint
        def _checkpointed_layer(state, layer_p):
            return _apply_layer(state, layer_p)

        def encoder(params, code_ind):
            state = _init_state(code_ind)
            state, _ = jax.lax.scan(
                lambda s, p: (_checkpointed_layer(s, p), None), state, params)
            return state
    else:
        def encoder(params, code_ind):
            state = _init_state(code_ind)
            for l in range(params.shape[0]):
                state = _apply_layer(state, params[l])
            return state

    return encoder, connections, params_per_layer


def create_jax_encoder_scan(n_qudit, d, connections=None):
    """Backward compat: use create_jax_encoder(use_scan=True) instead."""
    return create_jax_encoder(n_qudit, d, connections, use_scan=True)


VARIANT_ENTANGLERS = ("ms", "ms1", "zz", "ls", "csum", "cex")


def create_jax_encoder_variant(n_qudit, d, connections=None, entangler="ms",
                               use_scan=True):
    """The Sec. IV A ansatz with a swappable entangling block (paper Sec. VIII B).

    Layer structure is the SAME as create_jax_encoder for every entangler —
    two single-qudit XY rounds, the entangler block on `connections`, one Z
    round — only the entangler differs:

      ms   : delegates verbatim to create_jax_encoder (physical XX/YY
             Mølmer–Sørensen, Ringbauer Eq. 2); 2 params per (edge,
             transition).
      ms1  : single-transition MS: the SAME physical MS gate, applied on
             transition (0,1) ONLY — one gate per edge per layer instead
             of one per (edge, transition), so the entangling budget is
             L·|E| instead of L·|E|·(d-1); 2 params per edge.
      zz   : subspace ZZ (gates.zz_subspace_gate, Sec. II A; a diagonal
             exp(-i theta/2 z_a z_b) with z = +1/-1 on the transition's
             two levels, not a light-shift gate); 1 param per (edge,
             transition), applied per adjacent transition like ms.
      ls   : global light-shift gate (gates.ls_global_gate, Hrmo et al.
             arXiv:2206.04104 Eq. 3), diagonal phase exp(i theta) on every
             level pair with unequal levels; 1 param per edge.
      csum : qudit SUM (gates.CSUM_gate), parameter-free constant.
      cex  : embedded-qubit exchange (gates.CEX_gate defaults), parameter-free.

    params_per_layer: ms (5n+2|E|)(d-1); ms1 5n(d-1)+2|E|;
    zz 5n(d-1)+|E|(d-1); ls 5n(d-1)+|E|; csum/cex 5n(d-1).

    Returns (encoder_fn, connections, params_per_layer) like
    create_jax_encoder.
    """
    if entangler not in VARIANT_ENTANGLERS:
        raise ValueError(f"entangler must be one of {VARIANT_ENTANGLERS}, "
                         f"got '{entangler}'")
    if entangler == "ms":
        return create_jax_encoder(n_qudit, d, connections, use_scan=use_scan)

    if connections is None:
        connections = [[i, (i + 1) % n_qudit] for i in range(n_qudit)]
    connections_tuple = tuple(tuple(c) for c in connections)
    n_transitions = d - 1
    n_conn = len(connections_tuple)
    dim = d ** n_qudit

    single_ppl = 5 * n_qudit * n_transitions  # 2 XY rounds (4) + Z round (1)
    if entangler == "ms1":
        ent_ppl = 2 * n_conn
    elif entangler == "zz":
        ent_ppl = n_conn * n_transitions
    elif entangler == "ls":
        ent_ppl = n_conn
    else:  # csum / cex: parameter-free
        ent_ppl = 0
    params_per_layer = single_ppl + ent_ppl

    # Static structures for the entangler block.
    if entangler == "ms1":
        from src.gates import _build_ms_masks
        ms1_masks = tuple(jnp.array(m, dtype=jnp.complex128)
                          for m in _build_ms_masks(0, 1, d))
    elif entangler == "zz":
        # zz_subspace_gate(theta, t, t+1, d) = diag(exp(-i theta/2 z_a z_b)),
        # z = +1 on level t, -1 on level t+1, 0 elsewhere.
        zz_vecs = []
        for t in range(n_transitions):
            z = np.zeros(d)
            z[t], z[t + 1] = 1.0, -1.0
            zz_vecs.append(jnp.asarray(np.outer(z, z).reshape(-1)))
    elif entangler == "ls":
        off = np.ones((d, d)) - np.eye(d)
        ls_mask = jnp.asarray(off.reshape(-1))
    else:
        from src.gates import CSUM_gate, CEX_gate
        const_np = CSUM_gate(d=d) if entangler == "csum" else CEX_gate(d=d)
        const_mat = jnp.asarray(np.asarray(const_np), dtype=jnp.complex128)

    def _apply_layer(state, layer_p):
        pi = 0
        for q in range(n_qudit):
            for t in range(n_transitions):
                U = _xy_gate_jax(layer_p[pi], layer_p[pi + 1], t, t + 1, d)
                state = _apply_single_gate_jax(state, U, q, n_qudit, d)
                pi += 2
            for t in range(n_transitions):
                U = _xy_gate_jax(layer_p[pi], layer_p[pi + 1], t, t + 1, d)
                state = _apply_single_gate_jax(state, U, q, n_qudit, d)
                pi += 2
        for q1, q2 in connections_tuple:
            if entangler == "ms1":
                U = _ms_gate_jax(layer_p[pi], layer_p[pi + 1], 0, 1,
                                 ms1_masks)
                state = _apply_two_gate_jax(state, U, q1, q2, n_qudit, d)
                pi += 2
            elif entangler == "zz":
                for t in range(n_transitions):
                    U = jnp.diag(jnp.exp(-0.5j * layer_p[pi] * zz_vecs[t]))
                    state = _apply_two_gate_jax(state, U, q1, q2, n_qudit, d)
                    pi += 1
            elif entangler == "ls":
                U = jnp.diag(jnp.exp(1j * layer_p[pi] * ls_mask))
                state = _apply_two_gate_jax(state, U, q1, q2, n_qudit, d)
                pi += 1
            else:
                state = _apply_two_gate_jax(state, const_mat, q1, q2,
                                            n_qudit, d)
        for q in range(n_qudit):
            for t in range(n_transitions):
                U = _z_gate_jax(layer_p[pi], t, t + 1, d)
                state = _apply_single_gate_jax(state, U, q, n_qudit, d)
                pi += 1
        return state

    def _init_state(code_ind):
        state = jnp.zeros(dim, dtype=jnp.complex128)
        return state.at[code_ind * d ** (n_qudit - 1)].set(1.0)

    if use_scan:
        @jax.checkpoint
        def _checkpointed_layer(state, layer_p):
            return _apply_layer(state, layer_p)

        def encoder(params, code_ind):
            state = _init_state(code_ind)
            state, _ = jax.lax.scan(
                lambda s, p: (_checkpointed_layer(s, p), None), state, params)
            return state
    else:
        def encoder(params, code_ind):
            state = _init_state(code_ind)
            for l in range(params.shape[0]):
                state = _apply_layer(state, params[l])
            return state

    return encoder, connections, params_per_layer


# ── JAX loss functions ────────────────────────────────────────────────

def create_jax_loss(encoder_fn, E_det_np, K, distance):
    """Reference detection loss of Eq. (12) over DENSE error matrices.

    The straightforward implementation: every error is a (d**n, d**n)
    matrix, applied by a dense matmul inside @jit (the Python loop over
    errors is traced at compile time). Memory O(|E_det| d**(2n)), so only
    usable at small n; kept as the ground truth the wire-grouped path is
    tested against (tests/test_jax_backend.py), not used in training.

    Args:
        encoder_fn: (params, code_ind) -> statevector.
        E_det_np: list of dense (d**n, d**n) numpy error matrices.
        K: number of codewords.
        distance: code distance; the variance term enters iff >= 3.

    Returns loss_fn(params) -> scalar.
    """
    E_det_jax = jnp.stack([jnp.array(E, dtype=jnp.complex128) for E in E_det_np])
    n_errors = len(E_det_np)

    @jit
    def loss_fn(params):
        code_states = jnp.stack([encoder_fn(params, k) for k in range(K)])
        loss = 0.0

        for e_idx in range(n_errors):
            E = E_det_jax[e_idx]
            E_applied = E @ code_states.T  # (dim, K)
            overlaps = jnp.conj(code_states) @ E_applied  # (K, K)

            for i in range(K):
                for j in range(i + 1, K):
                    loss = loss + jnp.abs(overlaps[i, j]) ** 2

            if distance >= 3:
                diag_vals = jnp.array([overlaps[k, k] for k in range(K)])
                loss = loss + (K / 4) * jnp.real(jnp.var(diag_vals))

        return loss

    return loss_fn


def create_jax_loss_factored(encoder_fn, E_det_factors, n_qudit, d, K, distance):
    """Reference detection loss of Eq. (12) over FACTORED errors.

    Each error is a list of (qudit_idx, d x d matrix) pairs applied one
    qudit at a time (the per-wire contraction of App. D, without the
    wire-signature batching). Memory O(d**n) per state instead of
    O(d**(2n)) per dense operator, but one dispatch per error. Only used
    by tests/test_jax_backend.py as a cross-check of the dense path; the
    training path is `build_varqec_loss` / `build_varqec_loss_weighted`.

    Args:
        encoder_fn: (params, code_ind) -> statevector.
        E_det_factors: list of errors, each a list of (qudit_idx, op).
        n_qudit, d: qudit count and dimension.
        K: number of codewords.
        distance: code distance; the variance term enters iff >= 3.

    Returns loss_fn(params) -> scalar.
    """
    # Convert factors to JAX arrays
    E_det_jax_factors = []
    for factors in E_det_factors:
        jax_factors = [(q, jnp.array(op, dtype=jnp.complex128)) for q, op in factors]
        E_det_jax_factors.append(jax_factors)

    n_errors = len(E_det_factors)
    mask_upper = jnp.triu(jnp.ones((K, K), dtype=bool), k=1)

    @jit
    def loss_fn(params):
        code_states = jnp.stack([encoder_fn(params, k) for k in range(K)])
        loss = 0.0

        for e_idx in range(n_errors):
            factors = E_det_jax_factors[e_idx]
            E_states = jnp.stack([
                _apply_factored_error_jax(code_states[k], factors, n_qudit, d)
                for k in range(K)
            ])
            overlaps = jnp.conj(code_states) @ E_states.T  # (K, K)

            # Off-diagonal
            loss = loss + jnp.sum(jnp.abs(overlaps[mask_upper]) ** 2)

            # Diagonal variance
            if distance >= 3:
                diag_vals = jnp.diag(overlaps)
                loss = loss + (K / 4) * jnp.real(jnp.var(diag_vals))

        return loss

    return loss_fn


# ── Loss factory (wire-grouped vmap path, the training path) ─────────

def build_varqec_loss(encoder_fn, K, d, n_qudit, distance, E_det_grouped):
    """Return the unweighted full-batch detection loss as a function of params.

    Composes the encoder (vmapped over the K labels) with
    `create_jax_loss_vmap` into a single `@jax.jit` closure. Callers build
    the grouped-error representation with `ErrorModel.build_grouped()`.
    This is the objective of the unweighted protocol variants
    (paper Sec. IV D, Sec. VIII A); scripts/train.py uses it.

    Args:
        encoder_fn: (params, code_ind) -> statevector.
        K: number of codewords.
        d, n_qudit: qudit dimension and count.
        distance: code distance.
        E_det_grouped: output of ErrorModel.build_grouped().

    Returns:
        loss_fn(params) -> scalar.
    """
    if E_det_grouped is None:
        raise ValueError("build_varqec_loss requires E_det_grouped")
    core = create_jax_loss_vmap(E_det_grouped, K, d, n_qudit, distance)
    enc_vmapped = jax.vmap(encoder_fn, in_axes=(None, 0))

    @jit
    def loss_fn(params):
        code_states = enc_vmapped(params, jnp.arange(K))
        return core(code_states)

    return loss_fn


# ── Per-operator-weighted loss (channel weighting + sampled estimates) ─


def create_jax_loss_vmap_weighted(E_det_grouped, K, d, n_qudit, distance):
    """Like create_jax_loss_vmap, but each operator's term carries a weight.

    Computes sum_g sum_e w[g][e] * [ sum_{i<j} |<psi_i|E_e|psi_j>|^2
    + (K/4) Var_k <psi_k|E_e|psi_k> ], the same Term 1 / Term 2 structure
    as create_jax_loss_vmap with every operator's bracket multiplied by its
    weight. One weight tuple serves three purposes:
      - all ones: the full unweighted loss, exactly;
      - the operator's channel probability (src.meth_weights): the
        channel-weighted objective of paper Sec. IV D;
      - a sampler's inverse-probability weights: an unbiased single-step
        estimate of either loss (Sec. IX A). Uniform per-group subsampling
        (`stratified_weights`) sets N_g / n_g_sampled on the chosen
        operators and 0 elsewhere; `hoeffding_weights` sets
        N_total / n_samples; the importance sampler of src/sampling sets
        channel weight / draw probability.

    Returns loss_fn(code_states, weights_per_group) -> float64 scalar,
    jit-compiled, where code_states has shape (K, d**n_qudit) and
    weights_per_group is a tuple aligned with E_det_grouped whose entry g
    has shape (n_g,). Zero-weight operators are still multiplied through
    (cost is paid), which is why the op-EV accounting counts non-zero
    weights rather than wall time.
    """
    prepared = []
    for g in E_det_grouped:
        wires = g['wires']
        inverse_perm = g['inverse_perm']
        matrices = jnp.asarray(g['matrices'], dtype=jnp.complex128)
        prepared.append((wires, inverse_perm, matrices))

    vmap_apply_E = jax.vmap(
        _apply_local_matrix,
        in_axes=(None, 0, None, None, None, None))
    vmap_apply_M = jax.vmap(
        jax.vmap(_apply_local_matrix,
                 in_axes=(0, None, None, None, None, None)),
        in_axes=(None, 0, None, None, None, None))

    @jit
    def loss_fn(code_states, weights_per_group):
        loss = jnp.array(0.0, dtype=jnp.float64)
        for idx, (wires, inverse_perm, matrices) in enumerate(prepared):
            w = weights_per_group[idx]  # (n_g,)
            # Term 1: Σ_e w_e Σ_{i<j} |<psi_i|E_e|psi_j>|^2
            for i in range(K):
                for j in range(i + 1, K):
                    E_cj = vmap_apply_E(
                        code_states[j], matrices, wires, inverse_perm,
                        d, n_qudit)
                    inner = jnp.sum(jnp.conj(code_states[i]) * E_cj, axis=1)
                    loss = loss + jnp.sum(w * (jnp.abs(inner) ** 2))

            if distance >= 3:
                M_psi = vmap_apply_M(
                    code_states, matrices, wires, inverse_perm,
                    d, n_qudit)
                vals = jnp.sum(
                    jnp.conj(code_states)[None, :, :] * M_psi, axis=2)
                # weighted variance contribution per error
                var_per_err = jnp.real(jnp.var(vals, axis=1))
                loss = loss + (K / 4) * jnp.sum(w * var_per_err)
        return loss

    return loss_fn


def create_jax_loss_vmap_weighted_v2(E_det_grouped, K, d, n_qudit, distance):
    """Cross-group batched version of create_jax_loss_vmap_weighted.

    Same contract: loss_fn(code_states, weights_per_group) with
    weights_per_group a tuple aligned with E_det_grouped, ones = full loss.
    This is the default core of `build_varqec_loss_weighted` and hence the
    loss behind every channel-weighted and sampled run.

    Instead of one vmap dispatch per wire group (16 at n=5, 29 at n=7),
    groups are batched by arity (= len(wires)): every group's matrix stack
    in an arity class is applied in a single batched einsum GEMM — with an
    explicit (G, n_g, a, a) group axis when all stacks in the class have
    the same size (true for the completed bases used in the paper;
    broadcasting S over the op axis avoids an index-gather whose backward
    pass is a large scatter-add, measured 0.48x vs 1.7-1.9x at n=7), else
    along a flat op axis indexed by an op->group map. This is exact because
    both loss terms are additive over ops (Term 1 and Term 2 are sums of
    w_e * f(E_e)); per-group structure only determines WHICH wires each
    op acts on.

    The wire structure is handled by pre-permuting the codewords once per
    group: <phi|E|psi> is invariant under any relabeling of the state
    axes, so with psi_g = transpose(psi, wires + rest).reshape(a, b)
    (a = d**w, b = d**(n-w)) the overlap is vdot(phi_g, M @ psi_g) — no
    inverse_perm transpose-back is needed at all. Agreement with the v1
    path is pinned to 1e-12 (loss) / 1e-10 (grad) by
    tests/test_loss_fastpath.py.
    """
    # Group the wire groups into arity classes; record, per class, the
    # member group indices (for weight concatenation), each member's
    # front-wires permutation, the concatenated matrix stack, and the
    # op -> class-local group index map (None when stack sizes are uniform).
    classes = {}
    for gi, g in enumerate(E_det_grouped):
        wires = tuple(int(w) for w in g['wires'])
        classes.setdefault(len(wires), []).append((gi, wires, g['matrices']))

    prepared = []
    for w_arity in sorted(classes):
        members = classes[w_arity]
        a = d ** w_arity
        group_indices = [gi for gi, _, _ in members]
        perms = [wires + tuple(q for q in range(n_qudit) if q not in wires)
                 for _, wires, _ in members]
        sizes = [int(np.asarray(m).shape[0]) for _, _, m in members]
        if len(set(sizes)) == 1:
            # Uniform stack sizes: keep the group axis explicit,
            # (G, n_g, a, a) — broadcasting S over the op axis avoids an
            # index-gather whose backward pass is a large scatter-add.
            mats = jnp.stack(
                [jnp.asarray(m, dtype=jnp.complex128).reshape(-1, a, a)
                 for _, _, m in members])
            op_to_group = None
        else:
            mats = jnp.concatenate(
                [jnp.asarray(m, dtype=jnp.complex128).reshape(-1, a, a)
                 for _, _, m in members], axis=0)
            op_to_group = jnp.asarray(
                np.repeat(np.arange(len(members)), sizes), dtype=jnp.int32)
        prepared.append((w_arity, group_indices, perms, mats, op_to_group))

    iu, ju = np.triu_indices(K, k=1)
    diag_idx = jnp.arange(K)

    @jit
    def loss_fn(code_states, weights_per_group):
        loss = jnp.array(0.0, dtype=jnp.float64)
        psi_t = code_states.reshape((K,) + (d,) * n_qudit)
        for w_arity, group_indices, perms, mats, op_to_group in prepared:
            a = d ** w_arity
            # Per-group permuted codewords: (G, K, a, b), the group's wires
            # moved to the front and merged into the row index a = d**w.
            S = jnp.stack([
                jnp.transpose(psi_t, (0,) + tuple(q + 1 for q in perm))
                .reshape(K, a, -1)
                for perm in perms])
            w_cat = jnp.concatenate(
                [weights_per_group[gi] for gi in group_indices])
            if op_to_group is None:
                # mats: (G, n_g, a, a); broadcast S over the op axis.
                # applied[g,e,k] = E_e psi_k restricted to the group's wires.
                applied = jnp.einsum('geab,gkbc->gekac', mats, S)
                overlaps = jnp.einsum('giab,gejab->geij',
                                      jnp.conj(S), applied)
                overlaps = overlaps.reshape(-1, K, K)
            else:
                # Ragged stack sizes: index each op's permuted codewords.
                gathered = S[op_to_group]  # (E, K, a, b)
                applied = jnp.einsum('eab,ekbc->ekac', mats, gathered)
                overlaps = jnp.einsum('eiab,ejab->eij',
                                      jnp.conj(gathered), applied)
            # overlaps[e, i, j] = <psi_i | E_e | psi_j>, e class-flat
            # Term 1: Σ_e w_e Σ_{i<j} |<psi_i|E_e|psi_j>|^2
            t1 = jnp.sum(jnp.abs(overlaps[:, iu, ju]) ** 2, axis=1)
            loss = loss + jnp.sum(w_cat * t1)
            if distance >= 3:
                # Term 2: (K/4) Σ_e w_e Var_k <psi_k|E_e|psi_k>
                diag = overlaps[:, diag_idx, diag_idx]
                var_per_err = jnp.real(jnp.var(diag, axis=1))
                loss = loss + (K / 4) * jnp.sum(w_cat * var_per_err)
        return loss

    return loss_fn


def build_varqec_loss_weighted(encoder_fn, K, d, n_qudit, distance,
                               E_det_grouped, fast=True):
    """Per-operator-weighted detection loss as a function of params.

    The training objective of src/seed_race.py: with the channel
    probabilities as weights it is the channel-weighted loss of paper
    Sec. IV D; with a sampler's weights it is the per-step sampled
    estimate of Sec. IX A; with ones it is the full unweighted loss.

    Returns a jit-compiled callable `loss_fn(params, weights_per_group)`
    where weights_per_group is a tuple of jnp arrays, one per wire group,
    each of shape (n_g,). Setting all weights to 1 recovers the full loss;
    uniform subsampling uses zero-or-N_g/n_g_sampled weights
    (`stratified_weights`).

    fast=True (default) selects the cross-group batched core
    (create_jax_loss_vmap_weighted_v2, measured 1.7-1.9x faster at n=7 and
    2.1x at n=5 on value_and_grad); fast=False keeps the per-group vmap
    core for debugging. The two agree to 1e-12 (loss) / 1e-10 (grad),
    pinned by tests/test_loss_fastpath.py.
    """
    if E_det_grouped is None:
        raise ValueError("build_varqec_loss_weighted requires E_det_grouped")
    make_core = (create_jax_loss_vmap_weighted_v2 if fast
                 else create_jax_loss_vmap_weighted)
    core = make_core(E_det_grouped, K, d, n_qudit, distance)
    enc_vmapped = jax.vmap(encoder_fn, in_axes=(None, 0))

    @jit
    def loss_fn(params, weights_per_group):
        code_states = enc_vmapped(params, jnp.arange(K))
        return core(code_states, weights_per_group)

    return loss_fn


def stratified_weights(group_sizes, fraction_per_group, rng, dtype=None):
    """Uniform per-wire-group subsampling weights (the "uniform" sampler).

    For each wire group g of size N_g, draws n_g_sampled = max(1,
    ceil(fraction_per_group * N_g)) operators without replacement and
    returns weights of shape (N_g,) equal to N_g / n_g_sampled on the drawn
    positions and 0 elsewhere (inverse inclusion probability), so the
    weighted loss is an unbiased estimate of the full loss and every wire
    group is represented every step.

    Args:
        group_sizes: list of N_g from ErrorModel.build_grouped().
        fraction_per_group: sampling fraction f in (0, 1].
        rng: numpy random Generator.
        dtype: jnp dtype of the returned arrays (default float64).

    Returns the per-group weight tuple for `build_varqec_loss_weighted`.

    This is the same scheme as the per-group subsampling of U. Holzer's
    qutrit prototype (verified equivalent and unbiased;
    tests/test_sampling_unbiased.py). It draws uniformly within a group and
    is distinct from the channel-weight importance sampler of
    src/sampling/stratified_importance.py used by the paper's primary
    protocol (Sec. IX A).
    """
    if dtype is None:
        dtype = jnp.float64
    weights = []
    for n_g in group_sizes:
        if n_g == 0:
            weights.append(jnp.zeros((0,), dtype=dtype))
            continue
        n_keep = max(1, int(np.ceil(fraction_per_group * n_g)))
        n_keep = min(n_keep, n_g)
        idx = rng.choice(n_g, n_keep, replace=False)
        w = np.zeros(n_g, dtype=float)
        w[idx] = float(n_g) / float(n_keep)
        weights.append(jnp.asarray(w, dtype=dtype))
    return tuple(weights)


def hoeffding_weights(group_sizes, n_samples, rng, dtype=None):
    """Uniform subsampling of the flat detection set (no stratification).

    Draws n_samples operators uniformly without replacement from the
    concatenation of all wire groups (so each group is hit in proportion
    to its size, and small groups may be missed in a step) and assigns
    weight N_total / n_samples to the drawn operators, 0 elsewhere; an
    unbiased estimate of the full loss. Used by the tests as the
    worst-case sampler (a one-operator draw can report a loss of ~0 for a
    code whose full loss is far from 0, which is what the full-batch
    certification guards against).

    Args:
        group_sizes: list of N_g per wire group.
        n_samples: total draws (clipped to N_total).
        rng, dtype: as in `stratified_weights`.

    Returns the per-group weight tuple for `build_varqec_loss_weighted`.
    """
    if dtype is None:
        dtype = jnp.float64
    group_sizes = np.asarray(group_sizes, dtype=int)
    n_total = int(group_sizes.sum())
    if n_total == 0:
        return tuple(jnp.zeros((0,), dtype=dtype) for _ in group_sizes)
    n_samples = min(int(n_samples), n_total)
    flat_idx = rng.choice(n_total, n_samples, replace=False)
    boundaries = np.concatenate([[0], np.cumsum(group_sizes)])
    weights = []
    scale = float(n_total) / float(n_samples)
    for g, n_g in enumerate(group_sizes):
        mask = (flat_idx >= boundaries[g]) & (flat_idx < boundaries[g + 1])
        w = np.zeros(n_g, dtype=float)
        w[(flat_idx[mask] - boundaries[g])] = scale
        weights.append(jnp.asarray(w, dtype=dtype))
    return tuple(weights)


# ── JAX training loop ─────────────────────────────────────────────────

def train_jax(loss_fn, n_layers, params_per_layer, n_steps=2000,
              lr=0.05, lr_switch=0.01, seed=0):
    """Single-run Adam optimisation of a full-batch loss (scripts/train.py).

    Parameters are initialised uniformly in [0, 2pi) from
    jax.random.PRNGKey(seed) (paper Sec. IV D) and updated with optax Adam
    at learning rate `lr`; once the loss drops below 0.1 the optimizer is
    re-created at `lr_switch` (this re-init resets the Adam moments). The
    loop stops early when the loss falls below 1e-6 (the Knill-Laflamme
    certificate of Sec. IV D; `loss_fn` is assumed full-batch, so no
    separate confirmation is needed) and prints the loss every 50 steps.
    The multi-start protocol with sampled steps and certified stopping is
    `src.seed_race.run_race`, which has its own three-stage schedule.

    Args:
        loss_fn: params -> scalar, e.g. from `build_varqec_loss`.
        n_layers, params_per_layer: shape of the parameter array.
        n_steps: maximum number of Adam steps.
        lr, lr_switch: initial learning rate and the rate after the switch.
        seed: PRNG seed of the initialisation.

    Returns:
        (best_params, losses): the parameters with the lowest recorded
        loss (numpy array of shape (n_layers, params_per_layer)) and the
        per-step loss history (list of floats).
    """
    key = jax.random.PRNGKey(seed)
    theta = jax.random.uniform(key, (n_layers, params_per_layer),
                               minval=0.0, maxval=2 * np.pi)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(theta)
    val_grad_fn = value_and_grad(loss_fn)

    best_loss = 1e10
    best_theta = theta
    losses = []
    lr_switched = False

    for step in range(n_steps):
        loss_val, grads = val_grad_fn(theta)
        updates, opt_state = optimizer.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)

        lv = float(loss_val)
        losses.append(lv)

        if lv < best_loss:
            best_loss = lv
            best_theta = theta

        if not lr_switched and lv < 0.1:
            optimizer = optax.adam(lr_switch)
            opt_state = optimizer.init(theta)
            lr_switched = True

        if step % 50 == 0:
            print(f"  step {step:4d} | loss={lv:.4e}")

        if lv < 1e-6:
            print(f"  CONVERGED at step {step}")
            break

    return np.array(best_theta), losses
