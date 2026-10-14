"""
Native trapped-ion encoder (the variational ansatz) for qudits of any
dimension d, on the PennyLane / numpy-autograd backend.

Implements the encoder of paper Sec. IV A ("Variational ansatz", Fig. 2):
the logical label k is prepared as |init_k> = |k, 0, ..., 0> and followed
by L identical-structure layers, each applying in this order
  1. two rounds of XY rotations on every qudit, one rotation per adjacent
     transition (t, t+1), t = 0..d-2  -> 4(d-1) parameters per qudit;
  2. one MS gate per (edge, transition) of the connectivity graph
     -> 2(d-1) parameters per edge;
  3. one Z phase per (qudit, transition) -> (d-1) parameters per qudit;
for P_l = (5 n + 2|E|)(d-1) parameters per layer (paper Sec. IV A) and an
MS budget of L |E| (d-1) two-qudit gates (paper Sec. IV A, MS-budget
equation).

Two numerically equivalent backends share the signature
encoder(params, code_ind) -> state (flat complex vector of length d**n):
  * d == 3 (default): a PennyLane QNode on the `default.qutrit` device;
  * d != 3, or force_manual=True: explicit state-vector contractions with
    pennylane.numpy / qml.math so autograd can differentiate through them
    (PennyLane has no generic qudit device). This path is also faster for
    d = 3 because gradients come from autograd rather than parameter-shift.
The JAX backend (src/jax_backend.py) implements the same layer; tests
test_backend_agreement.py and test_central_comparison.py check that the
two agree.

State-index convention: an n-qudit basis state |k_0 ... k_{n-1}> has flat
index sum_q k_q d**(n-1-q) (qudit 0 most significant); a two-qudit gate
matrix uses index a*d + b with a on the first wire it is applied to.

Consumers: scripts/train.py, src/catalog.py (code catalogue), src/__init__.py
(re-export), notebooks/central_comparison.ipynb, tests/test_native.py,
tests/test_qutrit.py, tests/test_backend_agreement.py,
tests/test_jax_backend.py, tests/test_central_comparison.py.

Paper cross-check: layer structure, gate order, parameter count and
initial state match Sec. IV A; see the "Paper check" notes below. The
only convention to be aware of is that the d=3 QNode path prepares |k>
by a chain of XY(phi=0, alpha=pi) pulses, which gives codeword k the
global phase (-i)^k relative to the manual path (irrelevant for the code
space and for every loss or benchmark, which only see |<psi_i|E|psi_j>|).
"""
import pennylane as qml
from pennylane import numpy as np
from src.gates import XY_gate, Z_gate, MS_gate


def _apply_single_qudit_gate(state, U, q, n_qudit, d):
    """Apply a (d, d) gate U to qudit q of a flat n_qudit state vector.

    The state is viewed as a rank-n tensor s[k_0, ..., k_{n-1}] (qudit 0 is
    the leading axis, matching the flat index sum_q k_q d**(n-1-q)); U is
    contracted with axis q, which moves the output axis to the front, and
    the transpose puts it back at position q.

    Args:
        state: flat complex vector of length d**n_qudit.
        U: (d, d) matrix.
        q: target qudit index in [0, n_qudit).
        n_qudit, d: register size and local dimension.

    Returns:
        flat complex vector of length d**n_qudit (autograd-differentiable).
    """
    shape = tuple([d] * n_qudit)
    s = qml.math.reshape(state, shape)
    s = qml.math.tensordot(U, s, axes=[[1], [q]])
    # tensordot output axes: [out, k_0, ..., k_{q-1}, k_{q+1}, ...]; this
    # perm restores [k_0, ..., k_{q-1}, out, k_{q+1}, ...].
    perm = list(range(1, q + 1)) + [0] + list(range(q + 1, n_qudit))
    s = qml.math.transpose(s, perm)
    return qml.math.reshape(s, (-1,))


def _apply_two_qudit_gate(state, U, q1, q2, n_qudit, d):
    """Apply a (d^2, d^2) gate U to qudits (q1, q2) of a flat state vector.

    U is indexed as U[a_out*d + b_out, a_in*d + b_in] with a on qudit q1
    and b on qudit q2 (the a*d + b convention of src/gates.py), so q1 plays
    the role of the "first"/control wire. q1 > q2 is allowed.

    Args:
        state: flat complex vector of length d**n_qudit.
        U: (d^2, d^2) matrix.
        q1, q2: distinct target qudit indices.
        n_qudit, d: register size and local dimension.

    Returns:
        flat complex vector of length d**n_qudit (autograd-differentiable).
    """
    shape = tuple([d] * n_qudit)
    s = qml.math.reshape(state, shape)

    # Reshape U to (d, d, d, d): U[a_out, b_out, a_in, b_in]
    U_4d = qml.math.reshape(U, (d, d, d, d))

    # Contract input axes of U with state axes q1, q2
    s = qml.math.tensordot(U_4d, s, axes=[[2, 3], [q1, q2]])
    # Result axes: [a_out, b_out, remaining_0, remaining_1, ...] with the
    # untouched qudits in increasing order.

    # Build perm with perm[new_axis] = old_axis so that a_out lands on
    # position q1, b_out on q2 and every other qudit returns to its place.
    remaining = sorted(set(range(n_qudit)) - {q1, q2})
    perm = [0] * n_qudit
    perm[q1] = 0
    perm[q2] = 1
    for i, r in enumerate(remaining):
        perm[r] = i + 2

    s = qml.math.transpose(s, perm)
    return qml.math.reshape(s, (-1,))


def _ansatz_layer_manual(state, layer_p, n_qudit, d, connections):
    """Apply one ansatz layer (paper Sec. IV A, Fig. 2) to a flat state.

    Gate order and parameter layout are the single source of truth for the
    layer; `layer_gate_sequence` at the bottom of this file mirrors them.
    Parameter slots are consumed contiguously from layer_p in the order
    the gates are applied:
      [XY(phi, alpha)] x 2 rounds x (d-1) transitions, per qudit;
      [MS(phi, theta)] x (d-1) transitions, per edge of `connections`;
      [Z(theta)] x (d-1) transitions, per qudit;
    for (5 n_qudit + 2 |E|)(d-1) slots in total. Both XY rounds of a qudit
    are applied before moving to the next qudit; since single-qudit gates
    on different qudits commute this equals the "all qudits, round 1; all
    qudits, round 2" reading of the paper. Only adjacent transitions
    (t, t+1) are driven.

    Args:
        state: flat complex vector of length d**n_qudit.
        layer_p: 1-D parameter vector of this layer (angles in radians).
        n_qudit, d: register size and local dimension.
        connections: iterable of (q1, q2) edges; each MS gate uses q1 as
            its first wire (a*d + b convention).

    Returns:
        flat complex vector of length d**n_qudit.
    """
    n_transitions = d - 1
    param_idx = 0

    # 1. Single-qudit rotations: 2 rounds of XY per transition
    for q in range(n_qudit):
        for t in range(n_transitions):
            U = XY_gate(layer_p[param_idx], layer_p[param_idx + 1], t, t + 1, d)
            state = _apply_single_qudit_gate(state, U, q, n_qudit, d)
            param_idx += 2
        for t in range(n_transitions):
            U = XY_gate(layer_p[param_idx], layer_p[param_idx + 1], t, t + 1, d)
            state = _apply_single_qudit_gate(state, U, q, n_qudit, d)
            param_idx += 2

    # 2. MS entangling gates on all transitions
    for q1, q2 in connections:
        for t in range(n_transitions):
            U_MS = MS_gate(layer_p[param_idx], layer_p[param_idx + 1], t, t + 1, d)
            state = _apply_two_qudit_gate(state, U_MS, q1, q2, n_qudit, d)
            param_idx += 2

    # 3. Z corrections on all transitions
    for q in range(n_qudit):
        for t in range(n_transitions):
            U_z = Z_gate(layer_p[param_idx], t, t + 1, d)
            state = _apply_single_qudit_gate(state, U_z, q, n_qudit, d)
            param_idx += 1

    return state


def _create_manual_encoder(n_qudit, d, connections):
    """Create the explicit state-vector encoder (any d; default for d != 3).

    Returns encoder(params, code_ind) -> state where params has shape
    (n_layer, params_per_layer) and code_ind in [0, d) is the logical
    label. The state is the flat length-d**n_qudit vector obtained by
    applying `_ansatz_layer_manual` n_layer times to |code_ind, 0, ..., 0>.
    """
    dim = d ** n_qudit

    def encoder(params, code_ind):
        n_layer = params.shape[0]

        # State preparation: |code_ind, 0, ..., 0> (paper Sec. IV A,
        # |init_k>); qudit 0 is most significant, so the flat index is
        # code_ind * d**(n_qudit-1).
        state = np.zeros(dim, dtype=complex)
        state[int(code_ind) * d ** (n_qudit - 1)] = 1.0

        for l in range(n_layer):
            state = _ansatz_layer_manual(state, params[l], n_qudit, d, connections)

        return state

    return encoder


def _create_qutrit_qnode_encoder(n_qudit, connections):
    """Create the PennyLane QNode encoder on `default.qutrit` (d = 3 only).

    Same layer structure and parameter layout as `_ansatz_layer_manual`
    (hard-coded for d-1 = 2 transitions); gates are inserted as
    qml.QutritUnitary from the matrices of src/gates.py. Returns
    encoder(params, code_ind) -> flat length-3**n_qudit state. Gradients
    go through PennyLane's QNode machinery (slower than the manual path;
    `force_manual=True` in create_native_encoder switches to the latter).
    """
    dev = qml.device("default.qutrit", wires=n_qudit)

    @qml.qnode(dev, interface="autograd")
    def encoder(params, code_ind):
        n_layer = params.shape[0]

        # State preparation: chain of XY(phi=0, alpha=pi) pulses on wire 0
        # climbs |0> -> |1> -> ... -> |code_ind>. Each pulse is -i X_{t,t+1}
        # on its block, so the prepared state is (-i)^code_ind |code_ind,0..0>:
        # a codeword-dependent global phase relative to the manual path,
        # invisible to the code space and to every loss/benchmark.
        for t in range(int(code_ind)):
            U_prep = XY_gate(0.0, np.pi, t, t + 1, d=3)
            qml.QutritUnitary(U_prep, wires=0)

        for l in range(n_layer):
            layer_p = params[l]
            param_idx = 0

            # 1. Single-qutrit rotations
            for q in range(n_qudit):
                for t in range(2):
                    U = XY_gate(layer_p[param_idx], layer_p[param_idx + 1], t, t + 1, d=3)
                    qml.QutritUnitary(U, wires=q)
                    param_idx += 2
                for t in range(2):
                    U = XY_gate(layer_p[param_idx], layer_p[param_idx + 1], t, t + 1, d=3)
                    qml.QutritUnitary(U, wires=q)
                    param_idx += 2

            # 2. MS entangling gates on both (0,1) and (1,2)
            for q1, q2 in connections:
                for t in range(2):
                    U_MS = MS_gate(layer_p[param_idx], layer_p[param_idx + 1], t, t + 1, d=3)
                    qml.QutritUnitary(U_MS, wires=[q1, q2])
                    param_idx += 2

            # 3. Z corrections
            for q in range(n_qudit):
                for t in range(2):
                    U_z = Z_gate(layer_p[param_idx], t, t + 1, d=3)
                    qml.QutritUnitary(U_z, wires=q)
                    param_idx += 1

        return qml.state()

    return encoder


def create_native_encoder(n_qudit: int, d: int, connections=None, force_manual: bool = False):
    """
    Create the native trapped-ion encoder (paper Sec. IV A ansatz) for
    n_qudit qudits of dimension d.

    Ansatz per layer (d-1 adjacent transitions per qudit):
      1. Single-qudit: 2 rounds x (d-1) XY gates x 2 params = 4(d-1) per qudit
      2. MS entangling: (d-1) MS gates x 2 params per edge   = 2(d-1) per edge
      3. Z phases:      (d-1) Z gates x 1 param per qudit    = (d-1) per qudit

    Total params per layer P_l = (5*n_qudit + 2*|E|) * (d-1), |E| =
    len(connections). An L-layer encoder takes params of shape (L, P_l).

    Paper check: layer order, gate placement and P_l agree with Sec. IV A
    and Fig. 2. Note the DEFAULT connectivity here is the ring
    [i, (i+1) mod n] (for n = 2 this lists the edge (0,1) twice); every
    code reported in the paper uses all-to-all connectivity, |E| =
    n(n-1)/2, which scripts/train.py passes explicitly.

    Args:
        n_qudit: number of physical qudits n.
        d: qudit dimension (any d >= 2; the paper uses 3, 4, 5).
        connections: list of [q1, q2] edges of the connectivity graph; the
            MS gate on an edge uses q1 as its first wire. None -> ring.
        force_manual: if True use the explicit state-vector encoder also
            for d = 3 (autograd gradients instead of the QNode; faster and
            numerically equivalent up to the (-i)^k preparation phase noted
            in the module docstring).

    Returns:
        (encoder_fn, connections, params_per_layer):
          encoder_fn(params, code_ind) -> flat complex state of length
              d**n_qudit, params of shape (n_layer, params_per_layer),
              code_ind an integer logical label in [0, d);
          connections: the edge list actually used;
          params_per_layer: P_l as above.
    """
    if connections is None:
        connections = [[i, (i + 1) % n_qudit] for i in range(n_qudit)]

    n_transitions = d - 1
    n_conn = len(connections)
    params_per_layer = (5 * n_qudit + 2 * n_conn) * n_transitions

    if d == 3 and not force_manual:
        encoder = _create_qutrit_qnode_encoder(n_qudit, connections)
    else:
        encoder = _create_manual_encoder(n_qudit, d, connections)

    return encoder, connections, params_per_layer


def layer_gate_sequence(n_qudit: int, d: int, connections=None):
    """Ordered gate list of ONE ansatz layer (paper Sec. IV A, Fig. 2).

    Mirrors ``_ansatz_layer_manual`` exactly (same gate order, same parameter
    slots) without applying anything, so the layer can be inspected:

      1. two rounds of XY rotations, one per adjacent transition (t, t+1),
         on every qudit: 2 (d-1) gates and 4 (d-1) parameters per qudit;
      2. one MS gate per (edge, transition): 2 (d-1) parameters per edge;
      3. one Z phase per (qudit, transition): (d-1) parameters per qudit.

    Args:
        n_qudit, d: register size and local dimension.
        connections: edge list; defaults to the ring graph like
            ``create_native_encoder`` (all paper codes use all-to-all).

    Returns:
        list of dicts with keys 'gate' ('XY' | 'MS' | 'Z'), 'qudits' (tuple),
        'levels' (t, t+1), 'param_slots' (tuple of indices into the layer's
        parameter vector); the slots are contiguous and exhaust
        (5 n_qudit + 2 |E|)(d-1) parameters.
    """
    if connections is None:
        connections = [[i, (i + 1) % n_qudit] for i in range(n_qudit)]
    n_transitions = d - 1
    seq = []
    idx = 0
    for q in range(n_qudit):
        for _round in range(2):
            for t in range(n_transitions):
                seq.append({'gate': 'XY', 'qudits': (q,), 'levels': (t, t + 1),
                            'param_slots': (idx, idx + 1)})
                idx += 2
    for q1, q2 in connections:
        for t in range(n_transitions):
            seq.append({'gate': 'MS', 'qudits': (int(q1), int(q2)),
                        'levels': (t, t + 1), 'param_slots': (idx, idx + 1)})
            idx += 2
    for q in range(n_qudit):
        for t in range(n_transitions):
            seq.append({'gate': 'Z', 'qudits': (q,), 'levels': (t, t + 1),
                        'param_slots': (idx,)})
            idx += 1
    return seq
