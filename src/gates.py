"""
Native trapped-ion gate library for qudits of any dimension d.

This module is the gate library of paper Sec. II A ("Native gate set"):
two single-qudit gates and the two-qudit entanglers, each returned as a
dense unitary matrix (d x d, or d^2 x d^2 for two-qudit gates).

  Single-qudit (Sec. II A 1, "Single-qudit gates")
    XY_gate(phi, alpha, j, k, d)   rotation in the (j,k) two-level block
    Z_gate(theta, j, k, d)         relative phase between levels j and k
  Two-qudit (Sec. II A 2, "Two-qudit gates")
    MS_gate(phi, theta, j, k, d)   Molmer-Sorensen gate on the (j,k)
                                   transition; the default entangler of
                                   every encoder in the paper
    zz_subspace_gate(theta, j, k)  subspace-ZZ gate, paper Eq. (4)
    ls_global_gate(theta, d)       global light-shift gate, paper Eq. (5)
    CSUM_gate(d), CEX_gate(d,c,t1,t2)   parameter-free entanglers
    CSUB_gate(d)                   CSUM^dagger (library extra, not in paper)

Conventions
  * Levels are 0-indexed; a two-qudit basis state |a,b> has flat index
    a*d + b (first qudit most significant). This is the convention the
    encoders use when they reshape a d^2 x d^2 gate to (d, d, d, d).
  * Every gate acts as the identity on levels outside the addressed pair
    (the "spectator" levels), so all gates are defined for any d >= 2 and
    any level pair j != k, adjacent or not.
  * Matrices are built with pennylane.numpy and qml.math so that the
    continuous angles stay differentiable under autograd; the entries are
    closed-form (no matrix exponential), which is what keeps them
    autograd-safe.

Consumers: src/encoder.py (PennyLane/autograd ansatz), src/jax_backend.py
(re-uses _build_ms_masks, CSUM_gate, CEX_gate, zz_subspace_gate,
ls_global_gate for the JAX encoder), src/stabilizer_encoder.py (CSUM/CSUB
for the [[5,1,3]] benchmark encoder), scripts/train.py, src/__init__.py
(re-exports), tests/test_gates_physical.py, tests/test_native.py,
tests/test_ms1_variant.py, tests/test_encoder_variants.py, and the
notebook notebooks/central_comparison.ipynb.

Paper cross-check: every gate definition below was compared line by line
with Sec. II A of the manuscript; the outcome is recorded in the one-line
"Paper check" note of each docstring.
"""
import pennylane as qml
from pennylane import numpy as np


def XY_gate(phi: float, alpha: float, level_j: int, level_k: int, d: int = 3):
    """
    XY rotation U_XY(phi, alpha; j, k) in the (level_j, level_k) subspace.

    U = P_rest + cos(alpha/2) * I_jk
        - i sin(alpha/2) [cos(phi) X_jk + sin(phi) Y_jk]

    with I_jk = |j><j| + |k><k|, P_rest = I - I_jk (projector on the d-2
    spectator levels), X_jk = |j><k| + |k><j| and Y_jk = -i|j><k| + i|k><j|
    the two-level Pauli operators of the (j,k) block. Acts as the identity
    on every level outside {j, k}.

    Paper check: identical to the XY gate of Sec. II A 1 (paper Eq. (2)),
    including the sign conventions of X_jk and Y_jk.

    Args:
        phi: rotation-axis angle in the XY plane (radians).
        alpha: rotation angle (radians); alpha = pi maps |j> -> -i|k>.
        level_j, level_k: the two addressed levels (0-indexed, j != k).
        d: qudit dimension.

    Returns:
        (d, d) complex unitary; differentiable in phi and alpha.
    """
    I_jk = np.zeros((d, d), dtype=complex)
    I_jk[level_j, level_j] = 1.0
    I_jk[level_k, level_k] = 1.0

    X_jk = np.zeros((d, d), dtype=complex)
    X_jk[level_j, level_k] = 1.0
    X_jk[level_k, level_j] = 1.0

    Y_jk = np.zeros((d, d), dtype=complex)
    Y_jk[level_j, level_k] = -1j
    Y_jk[level_k, level_j] = 1j

    proj_rest = np.eye(d, dtype=complex) - I_jk

    c = qml.math.cos(alpha / 2) + 0j
    s = qml.math.sin(alpha / 2) + 0j
    term_x = (-1j * s * qml.math.cos(phi)) * X_jk
    term_y = (-1j * s * qml.math.sin(phi)) * Y_jk

    return proj_rest + c * I_jk + term_x + term_y


def Z_gate(theta: float, level_j: int, level_k: int, d: int = 3):
    """
    Relative-phase gate U_Z(theta; j, k) between levels j and k.

    U = P_rest + e^{+i theta/2} |j><j| + e^{-i theta/2} |k><k|

    Acts as the identity on every level outside {j, k}; the relative phase
    between |j> and |k> is e^{i theta}.

    Paper check: identical to the Z gate of Sec. II A 1 (paper Eq. (3)).

    Args:
        theta: phase angle (radians).
        level_j, level_k: the two addressed levels (0-indexed, j != k).
        d: qudit dimension.

    Returns:
        (d, d) complex diagonal unitary; differentiable in theta.
    """
    proj_j = np.zeros((d, d), dtype=complex)
    proj_j[level_j, level_j] = 1.0

    proj_k = np.zeros((d, d), dtype=complex)
    proj_k[level_k, level_k] = 1.0

    proj_rest = np.eye(d, dtype=complex) - proj_j - proj_k

    p_j = qml.math.exp(1j * theta / 2)
    p_k = qml.math.exp(-1j * theta / 2)

    return proj_rest + p_j * proj_j + p_k * proj_k


def _build_ms_masks(level_j, level_k, d):
    """
    Build the six 0/1 mask matrices from which MS_gate assembles the gate.

    The MS gate on the (j,k) transition is block-structured in the two-qudit
    computational basis |a,b> (flat index a*d + b). Each basis state is
    classified by how many of its two level indices lie in the active pair
    {j, k}:
      - both active   (a, b in {j,k}):  the 4-dimensional block where the
                                        gate rotates (cos/sin mixing);
      - one active    (exactly one):    a pure phase e^{-i theta/4};
      - none active   (a, b spectators): identity.

    Within the both-active block the generator S_phi^2 couples only
    |jj> <-> |kk> (with phase e^{-/+ 2i phi}) and |jk> <-> |kj> (no phase);
    the diagonal of the block carries the common factor c (see MS_gate).

    The masks are parameter-free; the parameter dependence enters only
    through the scalar coefficients in MS_gate, which keeps the gate
    autograd-differentiable (no matrix exponential). src/jax_backend.py
    re-uses the same masks for the JAX encoder.

    Returns:
        (M_c, M_p1, M_p0, M_s_minus, M_s_plus, M_s): six (d^2, d^2) complex
        0/1 matrices:
          M_c        diagonal projector on the both-active states
          M_p1       diagonal projector on the one-active states
          M_p0       diagonal projector on the no-active states
          M_s_minus  single entry [jj, kk]   (multiplied by s*e^{-2i phi})
          M_s_plus   single entry [kk, jj]   (multiplied by s*e^{+2i phi})
          M_s        entries [jk, kj] and [kj, jk] (multiplied by s)
    """
    dim2 = d * d
    active = {level_j, level_k}
    j, k = level_j, level_k

    M_c = np.zeros((dim2, dim2), dtype=complex)
    M_p1 = np.zeros((dim2, dim2), dtype=complex)
    M_p0 = np.zeros((dim2, dim2), dtype=complex)
    M_s_minus = np.zeros((dim2, dim2), dtype=complex)
    M_s_plus = np.zeros((dim2, dim2), dtype=complex)
    M_s = np.zeros((dim2, dim2), dtype=complex)

    for a in range(d):
        for b in range(d):
            idx = a * d + b
            a_in = a in active
            b_in = b in active
            if a_in and b_in:
                M_c[idx, idx] = 1.0
            elif a_in or b_in:
                M_p1[idx, idx] = 1.0
            else:
                M_p0[idx, idx] = 1.0

    # |jj> <-> |kk> coupling
    idx_jj = j * d + j
    idx_kk = k * d + k
    M_s_minus[idx_jj, idx_kk] = 1.0
    M_s_plus[idx_kk, idx_jj] = 1.0

    # |jk> <-> |kj> coupling
    idx_jk = j * d + k
    idx_kj = k * d + j
    M_s[idx_jk, idx_kj] = 1.0
    M_s[idx_kj, idx_jk] = 1.0

    return M_c, M_p1, M_p0, M_s_minus, M_s_plus, M_s


def MS_gate(phi: float, theta: float, level_j: int, level_k: int, d: int = 3):
    """
    Molmer-Sorensen gate U_MS(phi, theta; j, k) on the (level_j, level_k)
    transition: the physical XX/YY entangler of trapped-ion qudit processors.

        U_MS = exp[-i (theta/4) S_phi^2],
        S_phi = sigma_phi (x) I + I (x) sigma_phi,
        sigma_phi = cos(phi) X_jk + sin(phi) Y_jk

    [Ringbauer et al., Nat. Phys. 18, 1053 (2022), Eq. (2)]. The matrix is
    written in closed form rather than as a matrix exponential so that
    autograd can differentiate it. Using S_phi^2 = I_jk(x)I + I(x)I_jk
    + 2 sigma_phi(x)sigma_phi on the addressed block, the coefficients are

        c  = e^{-i theta/2} cos(theta/2)   diagonal of the both-active block
        s  = -i e^{-i theta/2} sin(theta/2) off-diagonal coupling
        p1 = e^{-i theta/4}                 phase of the one-active states

    and sigma_phi (x) sigma_phi has entries e^{-2i phi} on |jj><kk|,
    e^{+2i phi} on |kk><jj| and 1 on |jk><kj|, |kj><jk| (see
    _build_ms_masks for the basis classification). States with neither
    level in {j,k} are left unchanged.

    Paper check: matches Sec. II A 2 (collective generator S_phi, cos/sin
    mixing within (|jj>,|kk>) and (|jk>,|kj>), relative phase e^{+-2i phi}
    on the former, phase e^{-i theta/4} on one-active states, identity on
    the rest). Verified numerically against scipy.linalg.expm of the
    generator to < 1e-12 for d in {2,3,4} and all level pairs. At d = 2 it
    is the standard qubit MS gate up to the global phase e^{-i theta/2}.

    Args:
        phi: MS phase (radians), sets the rotation axis of sigma_phi.
        theta: MS angle (radians); theta = pi/2 is the maximally entangling
            gate on the addressed block.
        level_j, level_k: the addressed levels (0-indexed, j != k).
        d: qudit dimension.

    Returns:
        (d^2, d^2) complex unitary in the |a,b> = a*d + b basis;
        differentiable in phi and theta.
    """
    c = qml.math.exp(-1j * theta / 2) * qml.math.cos(theta / 2) + 0j
    s = -1j * qml.math.exp(-1j * theta / 2) * qml.math.sin(theta / 2) + 0j
    p1 = qml.math.exp(-1j * theta / 4) + 0j

    s_minus = s * qml.math.exp(-1j * 2 * phi)
    s_plus = s * qml.math.exp(1j * 2 * phi)

    M_c, M_p1, M_p0, M_s_minus, M_s_plus, M_s = _build_ms_masks(level_j, level_k, d)

    return (c * M_c + p1 * M_p1 + M_p0
            + s_minus * M_s_minus + s_plus * M_s_plus + s * M_s)


def CSUM_gate(d: int = 3):
    """
    Controlled-SUM gate: |a,b> -> |a, (a+b) mod d> (control a, target b).

    Parameter-free qudit Clifford gate generalizing the qubit CNOT (to
    which it reduces at d = 2); it entangles all d levels of the target.
    One of the entanglers compared in the paper's entangling-gate study
    (Sec. "Entangling-gate choice"); also used by the stabilizer benchmark
    encoder (src/stabilizer_encoder.py).

    Paper check: identical to CSUM of Sec. II A 2 (paper Eq. (6)).

    Args:
        d: qudit dimension (>= 2).

    Returns:
        (d^2, d^2) real permutation matrix in the |a,b> = a*d + b basis.
    """
    dim2 = d * d
    U = np.zeros((dim2, dim2), dtype=complex)
    for a in range(d):
        for b in range(d):
            idx_in = a * d + b
            idx_out = a * d + ((a + b) % d)
            U[idx_out, idx_in] = 1.0
    return U


def CSUB_gate(d: int = 3):
    """
    Controlled-SUB gate: |a,b> -> |a, (b-a) mod d>; equal to CSUM^dagger.

    Library extra used to uncompute CSUM (src/stabilizer_encoder.py and
    its tests). Not part of the paper's gate library and never used by the
    variational ansatz. Ported from U. Holzer's qutrit prototype.

    Args:
        d: qudit dimension (>= 2).

    Returns:
        (d^2, d^2) real permutation matrix in the |a,b> = a*d + b basis.
    """
    dim2 = d * d
    U = np.zeros((dim2, dim2), dtype=complex)
    for a in range(d):
        for b in range(d):
            idx_in = a * d + b
            idx_out = a * d + ((b - a) % d)
            U[idx_out, idx_in] = 1.0
    return U


def CEX_gate(d: int = 3, c: int = 1, t1: int = 0, t2: int = 1):
    """
    Controlled-EXchange gate CEX(c; t1, t2): an embedded-qubit entangler.

    Parameter-free two-qudit permutation. When the control qudit is in
    level |c>, the two target levels |t1> <-> |t2> are swapped; every other
    basis state is left unchanged:

        CEX(c; j, k) = |c><c| (x) S_jk + (I - |c><c|) (x) I,
        S_jk = |j><k| + |k><j| + P_rest          (here j = t1, k = t2)

    It is an involution (a single basis transposition, CEX = CEX^dag =
    CEX^-1). Unlike CSUM, which entangles all d levels of the target, CEX
    acts only inside the two-level target subspace {|t1>, |t2>} and so
    produces an embedded-qubit Bell state |c,t1> + |c,t2>-type
    entanglement. At d=2, c=1, t1=0, t2=1 it is exactly the qubit CNOT.
    For d=3, c=1, t1=0, t2=1 it swaps flat indices 3 (|1,0>) and 4 (|1,1>).

    Non-default entangler: the trained ansatz uses MS_gate; CEX enters only
    the paper's entangling-gate comparison (Sec. "Entangling-gate choice").

    Paper check: identical to CEX(c; j, k) of Sec. II A 2 (paper Eq. (7))
    with (j, k) = (t1, t2).

    Args:
        d: qudit dimension (>= 2).
        c: control level; the target swap happens only for control = |c>.
        t1, t2: the two exchanged target levels (distinct, in [0, d)).

    Returns:
        (d^2, d^2) real permutation matrix in the |a,b> = a*d + b basis
        (a = control, b = target).
    """
    assert d >= 2, f"d must be >= 2, got {d}"
    assert 0 <= c < d, f"control level c must be in [0, d), got {c}"
    assert 0 <= t1 < d and 0 <= t2 < d and t1 != t2, \
        f"target levels must be distinct and in [0, d), got t1={t1}, t2={t2}"
    dim2 = d * d
    U = np.eye(dim2, dtype=complex)
    i1 = c * d + t1
    i2 = c * d + t2
    U[i1, i1] = 0.0
    U[i2, i2] = 0.0
    U[i1, i2] = 1.0  # |c,t2> -> |c,t1>
    U[i2, i1] = 1.0  # |c,t1> -> |c,t2>
    return U


def zz_subspace_gate(theta: float, level_j: int, level_k: int, d: int = 3):
    """
    Subspace-ZZ gate U_ZZ(theta; j, k) (paper Eq. (4)).

    One-parameter diagonal interaction restricted to the (j,k) subspace:

        U_ZZ(theta) = exp(-i (theta/2) Z^(jk) (x) Z^(jk)),
        U|a,b> = exp(-i theta/2 * z(a) z(b)) |a,b>,

    where Z^(jk) has eigenvalue +1 on |j>, -1 on |k> and 0 on the spectator
    levels (z(a) in {+1, -1, 0}). A phase is applied only when BOTH qudits
    sit in {j, k}; any spectator level on either qudit gives no phase.

    This is not the physical light-shift gate of Hrmo et al. (that is
    ls_global_gate, which phases every unequal level pair); the two
    coincide only at d = 2. Non-default entangler: enters only the paper's
    entangling-gate comparison (Sec. "Entangling-gate choice"). Ported
    from U. Holzer's qutrit prototype. The deprecated alias
    `light_shift_gate` (bottom of this file) is this gate.

    Paper check: identical to U_ZZ of Sec. II A 2, paper Eq. (4).

    Args:
        theta: interaction angle (radians); identity at theta = 0.
        level_j, level_k: the addressed levels (0-indexed, j != k).
        d: qudit dimension.

    Returns:
        (d^2, d^2) complex diagonal unitary in the |a,b> = a*d + b basis;
        differentiable in theta.
    """
    dim2 = d * d
    diag = np.ones(dim2, dtype=complex)
    for a in range(d):
        for b in range(d):
            z_a = 0.0
            if a == level_j:
                z_a = 1.0
            elif a == level_k:
                z_a = -1.0
            z_b = 0.0
            if b == level_j:
                z_b = 1.0
            elif b == level_k:
                z_b = -1.0
            diag[a * d + b] = qml.math.exp(-1j * theta / 2 * z_a * z_b)
    return qml.math.diag(diag)


def ls_global_gate(theta: float, d: int = 3):
    """
    Global light-shift gate U_LS(theta) (paper Eq. (5); Hrmo et al. 2023,
    Eq. (3), arXiv:2206.04104).

    Diagonal on the full d^2-dimensional two-qudit space:

        U_LS(theta) |j,j> = |j,j>                for every j
        U_LS(theta) |j,k> = e^{i theta} |j,k>    for j != k

    i.e. the d^2 - d states with unequal level indices acquire the phase
    e^{i theta} and the d diagonal states |j,j> are unchanged. There is no
    level-pair argument: unlike zz_subspace_gate the gate acts on all
    level pairs at once and generates high-dimensional entanglement in a
    single application. The two gates coincide only at d = 2 (where this
    is a CZ-equivalent phase). Non-default entangler: enters only the
    paper's entangling-gate comparison (Sec. "Entangling-gate choice").
    The deprecated alias `light_shift_gate_hrmo` (bottom of this file) is
    this gate.

    Paper check: identical to U_LS of Sec. II A 2, paper Eq. (5).

    Args:
        theta: phase (radians) applied to every |j,k> with j != k;
            identity at theta = 0 (mod 2 pi).
        d: qudit dimension.

    Returns:
        (d^2, d^2) complex diagonal unitary in the |a,b> = a*d + b basis;
        differentiable in theta.
    """
    dim2 = d * d
    diag = np.ones(dim2, dtype=complex)
    phase = qml.math.exp(1j * theta)
    for a in range(d):
        for b in range(d):
            if a != b:
                diag[a * d + b] = phase
    return qml.math.diag(diag)


# ── Deprecated aliases (pre-rename names) ─────────────────────────────
# Kept because src/__init__.py re-exports them. Prefer the new names.
light_shift_gate = zz_subspace_gate
light_shift_gate_hrmo = ls_global_gate
