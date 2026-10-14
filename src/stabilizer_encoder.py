"""Native-gate encoding circuit for the [[5,1,3]]_{Z_3} benchmark code (paper App. C).

What this module implements
---------------------------
The analytical codewords of ``src.catalog.five_qudit_code_states`` (paper
Eq. 13) are graph states on a five-cycle restricted by the label constraint
sum_i k_i = a (mod q).  The standard graph-state preparation of these
codewords, adapted to the label constraint, is

    1. Fourier gates F on qudits 0..3            (4 single-qudit gates),
    2. CSUB_{i -> 4} for i = 0..3                 (4 two-qudit Cliffords),
       with qudit 4 initialised in the label state |a>,
    3. CZ on the cycle edges (0,1),(1,2),(2,3),(3,4),(4,0)   (5 two-qudit
       Cliffords),

nine two-qudit Clifford gates in all.  Each of the nine is compiled EXACTLY
into four Molmer-Sorensen (MS) gates plus single-qudit unitaries:

    CZ = exp[i (2 pi / 3) N (x) N],   N = diag(0, 1, 2) = I - Z_01 - Z_12,

where Z_jk is the embedded sigma_z on levels (j, k) (+1 on |j>, -1 on |k>,
0 elsewhere).  Expanding N (x) N gives four cross terms Z_ab (x) Z_cd plus
single-qudit diagonal terms and a global phase.  Every cross term
exp[i (2 pi / 3) Z_ab (x) Z_cd] is one MS gate conjugated by single-qudit
gates (``zz_from_ms``): a subspace Hadamard maps sigma_x -> sigma_z on the
active pair, a level permutation moves the second qudit's transition from
(a, b) to (c, d), and a diagonal correction cancels the MS phase
e^{-i theta / 4} on the single-active states.  CSUB is CZ^dag conjugated by
Fourier gates on the target, hence also 4 MS.  Total: 9 x 4 = 36 MS gates,
against the 40 MS gates of the variational ((5,3,3))_3 encoder (paper
Table III).

The single-qudit gates of the compiled list are kept as explicit 3 x 3
unitaries (Fourier, subspace Hadamard, level permutations, diagonal
corrections).  They are not further decomposed into the XY / Z native
rotations of ``src.gates``; the MS count, which is the quantity the paper
quotes, does not depend on that decomposition.

Conventions
-----------
* Basis index of an n-qudit computational state |k_0 ... k_{n-1}> is
  sum_q k_q d^{n-1-q} (qudit 0 most significant), as everywhere in ``src``.
* A gate list is a Python list of tuples, either
  ``('s', q, U)`` (single-qudit unitary U on qudit q) or
  ``('ms', q1, q2, phi, theta, j, k)`` (``src.gates.MS_gate(phi, theta, j, k)``
  on the pair (q1, q2)).  Gates are applied in list order.
* The MS gate is the physical Molmer-Sorensen gate of ``src.gates.MS_gate``
  (Ringbauer et al. Eq. 2; paper Sec. II A 2).

Consumers: ``notebooks/central_comparison.ipynb`` (paper App. C walk-through),
``tests/test_stabilizer_encoder.py``.  The original campaign script that
produced the committed ``results/campaigns/r2_campaign/t6a_circuit.json`` is
``results/campaigns/r2_campaign/t6a_build.py``; this module is its
library form and reproduces that circuit gate for gate.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.gates import MS_gate, CSUM_gate, CSUB_gate
from src.catalog import five_qudit_code_states

#: The module is written for the qutrit benchmark code of the paper.
D = 3
N_QUDIT = 5
#: Cycle edges of the five-cycle graph state (paper App. C).
CYCLE_EDGES: Tuple[Tuple[int, int], ...] = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0))


# ── elementary single- and two-qudit matrices ───────────────────────────

def fourier_gate(d: int = D) -> np.ndarray:
    """Qudit Fourier gate F[j, k] = omega^{jk} / sqrt(d), omega = e^{2 pi i / d}."""
    w = np.exp(2j * np.pi / d)
    return np.array([[w ** (j * k) for k in range(d)] for j in range(d)]) / np.sqrt(d)


def cz_gate(d: int = D) -> np.ndarray:
    """Qudit controlled-phase CZ = exp[i (2 pi / d) N (x) N], N = diag(0..d-1).

    Applies the phase omega^{ab} to the basis state |a, b>; the phase factor
    carried by each cycle edge of the codewords of paper Eq. (13).
    """
    phases = [(a * b) % d for a in range(d) for b in range(d)]
    return np.diag(np.exp(2j * np.pi / d * np.array(phases, dtype=float))).astype(complex)


def subspace_z(j: int, k: int, d: int = D) -> np.ndarray:
    """Embedded sigma_z on levels (j, k): +1 on |j>, -1 on |k>, 0 elsewhere."""
    z = np.zeros((d, d), dtype=complex)
    z[j, j], z[k, k] = 1.0, -1.0
    return z


def subspace_hadamard(j: int, k: int, d: int = D) -> np.ndarray:
    """Hadamard on levels (j, k), identity elsewhere.

    Conjugation H sigma_x H maps the embedded sigma_x_jk (the MS generator
    at phi = 0) to the embedded sigma_z_jk, so a ZZ-type interaction can be
    produced by an MS gate sandwiched between subspace Hadamards.
    """
    h = np.eye(d, dtype=complex)
    h[j, j] = h[j, k] = h[k, j] = 1 / np.sqrt(2)
    h[k, k] = -1 / np.sqrt(2)
    return h


def level_permutation(mapping: dict, d: int = D) -> np.ndarray:
    """Permutation matrix sending level ``src`` to level ``mapping[src]``."""
    pm = np.zeros((d, d), dtype=complex)
    for src, dst in mapping.items():
        pm[dst, src] = 1.0
    return pm


# ── one ZZ block from one MS gate ───────────────────────────────────────

def zz_from_ms(theta: float, ab: Tuple[int, int], cd: Tuple[int, int],
               d: int = D) -> list:
    """Gate list for exp(-i theta/2 Z_ab (x) Z_cd) as ONE MS gate plus singles.

    Pair-relative slots: slot 0 is the first qudit of the pair (levels
    (a, b)), slot 1 the second (levels (c, d)).  Returned entries are
    ``('s', slot, U)`` and ``('ms', phi, theta, j, k)``.

    Construction.  With H the subspace Hadamard on (a, b) and
    sigma_phi = X_ab at phi = 0, the MS gate exp[-i (theta/4) S_phi^2]
    conjugated by H (x) H becomes exp[-i (theta/4)(Z_ab (x) I + I (x) Z_ab)^2]
    = exp[-i (theta/2) Z_ab (x) Z_cd] times single-qudit diagonal phases
    exp[-i (theta/4) P_ab] on each qudit (P_ab the projector on the active
    levels, i.e. the e^{-i theta/4} phase the MS gate puts on single-active
    states).  Those phases are cancelled by the diagonal correction ``Corr``;
    the level permutation W then relabels the second qudit's transition from
    (a, b) to (c, d) (W Z_ab W^dag = Z_cd).
    """
    j, k = ab
    H = subspace_hadamard(j, k, d)
    a, b = ab
    c, dd = cd
    if (a, b) == (c, dd):
        W = np.eye(d, dtype=complex)
    else:
        m = {a: c, b: dd}
        rest_src = [x for x in range(d) if x not in (a, b)]
        rest_dst = [x for x in range(d) if x not in (c, dd)]
        for s_, d_ in zip(rest_src, rest_dst):
            m[s_] = d_
        W = level_permutation(m, d)
    # diagonal correction cancelling exp(-i theta/4 P_jk) on each qudit
    P = np.zeros((d, d), dtype=complex)
    P[j, j] = P[k, k] = 1.0
    Corr = np.diag(np.exp(1j * theta / 4 * np.diag(P)))
    # temporal order on slot 1: W (Corr H [MS] H) W^dag, i.e. pre = H W^dag,
    # post = W Corr H; slot 0 has W = I.
    pre0, pre1 = H.conj().T, H.conj().T @ W.conj().T
    post0, post1 = Corr @ H, W @ Corr @ H
    return [('s', 0, pre0), ('s', 1, pre1),
            ('ms', 0.0, theta, j, k),
            ('s', 0, post0), ('s', 1, post1)]


def cz_gatelist(power: int = 1, d: int = D) -> list:
    """CZ^power (power = 1 or 2 = CZ^dag at d = 3) as 4 MS gates + singles.

    N = I - Z_01 - Z_12 at d = 3, so
      N (x) N = I (x) I - (I (x) Z_01 + I (x) Z_12 + Z_01 (x) I + Z_12 (x) I)
                + sum_{ab, cd in {01, 12}} Z_ab (x) Z_cd.
    The four cross terms are the four MS blocks (``zz_from_ms`` with
    theta = -4 pi/3 * power, since exp(-i theta/2 ZZ) = exp(i 2 pi/3 power ZZ));
    the single-qudit terms are one diagonal gate per qudit, and the I (x) I
    term is a global phase (attached to slot 0's diagonal).
    """
    assert d == 3, "the (q-1)^2 = 4 block decomposition is written for q = 3"
    th = -4 * np.pi / 3 * power
    gl = []
    for ab in ((0, 1), (1, 2)):
        for cd in ((0, 1), (1, 2)):
            gl += zz_from_ms(th, ab, cd, d)
    loc = np.diag(np.exp(-2j * np.pi / d * power
                         * np.diag(subspace_z(0, 1, d) + subspace_z(1, 2, d))))
    glob = np.exp(2j * np.pi / d * power)   # the I (x) I term
    gl.append(('s', 0, glob * loc))
    gl.append(('s', 1, loc))
    return gl


def pair_gatelist_unitary(gates: list, d: int = D) -> np.ndarray:
    """Compose a pair-relative gate list into its d^2 x d^2 unitary."""
    U = np.eye(d * d, dtype=complex)
    for g in gates:
        if g[0] == 's':
            _, slot, u = g
            full = np.kron(u, np.eye(d)) if slot == 0 else np.kron(np.eye(d), u)
        else:
            _, phi, th, j, k = g
            full = np.asarray(MS_gate(phi, th, j, k, d))
        U = full @ U
    return U


def frobenius_distance_up_to_phase(A: np.ndarray, B: np.ndarray) -> float:
    """||e^{i chi} A - B||_F minimised over the global phase chi."""
    tr = np.trace(A.conj().T @ B)
    ph = tr / abs(tr) if abs(tr) > 1e-12 else 1.0
    return float(np.linalg.norm(A * ph - B))


def csub_gatelist(d: int = D) -> list:
    """CSUB |x, y> = |x, y - x mod d> as 4 MS gates + singles (pair-relative).

    CSUM = (I (x) F^dag) CZ (I (x) F) up to the Fourier convention; the
    orientation is checked numerically and CSUB = CSUM^dag is built from
    CZ^dag with the conjugate sandwich.  Raises if neither orientation
    reproduces CSUM (it cannot happen for the conventions above, but the
    check documents the derivation).
    """
    F = fourier_gate(d)
    CZ = cz_gate(d)
    csum = np.asarray(CSUM_gate(d))
    IF = np.kron(np.eye(d), F)
    if frobenius_distance_up_to_phase(IF.conj().T @ CZ @ IF, csum) < 1e-9:
        pre1, post1 = F, F.conj().T          # CSUM = (I x F^dag) CZ (I x F)
    elif frobenius_distance_up_to_phase(IF @ CZ @ IF.conj().T, csum) < 1e-9:
        pre1, post1 = F.conj().T, F
    else:  # pragma: no cover
        raise RuntimeError("CSUM is not CZ conjugated by Fourier gates in this convention")
    # CSUB = CSUM^dag: conjugate CZ^dag (= CZ^2 at d = 3) by the same sandwich.
    return [('s', 1, pre1)] + cz_gatelist(2, d) + [('s', 1, post1)]


# ── the full encoder ─────────────────────────────────────────────────────

def five_qudit_encoder_gatelist(d: int = D) -> list:
    """Compiled [[5,1,3]]_{Z_3} encoder as a flat gate list on absolute qudits.

    Input: |0, 0, 0, 0, a> (label a on qudit 4).  Output: the codeword |a_bar>
    of paper Eq. (13), up to a global phase.  Entries are
    ``('s', q, U)`` or ``('ms', q1, q2, phi, theta, j, k)``.
    """
    F = fourier_gate(d)
    csub_gl = csub_gatelist(d)
    cz_gl = cz_gatelist(1, d)
    gates = []
    for q in range(4):                                  # (i) Fourier gates
        gates.append(('s', q, F))
    for q in range(4):                                  # (ii) CSUB_{q -> 4}
        for g in csub_gl:
            if g[0] == 's':
                gates.append(('s', q if g[1] == 0 else 4, g[2]))
            else:
                gates.append(('ms', q, 4, g[1], g[2], g[3], g[4]))
    for (qa, qb) in CYCLE_EDGES:                        # (iii) CZ per cycle edge
        for g in cz_gl:
            if g[0] == 's':
                gates.append(('s', qa if g[1] == 0 else qb, g[2]))
            else:
                gates.append(('ms', qa, qb, g[1], g[2], g[3], g[4]))
    return gates


def count_ms(gates: list) -> int:
    """Number of MS gates in a gate list (the paper's entangling budget)."""
    return sum(1 for g in gates if g[0] == 'ms')


def count_single(gates: list) -> int:
    """Number of single-qudit gates in a gate list."""
    return sum(1 for g in gates if g[0] == 's')


def apply_gatelist(state: np.ndarray, gates: list, n_qudit: int = N_QUDIT,
                   d: int = D) -> np.ndarray:
    """Apply a gate list (absolute qudit indices) to a d^n state vector."""
    s = np.asarray(state, dtype=complex).reshape([d] * n_qudit)
    for g in gates:
        if g[0] == 's':
            _, q, u = g
            s = np.tensordot(u, s, axes=([1], [q]))
            s = np.moveaxis(s, 0, q)
        else:
            _, qa, qb, phi, th, j, k = g
            u9 = np.asarray(MS_gate(phi, th, j, k, d)).reshape(d, d, d, d)
            s = np.tensordot(u9, s, axes=([2, 3], [qa, qb]))
            s = np.moveaxis(s, [0, 1], [qa, qb])
    return s.reshape(d ** n_qudit)


def encoder_circuit_codewords(d: int = D) -> np.ndarray:
    """Run the compiled circuit on |0000a>, a = 0..d-1; returns (d, d^5)."""
    gates = five_qudit_encoder_gatelist(d)
    dim = d ** N_QUDIT
    out = np.zeros((d, dim), dtype=complex)
    for a in range(d):
        inp = np.zeros(dim, dtype=complex)
        inp[a] = 1.0          # |0,0,0,0,a>: qudit 4 is the least significant index
        out[a] = apply_gatelist(inp, gates, N_QUDIT, d)
    return out


def verify_encoder(d: int = D) -> dict:
    """Exactness checks of the compilation (paper App. C: all <= 1e-14).

    Returns a dict with the Frobenius residuals (up to global phase) of the
    CZ and CZ^dag 4-MS decompositions, of the CSUB decomposition, the
    per-codeword residuals of the full circuit against the analytical
    codewords, and the gate counts.
    """
    CZ = cz_gate(d)
    checks = {
        'cz_4ms_fro': frobenius_distance_up_to_phase(
            pair_gatelist_unitary(cz_gatelist(1, d), d), CZ),
        'czdag_4ms_fro': frobenius_distance_up_to_phase(
            pair_gatelist_unitary(cz_gatelist(2, d), d), CZ.conj().T),
        'csub_4ms_fro': frobenius_distance_up_to_phase(
            pair_gatelist_unitary(csub_gatelist(d), d), np.asarray(CSUB_gate(d))),
    }
    cs = np.asarray(five_qudit_code_states(d))
    circ = encoder_circuit_codewords(d)
    devs = []
    for a in range(d):
        ov = np.vdot(cs[a], circ[a])
        devs.append(float(np.linalg.norm(circ[a] - ov / abs(ov) * cs[a]))
                    if abs(ov) > 1e-12 else np.inf)
    gates = five_qudit_encoder_gatelist(d)
    checks['codeword_devs_up_to_phase'] = devs
    checks['ms_budget'] = count_ms(gates)
    checks['n_single_gates'] = count_single(gates)
    checks['two_qudit_cliffords'] = 9
    return checks
