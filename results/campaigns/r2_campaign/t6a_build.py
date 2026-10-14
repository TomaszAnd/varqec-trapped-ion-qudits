#!/usr/bin/env python3
"""Explicit [[5,1,3]]_Z3 encoding circuit (App. C): construction,
MS + single-qutrit compilation, and exact verification.

Construction (from the Chau/Rains codeword form, src/catalog.py):
  |a_bar> = (1/9) * sum_{k+l+m+n+p=a} w^{kl+lm+mn+np+pk} |klmnp>
i.e. a CZ-ring graph-state-like code. Encoder (input |psi> on qudit 4):
  1. F on qudits 0..3                       (4 single-qudit Fouriers)
  2. CSUB_{i->4} for i=0..3                 (4 two-qudit Cliffords)
  3. CZ on ring edges (0,1),(1,2),(2,3),(3,4),(4,0)   (5 CZ)
Exact by construction - no phase corrections.

MS compilation (native gate set, repo MS_gate = Ringbauer Eq. 2):
  CZ = exp(i(2pi/3) N x N), N = I - Zjk(0,1) - Zjk(1,2)  (Zjk = embedded
  sigma_z on levels (j,k)); the 4 cross terms exp(i(2pi/3) Z_ab x Z_cd)
  each cost ONE MS gate conjugated by singles (Hadamard-in-subspace maps
  sigma_x->sigma_z; level permutations map transitions; local diag gates
  cancel the MS's P_jk spectator terms); local terms are single-qudit
  diagonals. => CZ (and CZ^dag) = 4 MS + singles, exact.
  CSUM_{c->t} = (I x F^dag) CZ (I x F) (verified below) => CSUB = 4 MS.
MS budget: 9 two-qudit Cliffords x 4 = 36 MS gates (vs the central code's 40).

Writes t6a_circuit.json: the full compiled gate list (singles as 3x3
matrices, MS as (q1,q2,phi,theta,levels)), verification residuals, budget.
Feeds App. C of the paper.
"""
import os, sys, json
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)
import src.jax_backend  # x64 first
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = 3
N_QUDIT = 5
DIM = D ** N_QUDIT
w = np.exp(2j * np.pi / D)

from src.gates import MS_gate, CSUM_gate, CSUB_gate
from src.catalog import load_code

F = np.array([[w ** (j * k) for k in range(D)]
              for j in range(D)]) / np.sqrt(D)
NOP = np.diag(np.arange(D)).astype(complex)
CZ = np.diag(np.exp(2j * np.pi / D * np.array(
    [(a * b) % D for a in range(D) for b in range(D)], float))).astype(complex)


def zjk(j, k):
    z = np.zeros((D, D), complex)
    z[j, j], z[k, k] = 1.0, -1.0
    return z


def hjk(j, k):
    """Hadamard on levels (j,k), identity elsewhere: maps sigma_x_jk -> sigma_z_jk."""
    h = np.eye(D, dtype=complex)
    h[j, j] = h[j, k] = h[k, j] = 1 / np.sqrt(2)
    h[k, k] = -1 / np.sqrt(2)
    return h


def perm(mapping):
    pm = np.zeros((D, D), complex)
    for src, dst in mapping.items():
        pm[dst, src] = 1.0
    return pm


def zz_from_ms(theta, ab, cd):
    """Gate list for exp(-i theta/2 Z_ab x Z_cd) as 1 MS + singles.
    Returns list of ('s', qudit_slot, U3) / ('ms', phi, theta, j, k) with
    qudit_slot 0/1 relative to the pair."""
    j, k = ab
    H = hjk(j, k)
    # map qudit-1 transition ab -> cd: W with W Z_ab W^dag = Z_cd
    a, b = ab
    c, dd = cd
    if (a, b) == (c, dd):
        W = np.eye(D, dtype=complex)
    else:
        m = {a: c, b: dd}
        rest_src = [x for x in range(D) if x not in (a, b)]
        rest_dst = [x for x in range(D) if x not in (c, dd)]
        for s_, d_ in zip(rest_src, rest_dst):
            m[s_] = d_
        W = perm(m)
    # local correction cancelling exp(-i theta/4 P_jk) on each qudit
    P = np.zeros((D, D), complex)
    P[j, j] = P[k, k] = 1.0
    Corr = np.diag(np.exp(1j * theta / 4 * np.diag(P)))
    # slot 1 = W (Corr H [MS] H) W^dag: temporal pre = H W^dag, post = W Corr H
    pre0, pre1 = H.conj().T, H.conj().T @ W.conj().T
    post0, post1 = Corr @ H, W @ Corr @ H
    # note: (V x V) MS (V^dag x V^dag) with V=H gives ZZ form; W conjugates
    # qudit-1 transition; overall: post * MS * pre on each slot.
    return [('s', 0, pre0), ('s', 1, pre1),
            ('ms', 0.0, theta, j, k),
            ('s', 0, post0), ('s', 1, post1)]


def cz_gatelist(power=1):
    """CZ^power (power=1 or 2) as 4 MS + singles (pair-relative slots)."""
    th = -4 * np.pi / 3 * power   # exp(-i th/2 ZZ) = exp(i 2pi/3 * power ZZ)
    gl = []
    for ab in ((0, 1), (1, 2)):
        for cd in ((0, 1), (1, 2)):
            gl += zz_from_ms(th, ab, cd)
    # local diagonal remainder: exp(i 2pi/3 p * (IxI - IxA - IxB - AxI - BxI))
    # with A=Z01,B=Z12: single-qudit diag on each slot + global phase.
    loc = np.diag(np.exp(-2j * np.pi / D * power *
                         np.diag(zjk(0, 1) + zjk(1, 2))))
    glob = np.exp(2j * np.pi / D * power)   # I x I term
    gl.append(('s', 0, glob * loc))
    gl.append(('s', 1, loc))
    return gl


def apply_pair_gatelist(U9_or_none, gl):
    """Compose a pair-relative gate list into a 9x9 unitary (for checks)."""
    U = np.eye(D * D, dtype=complex) if U9_or_none is None else U9_or_none
    for g in gl:
        if g[0] == 's':
            _, slot, u = g
            full = np.kron(u, np.eye(D)) if slot == 0 else np.kron(np.eye(D), u)
        else:
            _, phi, th, j, k = g
            full = np.asarray(MS_gate(phi, th, j, k, D))
        U = full @ U
    return U


def fro_phase(A, B):
    """Frobenius distance up to global phase."""
    tr = np.trace(A.conj().T @ B)
    ph = tr / abs(tr) if abs(tr) > 1e-12 else 1.0
    return float(np.linalg.norm(A * ph - B))


def main():
    checks = {}
    # ---- 1. verify CZ decomposition ----
    U = apply_pair_gatelist(None, cz_gatelist(1))
    checks['cz_4ms_fro'] = fro_phase(U, CZ)
    U2 = apply_pair_gatelist(None, cz_gatelist(2))
    checks['czdag_4ms_fro'] = fro_phase(U2, CZ.conj().T)

    # ---- 2. CSUM/CSUB from CZ ----
    csum = np.asarray(CSUM_gate(D))
    csub = np.asarray(CSUB_gate(D))
    IF = np.kron(np.eye(D), F)
    cand = IF.conj().T @ CZ @ IF
    checks['csum_from_cz_fro'] = fro_phase(cand, csum)
    cand2 = IF @ CZ @ IF.conj().T
    checks['csum_from_cz_alt_fro'] = fro_phase(cand2, csum)
    # pick whichever matches; build csub gate list = (IxFpre) CZ^s (IxFpost)
    if checks['csum_from_cz_fro'] < 1e-9:
        pre1, post1 = F, F.conj().T          # CSUM = (IxF^dag) CZ (IxF)
    else:
        pre1, post1 = F.conj().T, F
    # CSUB = CSUM^dag = (IxF^dag_pre...)^dag: conjugate CZ^dag
    csub_gl = ([('s', 1, pre1)] + cz_gatelist(2) + [('s', 1, post1)])
    Ucsub = apply_pair_gatelist(None, csub_gl)
    checks['csub_4ms_fro'] = fro_phase(Ucsub, csub)
    cz_gl = cz_gatelist(1)

    # ---- 3. full encoder gate list (absolute qudit indices) ----
    gates = []                    # ('s', q, U3) or ('ms', q1, q2, phi, th, j, k)
    for q in range(4):
        gates.append(('s', q, F))
    for q in range(4):            # CSUB_{q->4}
        for g in csub_gl:
            if g[0] == 's':
                gates.append(('s', q if g[1] == 0 else 4, g[2]))
            else:
                gates.append(('ms', q, 4, g[1], g[2], g[3], g[4]))
    for (qa, qb) in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0)):
        for g in cz_gl:
            if g[0] == 's':
                gates.append(('s', qa if g[1] == 0 else qb, g[2]))
            else:
                gates.append(('ms', qa, qb, g[1], g[2], g[3], g[4]))
    n_ms = sum(1 for g in gates if g[0] == 'ms')
    n_single = sum(1 for g in gates if g[0] == 's')

    # ---- 4. apply to |0000a> and compare with catalog codewords ----
    def apply_gates(state):
        s = state.reshape([D] * N_QUDIT)
        for g in gates:
            if g[0] == 's':
                _, q, u = g
                s = np.tensordot(u, s, axes=([1], [q]))
                s = np.moveaxis(s, 0, q)
            else:
                _, qa, qb, phi, th, j, k = g
                u9 = np.asarray(MS_gate(phi, th, j, k, D)).reshape(D, D, D, D)
                s = np.tensordot(u9, s, axes=([2, 3], [qa, qb]))
                s = np.moveaxis(s, [0, 1], [qa, qb])
        return s.reshape(DIM)

    cs = np.asarray(load_code('five_qudit_d3')['code_states'])
    outs, devs = [], []
    for a in range(D):
        inp = np.zeros(DIM, complex)
        inp[a] = 1.0        # |0,0,0,0,a>: index a*D^0 -> position: qudit4=a
        out = apply_gates(inp)
        outs.append(out)
        ov = np.vdot(cs[a], out)
        devs.append(float(np.linalg.norm(out - ov / abs(ov) * cs[a]))
                    if abs(ov) > 1e-12 else np.inf)
    checks['codeword_devs_up_to_phase'] = devs
    checks['ms_budget'] = n_ms
    checks['n_single_gates'] = n_single
    checks['two_qudit_cliffords'] = 9

    print(json.dumps(checks, indent=2, default=str))
    ok = (checks['cz_4ms_fro'] < 1e-9 and checks['csub_4ms_fro'] < 1e-9
          and max(devs) < 1e-9)
    print('T6A BUILD:', 'ALL EXACT' if ok else 'MISMATCH - investigate')

    # ---- 5. persist gate list ----
    ser = []
    for g in gates:
        if g[0] == 's':
            ser.append({'type': 's', 'q': int(g[1]),
                        'u_re': np.real(g[2]).tolist(),
                        'u_im': np.imag(g[2]).tolist()})
        else:
            ser.append({'type': 'ms', 'q1': int(g[1]), 'q2': int(g[2]),
                        'phi': float(g[3]), 'theta': float(g[4]),
                        'j': int(g[5]), 'k': int(g[6])})
    json.dump({'checks': checks, 'gates': ser,
               'description': __doc__}, open(
        os.path.join(HERE, 't6a_circuit.json'), 'w'), indent=1, default=str)
    print('saved t6a_circuit.json (%d gates, %d MS)' % (len(gates), n_ms))


if __name__ == '__main__':
    main()
