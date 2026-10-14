#!/usr/bin/env python3
"""Explicit Meth channel -> discrete benchmark basis mapping (weighted
benchmark derivation).

Computes, from the corrected (Gaussian) src.correlated_noise Kraus families:

  (1) Exact Weyl-basis Pauli twirl per role (control/target/spectator) at
      the calibrated eta: a proper probability distribution (orthogonal
      basis).  The implemented channel is diagonal, so only the phase ops
      Z^a carry mass -> per-qudit error probability p_err = 1 - p(I).
  (2) The repo projection weights lambda(P) = (1/d^2) sum_n |Tr(P^dag K_n)|^2
      over the overcomplete discrete basis (I + hardware + closure), split
      into identity-shadow vs traceless-part contributions per op — this is
      the quantity src/meth_weights.py uses, and the split states exactly
      which Kraus mass each operator class represents.
  (3) Per-weight-sector error mass of the per-gate n-qudit twirled channel
      (2 active roles + n-2 spectators) at n in {5,7,9}: P(w=0,1,2,>=3) and
      the fraction of nontrivial mass captured by weight<=2 detection sets.
  (4) The n_max truncation check: p_err at n_max=2 vs converged n_max=40.
  (5) The two candidate benchmark draw distributions over the 2(d-1)
      hardware ops: 'meth-spectator' (normalized lambda, = the training
      prior, the primary weighted benchmark) and 'dephasing-exact'
      (normalized traceless-part mass lambda_perp — the
      diagonal-channel-honest alternative).
  (6) The calibration identity: every hardware op at every d has
      Tr = d-2, hence identical single-op average infidelity
      1 - F_avg = (d^2 - |Tr|^2) / (d(d+1)) ... verified numerically —
      so reweighting the draw does NOT change the injected error strength
      at fixed p; the injection probability alone calibrates the benchmark.

Writes t1a_derivation.json (+ printed summary). Feeds the weighted
benchmarks of the paper.
"""
import os, sys, json
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)
import src.jax_backend  # x64 first
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 't1a_derivation.json')

ETAS = [0.9296, 0.8642]
NMAX_CONV = 40


def weyl_ops(d):
    """Orthogonal Heisenberg-Weyl basis W_{a,b} = Z^a X^b (d^2 ops)."""
    w = np.exp(2j * np.pi / d)
    Z = np.diag(w ** np.arange(d))
    X = np.zeros((d, d), complex)
    for k in range(d):
        X[(k + 1) % d, k] = 1.0
    ops, labels = [], []
    for a in range(d):
        for b in range(d):
            ops.append(np.linalg.matrix_power(Z, a) @ np.linalg.matrix_power(X, b))
            labels.append(f'Z^{a}X^{b}')
    return labels, ops


def twirl_probs(kraus, ops, d):
    """p(W) = (1/d^2) sum_n |Tr(W^dag K_n)|^2 — probabilities iff ops are the
    orthogonal Weyl basis."""
    out = []
    for W in ops:
        s = 0.0
        for K in kraus:
            Km = np.diag(K) if K.ndim == 1 else K
            s += abs(np.trace(W.conj().T @ Km)) ** 2
        out.append(s / d ** 2)
    return np.array(out)


def role_kraus(d, eta, n_max):
    from src.correlated_noise import (control_qudit_kraus, target_qudit_kraus,
                                      spectator_qudit_kraus)
    return {
        'control': [np.asarray(K, complex) for K in control_qudit_kraus(d, control_level=0, eta=eta, n_max=n_max)],
        'target': [np.asarray(K, complex) for K in target_qudit_kraus(d, control_level=0, target_level=1, eta=eta, n_max=n_max)],
        'spectator': [np.asarray(K, complex) for K in spectator_qudit_kraus(d, eta=eta, n_max=n_max)],
    }


def as_mat(K):
    return np.diag(K) if K.ndim == 1 else K


def main():
    res = {'protocol': __doc__, 'etas': ETAS, 'd': {}}
    for d in (3, 4, 5):
        from src.meth_weights import build_single_qudit_pauli_basis
        pl = build_single_qudit_pauli_basis(d)
        labels = [l for l, _ in pl]
        ops = [o for _, o in pl]
        n_hw = 2 * (d - 1)
        hw_labels = labels[1:1 + n_hw]

        # (6) calibration identity: all hardware ops share |Tr| = d-2
        traces = [abs(np.trace(o)) for o in ops[1:1 + n_hw]]
        infid = [(d ** 2 - t ** 2) / (d ** 2 + d) for t in traces]
        dres = {'hw_labels': hw_labels,
                'hw_traces': traces,
                'hw_infidelity_1mFavg': infid,
                'calibration_identity_ok': bool(np.allclose(traces, d - 2)),
                'eta': {}}

        wl, wops = weyl_ops(d)
        for eta in ETAS:
            kr = role_kraus(d, eta, n_max=2)
            kr_conv = role_kraus(d, eta, n_max=NMAX_CONV)
            eres = {'roles': {}}
            for role in ('control', 'target', 'spectator'):
                p = twirl_probs(kr[role], wops, d)
                p_conv = twirl_probs(kr_conv[role], wops, d)
                # (2) repo projections over the discrete basis, with
                # identity-shadow / traceless split
                lam, lam_perp = [], []
                for P in ops:
                    Pp = P - (np.trace(P) / d) * np.eye(d)
                    s = sp = 0.0
                    for K in kr[role]:
                        Km = as_mat(K)
                        s += abs(np.trace(P.conj().T @ Km)) ** 2
                        sp += abs(np.trace(Pp.conj().T @ Km)) ** 2
                    lam.append(s / d ** 2)
                    lam_perp.append(sp / d ** 2)
                eres['roles'][role] = {
                    'weyl_labels': wl,
                    'weyl_probs_nmax2': p.tolist(),
                    'weyl_probs_conv': p_conv.tolist(),
                    'p_err_nmax2': float(1 - p[0]),
                    'p_err_conv': float(1 - p_conv[0]),
                    'nmax2_captured_frac': float((1 - p[0]) / (1 - p_conv[0])),
                    'weyl_sum_nmax2': float(p.sum()),
                    'discrete_labels': labels,
                    'lambda': lam,
                    'lambda_perp': lam_perp,
                }
            # (3) weight sectors per MS gate: 2 active + n-2 spectators
            pe = {r: eres['roles'][r]['p_err_conv'] for r in
                  ('control', 'target', 'spectator')}
            sect = {}
            for n in (5, 7, 9):
                probs = ([pe['control'], pe['target']]
                         + [pe['spectator']] * (n - 2))
                # P(weight = w) via poisson-binomial over n independent roles
                pw = np.zeros(n + 1)
                pw[0] = 1.0
                for q in probs:
                    pw = (1 - q) * pw + q * np.roll(pw, 1)
                nontriv = 1 - pw[0]
                sect[str(n)] = {
                    'P_w': pw.tolist(),
                    'P_nontrivial': float(nontriv),
                    'frac_w_le_1': float(pw[1] / nontriv),
                    'frac_w_le_2': float((pw[1] + pw[2]) / nontriv),
                    'frac_w_le_3': float((pw[1] + pw[2] + pw[3]) / nontriv),
                }
            eres['weight_sectors_per_gate'] = sect
            # (5) benchmark draw distributions over the hardware ops
            lam = np.array(eres['roles']['spectator']['lambda'][1:1 + n_hw])
            lam_perp = np.array(
                eres['roles']['spectator']['lambda_perp'][1:1 + n_hw])
            eres['draw_meth_spectator'] = (lam / lam.sum()).tolist()
            eres['draw_dephasing_exact'] = (lam_perp / lam_perp.sum()).tolist()
            dres['eta'][str(eta)] = eres
        res['d'][str(d)] = dres

    json.dump(res, open(OUT, 'w'), indent=1)

    # summary print
    for d in ('3',):
        for eta in ('0.9296', '0.8642'):
            e = res['d'][d]['eta'][eta]
            print(f'--- d={d} eta={eta} ---')
            for role in ('control', 'target', 'spectator'):
                r = e['roles'][role]
                print(f'{role:9s} p_err(conv)={r["p_err_conv"]:.5f} '
                      f'nmax2 captures {r["nmax2_captured_frac"]:.4f}')
            r = e['roles']['spectator']
            hw = res['d'][d]['hw_labels']
            lam = r['lambda'][1:1 + len(hw)]
            lp = r['lambda_perp'][1:1 + len(hw)]
            for l, a, b in zip(hw, lam, lp):
                print(f'  {l:6s} lambda={a:.5f} traceless={b:.6f} '
                      f'shadow_frac={(a - b) / a:.3f}')
            print('  draw meth-spectator:',
                  [round(x, 4) for x in e['draw_meth_spectator']])
            print('  draw dephasing-exact:',
                  [round(x, 4) for x in e['draw_dephasing_exact']])
            s = e['weight_sectors_per_gate']['5']
            print(f'  n=5 per-gate P(w>=1)={s["P_nontrivial"]:.4f} '
                  f'frac w<=2 = {s["frac_w_le_2"]:.4f}')
    print('[t1a] written', OUT)


if __name__ == '__main__':
    main()
