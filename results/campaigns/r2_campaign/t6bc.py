#!/usr/bin/env python3
"""Declare-threshold insensitivity + Haar-vs-basis check.

Central cell: A_L2_s100 (the central code), iid hardware channel p=0.05,
1M shots, seed 2000 — the exact r1_campaign instrument stream.

T6b: one pass, error counts at declare thresholds tau in {0.25, 0.5, 0.75}
     (src instrument hardcodes fc < 0.5). Haar prep draws are identical to
     src/simulation.py::simulate_ler_with_correction_factored (same rng
     order, batch=1024), so the tau=0.5 count MUST equal the stored 333/1M
     (t1_results.json A_L2_s100) — stream-identity validation gate.
T6c: same instrument, initial logical state drawn uniformly over the K
     logical BASIS states instead of Haar; 1M shots (own stream, seed 2000).
     Also run for [[5,1,3]] for a paired check.
Writes t6bc_results.json.
"""
import os, sys, json, math, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import src.jax_backend  # x64 first (must precede any jnp array creation)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 't6bc_results.json')
D, N, P = 3, 5, 0.05
SHOTS, SEED = 1_000_000, 2000
TAUS = (0.25, 0.5, 0.75)
HEADLINE_NPZ = os.path.join(REPO, 'results/campaigns/r1_campaign',
                            't1_codes', 'A_L2_s100.npz')
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    dn = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / dn
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / dn
    return (max(0.0, c - h), c + h)


def ler_multi_tau(code_states, noise_fn, single_errors, n_qudit, d,
                  n_shots, seed, taus=TAUS, prep='haar', batch=1024):
    """Inline variant of simulate_ler_with_correction_factored: identical
    correction stack + decode; error counted at several thresholds in one
    pass; prep='haar' reproduces the src rng stream exactly, prep='basis'
    draws a uniform logical basis state per shot."""
    from src.simulation import _apply_factored_op
    K, dim = code_states.shape
    ces = [np.asarray(code_states)]
    for q in range(n_qudit):
        for E in single_errors:
            ces.append(np.stack([
                _apply_factored_op(code_states[k], E, q, n_qudit, d)
                for k in range(K)]))
    CES = np.stack(ces)
    n_corr = CES.shape[0]
    M = np.conj(CES.reshape(n_corr * K, dim))

    rng = np.random.default_rng(seed)
    counts = {t: 0 for t in taus}
    fsum = 0.0
    for start in range(0, n_shots, batch):
        b = min(batch, n_shots - start)
        LOGICAL = np.empty((b, dim), dtype=complex)
        NOISY = np.empty((b, dim), dtype=complex)
        for j in range(b):
            if prep == 'haar':
                alpha = rng.standard_normal(K) + 1j * rng.standard_normal(K)
                alpha /= np.linalg.norm(alpha)
            else:
                alpha = np.zeros(K, complex)
                alpha[rng.integers(K)] = 1.0
            LOGICAL[j] = alpha @ code_states
            NOISY[j] = noise_fn(LOGICAL[j], rng)
        inner = (M @ NOISY.T).reshape(n_corr, K, b)
        best = np.argmax(np.sum(np.abs(inner) ** 2, axis=1), axis=0)
        coeffs = inner[best, :, np.arange(b)]
        projected = coeffs @ code_states
        norms = np.linalg.norm(projected, axis=1)
        ok = norms > 1e-10
        projected[ok] /= norms[ok, None]
        fc = np.abs(np.einsum('jd,jd->j',
                              np.conj(LOGICAL), projected)) ** 2
        fsum += float(np.sum(fc))
        for t in taus:
            counts[t] += int(np.sum(fc < t))
    return counts, fsum / n_shots


def main():
    from src.errors import qudit_hardware_error_basis, make_hardware_noise_fn
    from src.decoders._common import encoder_forward
    from src.catalog import load_code
    singles = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]
    noise = make_hardware_noise_fn(D, N, P)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}

    codes = {
        'headline_A_L2_s100':
            np.asarray(encoder_forward(HEADLINE_NPZ, CONNS)[0]),
        '513': np.asarray(load_code('five_qudit_d3')['code_states']),
    }

    for name, cs in codes.items():
        for prep in ('haar', 'basis'):
            key = f'{name}_{prep}'
            if key in res:
                continue
            t0 = time.time()
            counts, mf = ler_multi_tau(cs, noise, singles, N, D,
                                       SHOTS, SEED, prep=prep)
            res[key] = {
                'code': name, 'prep': prep, 'p': P, 'shots': SHOTS,
                'seed': SEED, 'mean_fidelity': mf,
                'by_tau': {str(t): {'k': counts[t],
                                    'ler': counts[t] / SHOTS,
                                    'wilson95': wilson(counts[t], SHOTS)}
                           for t in TAUS},
                'wall_s': round(time.time() - t0, 1)}
            json.dump(res, open(OUT, 'w'), indent=2)
            print(f'[t6bc] {key}: ' + ' '.join(
                f'tau{t}={counts[t]}' for t in TAUS) +
                f' [{res[key]["wall_s"]}s]', flush=True)

    # validation gate: headline haar tau=0.5 must equal the stored 333/1M
    k05 = res['headline_A_L2_s100_haar']['by_tau']['0.5']['k']
    res['validation'] = {
        'expected_k_tau05_headline_haar': 333,
        'measured': k05, 'stream_identity_ok': (k05 == 333)}
    json.dump(res, open(OUT, 'w'), indent=2)
    print(f'[t6bc] validation stream-identity: k={k05} expected 333 -> '
          f'{"OK" if k05 == 333 else "MISMATCH (investigate)"}', flush=True)
    print('[t6bc] DONE', flush=True)


if __name__ == '__main__':
    main()
