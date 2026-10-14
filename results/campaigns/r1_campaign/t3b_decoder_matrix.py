#!/usr/bin/env python3
"""The decoder x metric matrix that completes the Petz comparison.

t3_petz.py showed the Petz mean-infidelity ordering INVERTS the lookup
LER ordering, but confounded decoder (Petz vs lookup) with metric (mean
infidelity vs binary LER). This measures the full 2x2 on the SAME 200k-shot
i.i.d. channel stream (p=0.05, seed 4000) per code:

  (lookup, LER)       (lookup, mean infidelity)
  (Petz,   LER)       (Petz,   mean infidelity)

Petz recovery rebuilt exactly as in t3_petz.py (weight<=2 truncated
renormalized channel). Writes t3b_results.json (resumable per code).
"""
import os, sys, json, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import src.jax_backend  # x64 first
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 't3b_results.json')
D, N, P, SHOTS, SEED = 3, 5, 0.05, 200_000, 4000

sys.path.insert(0, HERE)
from t3_petz import CODES, get_states   # same code list + loader


def wilson(k, n, z=1.96):
    import math
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    dn = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / dn
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / dn
    return (max(0.0, c - h), c + h)


def main():
    from src.errors import (ErrorModel, make_hardware_noise_fn,
                            qudit_hardware_error_basis)
    from src.decoders.priors import uniform_depolarizing_prior
    from src.decoders.petz import build_petz_recovery, fidelity_under_petz
    from src.simulation import simulate_ler_with_correction_factored

    res = json.load(open(OUT)) if os.path.exists(OUT) else {'codes': {}}
    hw_grouped = ErrorModel(d=D, n_qudit=N, distance=3, closed=True,
                            basis=None).build_grouped(verbose=False)
    w_raw = uniform_depolarizing_prior(hw_grouped, N, D, P, n_single=4)
    w = w_raw / w_raw.sum()
    singles = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]

    codes = list(CODES)
    t1p = os.path.join(HERE, 't1_results.json')
    if os.path.exists(t1p):
        t1 = json.load(open(t1p))
        if 'headline' in t1:
            codes.append(('t1_headline', t1['headline']['selected_npz'],
                          None, 'T1'))

    for name, spec, *_ in codes:
        if name in res['codes']:
            continue
        t0 = time.time()
        cs = get_states(spec)
        K = cs.shape[0]
        # lookup arm (batched instrument)
        noise_fn = make_hardware_noise_fn(D, N, P)
        rr = simulate_ler_with_correction_factored(
            cs, noise_fn, singles, N, D, n_shots=SHOTS, seed=SEED)
        lk_ler = float(rr['logical_error_rate'])
        lk_k = int(round(lk_ler * SHOTS))
        # Petz arm — same channel, same seed stream convention as the
        # instrument (fresh rng, identical seed => same shot statistics
        # class, not the identical per-shot stream as the batched
        # instrument; both arms use 200k shots)
        petz = build_petz_recovery(cs, hw_grouped, w, D, N)
        rng = np.random.default_rng(SEED)
        fids = np.empty(SHOTS)
        for t in range(SHOTS):
            a = rng.normal(size=K) + 1j * rng.normal(size=K)
            a /= np.linalg.norm(a)
            noisy = noise_fn(a @ cs, rng)
            fids[t] = fidelity_under_petz(petz, cs, a, noisy)
        pz_k = int((fids < 0.5).sum())
        row = {'shots': SHOTS,
               'lookup_ler': lk_ler, 'lookup_k': lk_k,
               'lookup_ler_wilson95': wilson(lk_k, SHOTS),
               'lookup_mean_infid': 1.0 - float(rr['mean_fidelity']),
               'petz_ler': pz_k / SHOTS, 'petz_k': pz_k,
               'petz_ler_wilson95': wilson(pz_k, SHOTS),
               'petz_mean_infid': 1.0 - float(fids.mean()),
               'wall_s': round(time.time() - t0, 1)}
        res['codes'][name] = row
        json.dump(res, open(OUT, 'w'), indent=2)
        print(f"[t3b] {name}: lookup LER={lk_ler:.3e} infid="
              f"{row['lookup_mean_infid']:.3e} | petz LER={pz_k/SHOTS:.3e} "
              f"infid={row['petz_mean_infid']:.3e} [{row['wall_s']}s]",
              flush=True)
    print('[t3b] DONE', flush=True)


if __name__ == '__main__':
    main()
