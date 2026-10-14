#!/usr/bin/env python3
"""Distance trio: n=7 distance-panel measurement.

Requires t1a_results.json arms complete. Per d in {2,3,4}: select the
best-by-probe code at the arm's minimal viable L — lexicographic
(k_w(d-1), k_w(d), seed); d=4 uses (k_w3, k_w4, seed). Measure the iid
LER curve p in {0.02, 0.05, 0.08, 0.10, 0.12, 0.15}, >=3e5 shots
escalated x2 until >=30 errors (cap 5M), Wilson 95%, MS budget 42L
annotated. 0-error cells reported below-detection + Wilson UB.
Writes t1b_results.json (resumable). Feeds the distance-trio results of
the paper.
"""
import os, sys, json, math, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import src.jax_backend  # x64 first
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T1A = os.path.join(HERE, 't1a_results.json')
OUT = os.path.join(HERE, 't1b_results.json')
D, N = 3, 7
PS = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15]
START_SHOTS, SHOT_CAP, MIN_ERR = 300_000, 5_000_000, 30
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    dn = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / dn
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / dn
    return (max(0.0, c - h), c + h)


def select(res_a, dist):
    arm = res_a[f'arm_d{dist}']
    L = arm['minimal_viable_L']
    if L is None:
        return None, arm
    wp = dist - 1
    units = [u for u in res_a['units'].values()
             if u['d'] == dist and u['L'] == L]
    tiebreak = (f'ler_w{dist}' if dist < 4 else 'ler_w4')
    units.sort(key=lambda u: (u[f'ler_w{wp}']['k'],
                              u.get(tiebreak, {}).get('k', 0), u['seed']))
    return units[0], arm


def main():
    from src.simulation import simulate_ler_with_correction_factored
    from src.errors import qudit_hardware_error_basis, make_hardware_noise_fn
    from src.decoders._common import encoder_forward
    singles = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]
    res_a = json.load(open(T1A))
    res = json.load(open(OUT)) if os.path.exists(OUT) else {
        'protocol': 'Distance panel n=7: best-by-probe per d, iid p-sweep escalated to '
                    '>=30 errors (cap 5M), MS budget 42L', 'panel': {}}

    for dist in (2, 3, 4):
        sel, arm = select(res_a, dist)
        if sel is None:
            res['panel'][f'd{dist}'] = {'SKIPPED': 'no viable L in arm',
                                        'arm': arm}
            json.dump(res, open(OUT, 'w'), indent=2)
            continue
        tag = f'd{dist}'
        ent = res['panel'].setdefault(tag, {
            'npz': sel['npz'], 'L': sel['L'], 'seed': sel['seed'],
            'ms_budget': sel['ms_budget'],
            'probe_selection': {k: sel[k]['k'] for k in sel
                                if k.startswith('ler_w')},
            'minimal_viable_L': arm['minimal_viable_L'],
            'probe_selection_mandatory': arm['probe_selection_mandatory'],
            'curve': {}})
        cs = np.asarray(encoder_forward(sel['npz'], CONNS)[0])
        for pi, p in enumerate(PS):
            pk = f'{p:g}'
            if pk in ent['curve']:
                continue
            t0 = time.time()
            shots = START_SHOTS
            while True:
                rr = simulate_ler_with_correction_factored(
                    cs, make_hardware_noise_fn(D, N, p), singles, N, D,
                    n_shots=shots, seed=7000 + pi)
                k = int(round(rr['logical_error_rate'] * shots))
                if k >= MIN_ERR or shots >= SHOT_CAP:
                    break
                shots = min(SHOT_CAP, shots * 2)
            ent['curve'][pk] = {
                'p': p, 'k': k, 'n': shots, 'ler': k / shots,
                'wilson95': wilson(k, shots),
                'resolved': bool(k >= MIN_ERR),
                'below_detection': bool(k == 0),
                'wall_s': round(time.time() - t0, 1)}
            json.dump(res, open(OUT, 'w'), indent=2)
            print(f"[t1b] d{dist} p={p}: k={k}/{shots} "
                  f"ler={k/shots:.3e} [{ent['curve'][pk]['wall_s']}s]",
                  flush=True)
    print('[t1b] DONE', flush=True)


if __name__ == '__main__':
    main()
