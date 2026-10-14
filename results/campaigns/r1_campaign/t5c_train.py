#!/usr/bin/env python3
"""((7,3,4))_3 2-seed train + probes (GO per t5c_timing.json).

Arm: unweighted hw-closure basis extended to weight-3 products (2661 ops),
full batch (the weighted machinery rejects weight-3 ops — recorded), n=7,
L=16 a2a MS, 3000 steps, seeds {100,101}. Per code: loss; conditioned
LER|w for w=1,2,3 (100k each; d=4 certifies w1 correction + w3 detection);
iid LER @p=0.05 (1M, seed 2000). Writes t5c_train.json (resumable).
"""
import os, sys, json, math, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import src.jax_backend  # x64 first
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 't5c_train.json')
D, N, L, STEPS, P = 3, 7, 16, 3000, 0.05
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    dn = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / dn
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / dn
    return (max(0.0, c - h), c + h)


def main():
    from src.seed_race import run_race
    from src.errors import qudit_hardware_error_basis, make_hardware_noise_fn
    from src.simulation import (simulate_ler_with_correction_factored,
                                _apply_factored_op)
    from src.decoders._common import encoder_forward
    singles = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]

    def wk_noise(w):
        def fn(state, rng):
            qs = rng.choice(N, size=w, replace=False)
            s = state
            for q in qs:
                s = _apply_factored_op(s, singles[rng.integers(len(singles))],
                                       int(q), N, D)
            nrm = np.linalg.norm(s)
            return s / nrm if nrm > 1e-15 else s
        return fn

    res = json.load(open(OUT)) if os.path.exists(OUT) else {'units': {}}
    for seed in (100, 101):
        key = f'd4_n7_s{seed}'
        if key in res['units']:
            continue
        t0 = time.time()
        npz = os.path.join(HERE, 't1_codes', f't5c_{key}.npz')
        r = run_race(d=D, n=N, distance=4, layers=L, num_seeds=1,
                     steps=STEPS, target_loss=1e-9, noise='hw', out=npz,
                     first_seed=seed, verbose=False,
                     connectivity='all-to-all', entangler='ms')
        row = {'seed': seed, 'npz': npz,
               'hw_d4_loss': float(r['result']['full_loss']),
               'ms_budget': L * len(CONNS) * (D - 1),
               'train_wall_s': round(time.time() - t0, 1)}
        cs = np.asarray(encoder_forward(npz, CONNS)[0])
        for w in (1, 2, 3):
            rr = simulate_ler_with_correction_factored(
                cs, wk_noise(w), singles, N, D, n_shots=100_000,
                seed=1000 + w)
            ler = float(rr['logical_error_rate'])
            k = int(round(ler * 100_000))
            row[f'ler_w{w}'] = {'ler': ler, 'k': k, 'n': 100_000,
                                'wilson95': wilson(k, 100_000)}
        rr = simulate_ler_with_correction_factored(
            cs, make_hardware_noise_fn(D, N, P), singles, N, D,
            n_shots=1_000_000, seed=2000)
        ler = float(rr['logical_error_rate'])
        k = int(round(ler * 1_000_000))
        row['ler_iid_p05'] = {'ler': ler, 'k': k, 'n': 1_000_000,
                              'wilson95': wilson(k, 1_000_000)}
        row['wall_s'] = round(time.time() - t0, 1)
        res['units'][key] = row
        json.dump(res, open(OUT, 'w'), indent=2)
        print(f"[t5c-train] {key}: loss={row['hw_d4_loss']:.3e} "
              f"w1={row['ler_w1']['k']} w2={row['ler_w2']['k']} "
              f"w3={row['ler_w3']['k']} iid={row['ler_iid_p05']['ler']:.3e}"
              f" [{row['wall_s']}s]", flush=True)
    print('[t5c-train] DONE', flush=True)


if __name__ == '__main__':
    main()
