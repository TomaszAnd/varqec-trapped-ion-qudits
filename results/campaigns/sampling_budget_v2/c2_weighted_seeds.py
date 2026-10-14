#!/usr/bin/env python3
"""Does the WEIGHTED importance@0.2 recipe land in the good weight-2 region,
or is full-batch training part of the training-basis effect seen in
loss_ler_diag/t1g?

8 fresh single-seed runs (seeds 100..107, same seeds as t1g), n=5 L=16 a2a
MS, 3000 steps, WEIGHTED corrected-Meth KL + importance sampling @0.2 (the
paper's best-practice line; run_race weighted=True arm, hardware closure
basis), target 1e-9 (never certifies; loss-best theta kept). Per code, the
exact t1g instruments: LER|w1 & LER|w2 @100k conditioned (seeds 1001/1002),
iid LER p=0.05 @300k (seed 2000), plus cross-basis KL losses and race op-EV
accounting.

Writes c2_results.json (sequential, resumable: units already present are
skipped; incremental JSON after every unit).
Launch detached: nohup nice -n 19 python3 c2_weighted_seeds.py >> c2.log 2>&1 &
"""
import os, sys, json, math, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import src.jax_backend  # x64 first (must precede any jnp array creation)
import numpy as np
import jax.numpy as jnp
from src.seed_race import run_race, build_race
from src.errors import qudit_hardware_error_basis, make_hardware_noise_fn
from src.simulation import (simulate_ler_with_correction_factored,
                            _apply_factored_op)
from src.decoders._common import encoder_forward

D, N, L, STEPS, P, FRAC = 3, 5, 16, 3000, 0.05, 0.2
SEEDS = list(range(100, 108))
HERE = os.path.dirname(os.path.abspath(__file__))
CODEDIR = os.path.join(HERE, 'c2')
OUT = os.path.join(HERE, 'c2_results.json')
STATUS = os.path.join(HERE, 'c2_status.json')
SINGLES = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    dn = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / dn
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / dn
    return (max(0.0, c - h), c + h)


def wk_noise(w):
    def fn(state, rng):
        qs = rng.choice(N, size=w, replace=False)
        s = state
        for q in qs:
            s = _apply_factored_op(s, SINGLES[rng.integers(len(SINGLES))],
                                   int(q), N, D)
        nrm = np.linalg.norm(s)
        return s / nrm if nrm > 1e-15 else s
    return fn


def set_status(**kw):
    kw['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    json.dump(kw, open(STATUS, 'w'), indent=2)


def main():
    os.makedirs(CODEDIR, exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {
        'protocol': 'Sec IX weighted line, num_seeds=1: n=5 L=16 a2a MS, '
                    '3000 steps, weighted corrected-Meth KL + importance@0.2, '
                    'target 1e-9; probes: LER|w1 & LER|w2 @100k conditioned '
                    '(seeds 1001/1002), iid LER p=0.05 @300k (seed 2000). '
                    'Same seeds + instruments as t1g (full-batch unweighted '
                    'hw/full arms) for direct comparison.',
        'units': {}}
    total = len(SEEDS)

    print('[c2] building cross-basis loss evaluators ...', flush=True)
    _, hw_loss_fn, *_ = build_race(D, N, 3, L, noise='hw')
    _, full_loss_fn, *_ = build_race(D, N, 3, L, noise='full')

    for i, seed in enumerate(SEEDS):
        key = f'wimp02_s{seed}'
        if key in res['units']:
            print(f'[c2] {key} already done -- skip', flush=True)
            continue
        set_status(state='running', unit=key, done=len(res['units']),
                   total=total)
        t0 = time.time()
        npz = os.path.join(CODEDIR, key + '.npz')
        r = run_race(d=D, n=N, distance=3, layers=L, num_seeds=1, steps=STEPS,
                     target_loss=1e-9, sample_frac=FRAC, sampler='importance',
                     trigger_margin=0.25, trigger_patience=5, out=npz,
                     first_seed=seed, verbose=False, weighted=True,
                     connectivity='all-to-all', entangler='ms')
        theta = jnp.asarray(r['result']['theta'])
        row = {'objective': 'weighted_importance_0.2', 'seed': seed,
               'npz': npz,
               'train_loss_weighted': float(r['result']['full_loss']),
               'hw_basis_loss': float(hw_loss_fn(theta)),
               'full_basis_loss': float(full_loss_fn(theta)),
               'race_op_evs': int(r['race_op_evs']),
               'n_trigger_attempts': int(r['n_trigger_attempts']),
               'n_false_triggers': int(r['n_false_triggers']),
               'train_wall_s': round(time.time() - t0, 1)}

        cs = np.asarray(encoder_forward(npz, CONNS)[0])
        for w, sd in ((1, 1001), (2, 1002)):
            rr = simulate_ler_with_correction_factored(
                cs, wk_noise(w), SINGLES, N, D, n_shots=100_000, seed=sd)
            ler = float(rr['logical_error_rate'])
            k = int(round(ler * 100_000))
            row[f'ler_w{w}'] = {'ler': ler, 'k': k, 'n': 100_000,
                                'wilson95': wilson(k, 100_000)}
        rr = simulate_ler_with_correction_factored(
            cs, make_hardware_noise_fn(D, N, P), SINGLES, N, D,
            n_shots=300_000, seed=2000)
        ler = float(rr['logical_error_rate'])
        k = int(round(ler * 300_000))
        row['ler_iid_p05'] = {'ler': ler, 'k': k, 'n': 300_000,
                              'wilson95': wilson(k, 300_000)}
        row['wall_s'] = round(time.time() - t0, 1)
        res['units'][key] = row
        json.dump(res, open(OUT, 'w'), indent=2)
        print(f"[c2] UNIT DONE {key}: wloss={row['train_loss_weighted']:.3e} "
              f"hw={row['hw_basis_loss']:.3e} "
              f"w1={row['ler_w1']['k']}/100k w2={row['ler_w2']['k']}/100k "
              f"iid@p05={row['ler_iid_p05']['ler']:.3e} "
              f"opEVs={row['race_op_evs']} [{row['wall_s']}s]", flush=True)

    rows = list(res['units'].values())
    res['tally'] = {
        'n': len(rows),
        'perfect_w2': sum(1 for u in rows if u['ler_w2']['k'] == 0),
        'perfect_w1': sum(1 for u in rows if u['ler_w1']['k'] == 0),
        'w2_lers': sorted(round(u['ler_w2']['ler'], 6) for u in rows),
        'iid_p05_lers': sorted(round(u['ler_iid_p05']['ler'], 6)
                               for u in rows),
        'race_op_evs': sorted(u['race_op_evs'] for u in rows)}
    json.dump(res, open(OUT, 'w'), indent=2)
    set_status(state='done', done=len(res['units']), total=total)
    print(f"[c2] QUEUE DONE tally={json.dumps(res['tally'])}", flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        set_status(state='FAILED', error=str(e))
        raise
