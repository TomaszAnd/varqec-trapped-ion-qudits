#!/usr/bin/env python3
"""Training basis -> weight-2 systematicity (residual-vs-LER diagnostics).

Question: does hardware-basis training SYSTEMATICALLY land in the good
weight-2 region (tally so far: hw-trained 3/3 perfect w2 — published,
hw-L16, hw-L24 — vs full-trained 0/5), or is the published code's
beyond-distance advantage unoptimized luck?

Protocol: 8 fresh single-seed runs per basis (seeds 100..107, disjoint from
every prior run), n=5 L=16 all-to-all XX/YY MS, 3000 steps, full-batch
uniform, target 1e-9 (never certifies; loss-best theta kept) — the
t1c_hw_retrain.py protocol with num_seeds=1. Per resulting code: LER|w=1
and LER|w=2 at 100k conditioned shots (t1b instrument, seeds 1001/1002),
i.i.d. LER @ p=0.05 (300k, seed 2000, the standard instrument), and both
basis losses.

Writes t1g_results.json (sequential, resumable: units already present are
skipped; JSON is rewritten after every completed unit) and the codes to
t1g/. Feeds the Table IV rows and the selector studies.
Launch detached: nohup nice -n 19 python3 t1g_queue.py >> t1g.log 2>&1 &

x64 footgun: import src.jax_backend before any jnp array creation.
"""
import os, sys, json, math, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import src.jax_backend  # x64 first
import numpy as np
import jax.numpy as jnp
from src.seed_race import run_race, build_race
from src.errors import qudit_hardware_error_basis, make_hardware_noise_fn
from src.simulation import (simulate_ler_with_correction_factored,
                            _apply_factored_op)
from src.decoders._common import encoder_forward

D, N, L, STEPS, P = 3, 5, 16, 3000, 0.05
SEEDS = list(range(100, 108))
HERE = os.path.dirname(os.path.abspath(__file__))
CODEDIR = os.path.join(HERE, 't1g')
OUT = os.path.join(HERE, 't1g_results.json')
STATUS = os.path.join(HERE, 't1g_status.json')
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
        'protocol': 't1c_hw_retrain protocol, num_seeds=1: n=5 L=16 a2a MS, 3000 steps, '
                    'full-batch uniform, target 1e-9; probes: LER|w1 & LER|w2 '
                    '@100k conditioned (seeds 1001/1002), iid LER p=0.05 '
                    '@300k (seed 2000)',
        'units': {}}
    units = [(b, s) for b in ('hw', 'full') for s in SEEDS]
    total = len(units)

    # cross-basis loss evaluators (built once; ~15 s)
    print('[t1g] building loss evaluators ...', flush=True)
    _, hw_loss_fn, *_ = build_race(D, N, 3, L, noise='hw')
    _, full_loss_fn, *_ = build_race(D, N, 3, L, noise='full')

    for i, (basis, seed) in enumerate(units):
        key = f'{basis}_s{seed}'
        if key in res['units']:
            print(f'[t1g] {key} already done -- skip', flush=True)
            continue
        set_status(state='running', unit=key, done=len(res['units']),
                   total=total)
        t0 = time.time()
        npz = os.path.join(CODEDIR, key + '.npz')
        r = run_race(d=D, n=N, distance=3, layers=L, num_seeds=1, steps=STEPS,
                     target_loss=1e-9, sample_frac=1.0, sampler='uniform',
                     trigger_margin=0.25, trigger_patience=5, out=npz,
                     first_seed=seed, verbose=False,
                     connectivity='all-to-all', entangler='ms', noise=basis)
        theta = jnp.asarray(r['result']['theta'])
        row = {'basis': basis, 'seed': seed, 'npz': npz,
               'train_loss': float(r['result']['full_loss']),
               'hw_basis_loss': float(hw_loss_fn(theta)),
               'full_basis_loss': float(full_loss_fn(theta)),
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
        print(f"[t1g] UNIT DONE {key}: train={row['train_loss']:.3e} "
              f"hw={row['hw_basis_loss']:.3e} full={row['full_basis_loss']:.3e} "
              f"w1={row['ler_w1']['k']}/100k w2={row['ler_w2']['k']}/100k "
              f"iid@p05={row['ler_iid_p05']['ler']:.3e} "
              f"[{row['wall_s']}s]", flush=True)

    # summary tally
    tally = {}
    for b in ('hw', 'full'):
        rows = [u for u in res['units'].values() if u['basis'] == b]
        tally[b] = {
            'n': len(rows),
            'perfect_w2': sum(1 for u in rows if u['ler_w2']['k'] == 0),
            'perfect_w1': sum(1 for u in rows if u['ler_w1']['k'] == 0),
            'w2_lers': sorted(round(u['ler_w2']['ler'], 6) for u in rows),
            'iid_p05_lers': sorted(round(u['ler_iid_p05']['ler'], 6)
                                   for u in rows)}
    res['tally'] = tally
    json.dump(res, open(OUT, 'w'), indent=2)
    set_status(state='done', done=len(res['units']), total=total)
    print(f"[t1g] QUEUE DONE tally={json.dumps(tally)}", flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        set_status(state='FAILED', error=str(e))
        raise
