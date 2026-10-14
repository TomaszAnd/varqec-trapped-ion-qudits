#!/usr/bin/env python3
"""Entangler-study seed extension: 3 additional seeds per gate.

The earlier entangler study rests on ONE seed per gate; its main
observation (iid LER ordering INVERTS vs loss, CEX lowest) may be seed
noise. This trains seeds 1..3 per gate at the EXACT study setting —
((6,3,3))_3, L=16 a2a, 3000 steps, weighted corrected-Meth KL + importance@0.2
— by calling scripts/ansatz_study_train.train_one verbatim (protocol identity
by construction), then measures each code with the study instruments:
iid LER @p=0.05 (1M shots, seed 2000) + LER|w1 & LER|w2 (100k, seeds
1001/1002).

Codes -> results/campaigns/ansatz_study/seed_ext/{gate}_s{seed}.npz
(npz schema identical to ansatz_study_train output, decoder-compatible).
Results -> ansatz_study/seed_ext/seed_ext.json (incremental, resumable per
(gate,seed) unit and per measurement block).
Launch detached: nohup nice -n 19 python3 d_ansatz_seeds.py >> d.log 2>&1 &
"""
import os, sys, json, math, time, argparse
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
os.chdir(REPO)

import src.jax_backend  # x64 first (must precede any jnp array creation)
import numpy as np
from ansatz_study_train import train_one  # the study trainer, verbatim

D, N, L, STEPS, P = 3, 6, 16, 3000, 0.05
GATES = ['ms', 'zz', 'ls', 'csum', 'cex']
SEEDS = [1, 2, 3]  # study seed 0 already trained+measured (measure.json)
OUTDIR = os.path.join(REPO, 'results', 'campaigns', 'ansatz_study',
                      'seed_ext')
OUT = os.path.join(OUTDIR, 'seed_ext.json')
STATUS = os.path.join(OUTDIR, 'seed_ext_status.json')


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    dn = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / dn
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / dn
    return (max(0.0, c - h), c + h)


def set_status(**kw):
    kw['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    json.dump(kw, open(STATUS, 'w'), indent=2)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {
        'protocol': 'ansatz_study_train.train_one verbatim: ((6,3,3))_3 '
                    'L=16 a2a, 3000 steps, weighted Meth KL + importance@0.2, '
                    'seeds 1-3 per gate (study seed 0 in measure.json); '
                    'instruments: iid @p=0.05 1M (seed 2000), LER|w1/w2 100k '
                    '(seeds 1001/1002)', 'units': {}}
    units = [(g, s) for g in GATES for s in SEEDS]

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

    for i, (gate, seed) in enumerate(units):
        key = f'{gate}_s{seed}'
        row = res['units'].get(key, {})
        npz = os.path.join(OUTDIR, key + '.npz')
        set_status(state='running', unit=key, done=i, total=len(units))
        t0 = time.time()

        if 'train' not in row or not os.path.exists(npz):
            args = argparse.Namespace(n=N, d=D, layers=L, steps=STEPS,
                                      seed=seed, lr=0.05)
            (steps_l, loss_l, opev_l, sampled_l, tops, ppl, conns,
             theta) = train_one(gate, args)
            np.savez(npz, steps=np.array(steps_l), loss=np.array(loss_l),
                     op_evs=np.array(opev_l),
                     sampled_loss=np.array(sampled_l),
                     total_ops=tops, train_steps=STEPS,
                     params=theta, d=D, n_qudit=N, K=D,
                     n_layers=L, entangler=gate,
                     sampler='importance', sample_frac=0.2,
                     seed=seed, ppl=ppl,
                     connections=np.asarray(conns, dtype=int))
            row['train'] = {'min_full_loss': float(min(loss_l)),
                            'final_full_loss': float(loss_l[-1]),
                            'final_op_evs': int(opev_l[-1]),
                            'wall_s': round(time.time() - t0, 1)}
            row['gate'], row['seed'], row['npz'] = gate, seed, npz
            res['units'][key] = row
            json.dump(res, open(OUT, 'w'), indent=2)
            print(f"[d] {key} trained min_loss="
                  f"{row['train']['min_full_loss']:.3e} "
                  f"[{row['train']['wall_s']}s]", flush=True)

        if 'ler_iid_p05' not in row or 'ler_w2' not in row:
            data = np.load(npz, allow_pickle=True)
            conns = [list(map(int, c)) for c in data['connections']]
            cs = np.asarray(encoder_forward(npz, conns)[0])
            if 'ler_iid_p05' not in row:
                r = simulate_ler_with_correction_factored(
                    cs, make_hardware_noise_fn(D, N, P), singles, N, D,
                    n_shots=1_000_000, seed=2000)
                ler = float(r['logical_error_rate'])
                k = int(round(ler * 1_000_000))
                row['ler_iid_p05'] = {'ler': ler, 'k': k, 'n': 1_000_000,
                                      'wilson95': wilson(k, 1_000_000)}
                res['units'][key] = row
                json.dump(res, open(OUT, 'w'), indent=2)
            for w, sd in ((1, 1001), (2, 1002)):
                kk = f'ler_w{w}'
                if kk in row:
                    continue
                r = simulate_ler_with_correction_factored(
                    cs, wk_noise(w), singles, N, D, n_shots=100_000, seed=sd)
                ler = float(r['logical_error_rate'])
                k = int(round(ler * 100_000))
                row[kk] = {'ler': ler, 'k': k, 'n': 100_000,
                           'wilson95': wilson(k, 100_000)}
                res['units'][key] = row
                json.dump(res, open(OUT, 'w'), indent=2)
            row['wall_s'] = round(time.time() - t0, 1)
            res['units'][key] = row
            json.dump(res, open(OUT, 'w'), indent=2)
            print(f"[d] UNIT DONE {key}: iid={row['ler_iid_p05']['ler']:.3e} "
                  f"w1={row['ler_w1']['k']}/100k w2={row['ler_w2']['k']}/100k "
                  f"[{row['wall_s']}s]", flush=True)

    # per-gate tally across the 4 seeds (incl. study seed 0 from measure.json)
    mj = os.path.join(REPO, 'results', 'campaigns', 'ansatz_study',
                      'measure.json')
    m = json.load(open(mj)) if os.path.exists(mj) else {'gates': {}}
    tally = {}
    for g in GATES:
        lers, losses = [], []
        g0 = m['gates'].get(g, {})
        if 'iid' in g0:
            lers.append(g0['iid']['ler'])
        for s in SEEDS:
            u = res['units'].get(f'{g}_s{s}', {})
            if 'ler_iid_p05' in u:
                lers.append(u['ler_iid_p05']['ler'])
            if 'train' in u:
                losses.append(u['train']['min_full_loss'])
        tally[g] = {'iid_lers_4seed': sorted(round(x, 7) for x in lers),
                    'iid_median': float(np.median(lers)) if lers else None,
                    'ext_min_losses': sorted(round(x, 9) for x in losses)}
    res['tally'] = tally
    json.dump(res, open(OUT, 'w'), indent=2)
    set_status(state='done', done=len(units), total=len(units))
    print(f"[d] QUEUE DONE tally={json.dumps(tally)}", flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        set_status(state='FAILED', error=str(e))
        raise
