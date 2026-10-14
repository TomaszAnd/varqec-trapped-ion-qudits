#!/usr/bin/env python3
"""Gate study: i.i.d. sweeps and MAP bars that complete the minimal-L
entangler study.

Job 1 (p-sweep): per gate, the best-by-probe r3_campaign/t3 code
  (lexicographic (k_w2, k_w3, seed)), iid LER at p in
  {0.02,0.03,0.05,0.08,0.12,0.15}, shots 3e5 escalated x2 until >=30
  errors, cap 5e6. Feeds the L=2 noise-value sweep of the gate study.
Job 2 (MAP bars): per gate x seed (5x4), weighted-MAP LER under the
  corrected correlated channel at eta in {0.8642, 0.90}, 20k shots
  (weighted_race_test.map_channel_ler — the entangler-study instrument).
  Feeds the MAP bars of the L=2 gate-study panel.

Writes a1_results.json (incremental; resume anchor: completed units are
skipped on relaunch).
Launch: nohup nice -n 19 python3 a1_queue.py >> a1.log 2>&1 &
"""
import os, sys, json, math, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
os.chdir(REPO)
import src.jax_backend  # x64 first
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'a1_results.json')
STATUS = os.path.join(HERE, 'a1_status.json')
T3 = os.path.join(REPO, 'results/campaigns/r3_campaign')

D, N = 3, 6
GATES = ['ms', 'zz', 'ls', 'csum', 'cex']
SEEDS = [100, 101, 102, 103]
PGRID = [0.02, 0.03, 0.05, 0.08, 0.12, 0.15]
ETAS = [0.8642, 0.90]
START_SHOTS, CAP_SHOTS, MIN_ERR = 300_000, 5_000_000, 30
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]


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


_SING = None
def singles():
    global _SING
    if _SING is None:
        from src.errors import qudit_hardware_error_basis
        _SING = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]
    return _SING


def best_by_probe(t3res, gate):
    """Lexicographic (k_w2, k_w3, seed) over the r3_campaign/t3 units."""
    rows = [t3res['units'][f'{gate}_s{s}'] for s in SEEDS]
    rows.sort(key=lambda r: (r['ler_w2']['k'], r['ler_w3']['k'], r['seed']))
    return rows[0]


def job1_psweep(res, t3res):
    from src.decoders._common import encoder_forward
    from src.simulation import simulate_ler_with_correction_factored
    from src.errors import make_hardware_noise_fn
    for gate in GATES:
        sel = best_by_probe(t3res, gate)
        key = f'psweep_{gate}'
        unit = res['units'].setdefault(key, {
            'gate': gate, 'seed': sel['seed'], 'npz': sel['npz'],
            'selection': 'lexicographic (k_w2, k_w3, seed) over r3_campaign/t3',
            'cells': {}})
        cs = None
        for p in PGRID:
            pk = f'p{p}'
            if pk in unit['cells']:
                continue
            set_status(state='running', unit=key, p=p)
            if cs is None:
                cs = np.asarray(encoder_forward(sel['npz'], CONNS)[0])
            shots = START_SHOTS
            t0 = time.time()
            while True:
                sd = 7000 + int(p * 1000) + shots // START_SHOTS
                rr = simulate_ler_with_correction_factored(
                    cs, make_hardware_noise_fn(D, N, p), singles(), N, D,
                    n_shots=shots, seed=sd)
                k = int(round(float(rr['logical_error_rate']) * shots))
                if k >= MIN_ERR or shots >= CAP_SHOTS:
                    break
                shots = min(shots * 2, CAP_SHOTS)
            unit['cells'][pk] = {
                'p': p, 'k': k, 'n': shots, 'ler': k / shots,
                'wilson95': wilson(k, shots),
                'resolved': bool(k >= MIN_ERR),
                'wall_s': round(time.time() - t0, 1)}
            json.dump(res, open(OUT, 'w'), indent=2)
            print(f"[a1] {key} p={p}: {k}/{shots} = {k/shots:.3e} "
                  f"[{unit['cells'][pk]['wall_s']}s]", flush=True)
        # monotonicity check over resolved cells
        resolved = [(c['p'], c['ler']) for c in unit['cells'].values()
                    if c['resolved']]
        resolved.sort()
        dips = [(resolved[i][0], resolved[i + 1][0])
                for i in range(len(resolved) - 1)
                if resolved[i + 1][1] < resolved[i][1]]
        unit['monotone_in_p'] = not dips
        unit['dips'] = dips
        json.dump(res, open(OUT, 'w'), indent=2)
        print(f"[a1] {key}: monotone={unit['monotone_in_p']} dips={dips}",
              flush=True)


def job2_map(res):
    from weighted_race_test import map_channel_ler
    for gate in GATES:
        for seed in SEEDS:
            key = f'map_{gate}_s{seed}'
            if key in res['units']:
                continue
            npz = os.path.join(T3, 't3_codes', f'{gate}_s{seed}.npz')
            row = {'gate': gate, 'seed': seed, 'npz': npz, 'cells': {}}
            for eta in ETAS:
                set_status(state='running', unit=key, eta=eta)
                t0 = time.time()
                ler, boot = map_channel_ler(npz, D, N, eta=eta,
                                            n_shots=20_000,
                                            seed=31 + int(eta * 1e4))
                k = int(round(ler * 20_000))
                row['cells'][f'eta{eta}'] = {
                    'eta': eta, 'ler': ler, 'k': k, 'n': 20_000,
                    'wilson95': wilson(k, 20_000),
                    'boot': list(boot) if boot is not None else None,
                    'wall_s': round(time.time() - t0, 1)}
                print(f"[a1] {key} eta={eta}: {ler:.3e} ({k}/20k) "
                      f"[{row['cells'][f'eta{eta}']['wall_s']}s]", flush=True)
            res['units'][key] = row
            json.dump(res, open(OUT, 'w'), indent=2)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {
        'protocol': __doc__, 'units': {}}
    t3res = json.load(open(os.path.join(T3, 't3_results.json')))
    json.dump(res, open(OUT, 'w'), indent=2)
    job1_psweep(res, t3res)
    job2_map(res)
    set_status(state='done')
    print('[a1] DONE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        set_status(state='FAILED', error=str(e))
        raise
