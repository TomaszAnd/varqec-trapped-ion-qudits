#!/usr/bin/env python3
"""Distance-trio scrutiny: independent verification of the n=7 distance panel.

The identical d=2/3/4 i.i.d. curves at n=7 (t1b_results.json) are checked
by two independent measurements on the three best-by-probe panel codes
(t1b_results.json['panel']):

(i)  FRESH-SEED weight-conditioned probes, w in {1,2,3}, 200k conditioned
     shots each, seeds {5001,5002,5003} — independent of the t1a probe
     streams (1001/1002/1003, 100-200k), doubled shots at w2/w3.
(ii) Weighted-MAP LER under the corrected correlated channel at
     eta in {0.90, 0.92, 0.93}, shots escalated x2 until >=100 errors,
     cap 640k (2x the t4_queue.py n=7 cap, to resolve the 0.93 cells
     left at 63-64 errors) — the t4_queue.py correlated-family instrument
     (map_channel_ler), same seed convention.

Question: do the three detection-set arms separate ANYWHERE — on
re-probed conditioned weights, or on the correlated channel?

Writes a1_results.json (sequential, resumable). Feeds the distance-trio
results of the paper.
Launch: nohup nice -n 19 python3 a1_verification.py >> a1.log 2>&1 &
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
T1B = os.path.join(HERE, 't1b_results.json')

D, N = 3, 7
ETAS = [0.90, 0.92, 0.93]
ETA_START_SHOTS = {0.90: 10_000, 0.92: 40_000, 0.93: 80_000}
SHOT_CAP = 640_000
MIN_ERR_MAP = 100
PROBE_SHOTS = 200_000
PROBE_SEEDS = {1: 5001, 2: 5002, 3: 5003}


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


def conns_of(n):
    return [[i, j] for i in range(n) for j in range(i + 1, n)]


_SING = None
def singles():
    global _SING
    if _SING is None:
        from src.errors import qudit_hardware_error_basis
        _SING = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]
    return _SING


def wk_noise(n, w):
    from src.simulation import _apply_factored_op
    S = singles()
    def fn(state, rng):
        qs = rng.choice(n, size=w, replace=False)
        s = state
        for q in qs:
            s = _apply_factored_op(s, S[rng.integers(len(S))], int(q), n, D)
        nrm = np.linalg.norm(s)
        return s / nrm if nrm > 1e-15 else s
    return fn


def fresh_probes(res, arm, npz):
    from src.simulation import simulate_ler_with_correction_factored
    from src.decoders._common import encoder_forward
    cs = None
    for w in (1, 2, 3):
        key = f'probe_{arm}_w{w}'
        if key in res['probes']:
            continue
        set_status(state='running', unit=key)
        if cs is None:
            cs = np.asarray(encoder_forward(npz, conns_of(N))[0])
        t0 = time.time()
        rr = simulate_ler_with_correction_factored(
            cs, wk_noise(N, w), singles(), N, D,
            n_shots=PROBE_SHOTS, seed=PROBE_SEEDS[w])
        ler = float(rr['logical_error_rate'])
        k = int(round(ler * PROBE_SHOTS))
        res['probes'][key] = {
            'arm': arm, 'w': w, 'seed': PROBE_SEEDS[w],
            'shots': PROBE_SHOTS, 'k': k, 'ler': ler,
            'wilson95': wilson(k, PROBE_SHOTS),
            'wall_s': round(time.time() - t0, 1)}
        json.dump(res, open(OUT, 'w'), indent=2)
        print(f"[a1] {key}: k={k}/{PROBE_SHOTS} "
              f"[{res['probes'][key]['wall_s']}s]", flush=True)


def map_cells(res, arm, npz):
    from weighted_race_test import map_channel_ler
    for eta in ETAS:
        key = f'map_{arm}@{eta}'
        if key in res['map']:
            continue
        set_status(state='running', unit=key)
        t0 = time.time()
        shots = ETA_START_SHOTS[eta]
        while True:
            ler, boot = map_channel_ler(npz, D, N, eta=eta, n_shots=shots,
                                        seed=41 + int(eta * 1e4))
            k = int(round(ler * shots))
            if k >= MIN_ERR_MAP or shots >= SHOT_CAP:
                break
            shots = min(SHOT_CAP, shots * 2)
        lo, hi = wilson(k, shots)
        res['map'][key] = {
            'arm': arm, 'eta': eta, 'shots': shots, 'k': k,
            'ler': k / shots, 'wilson95': [lo, hi],
            'bootstrap_ci': list(boot),
            'resolved': bool(k >= MIN_ERR_MAP),
            'below_detection': bool(k == 0),
            'wall_s': round(time.time() - t0, 1)}
        json.dump(res, open(OUT, 'w'), indent=2)
        print(f"[a1] {key}: k={k}/{shots} LER={k/shots:.3e} "
              f"[{res['map'][key]['wall_s']}s]", flush=True)


def main():
    panel = json.load(open(T1B))['panel']
    res = json.load(open(OUT)) if os.path.exists(OUT) else {
        'protocol': 'Distance-panel verification: fresh-seed w1/w2/w3 probes '
                    '(200k each, seeds 5001-5003) + correlated-channel '
                    'weighted-MAP LER at eta {0.90,0.92,0.93}, >=100 '
                    'errors or 640k cap, for the three t1b panel codes.',
        'codes': {d: panel[d]['npz'] for d in ('d2', 'd3', 'd4')},
        'probes': {}, 'map': {}}
    json.dump(res, open(OUT, 'w'), indent=2)
    for arm in ('d2', 'd3', 'd4'):
        fresh_probes(res, arm, panel[arm]['npz'])
    for arm in ('d2', 'd3', 'd4'):
        map_cells(res, arm, panel[arm]['npz'])
    set_status(state='done')
    print('[a1] DONE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        set_status(state='FAILED', error=str(e))
        raise
