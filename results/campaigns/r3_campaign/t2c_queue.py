#!/usr/bin/env python3
"""Weighted distance trio ((7,3,d))_3, d=2,3,4:
every arm trained on the NOISE-WEIGHTED objective over ITS OWN detection set
(d=4 uses the weight-3 channel weights, so the d=4 arm no longer has to run
unweighted as in r2_campaign/t1a_queue.py).

Protocol (pre-registered):
- run_race (standard recipe): weighted corrected-Meth KL, importance@0.2,
  num_seeds=1, 3000 steps, a2a MS, n=7, K=3. Arm detection sets:
  d=2 weight-1 set; d=3 weight-2 hw closure; d=4 weight-3 hw closure
  (2661 ops), ALL with channel weights at eta=0.9296.
- L ascent {2,4,6,8}, 4 seeds {100..103}; minimal L per d = first L with
  4/4 seeds weight-(d-1)-probe-perfect (k=0); fallback = smallest L with
  >=1 perfect seed, flagged probe_selection_mandatory.
- Budget-matched variant: if the three minimal Ls differ, the arms below
  the max are ALSO reported at L_budget = max_d L_min(d) (units exist from
  the ascent when swept; otherwise trained on demand). At n=7 a2a the MS
  budget is 42L, so equal L = equal budget.
- Selection per arm at L_min: lexicographic (k_w(d-1), k_wd, seed), the
  r2_campaign/t1b_panel.py convention (d=4 uses (k_w3, k_w4, seed)).
- Evaluation of the three selected codes:
  (i) conditioned probes w=1..4 (200k/100k/100k/100k, seeds 1001..1004);
  (ii) channel-WEIGHTED iid benchmark (meth-spectator + dephasing-exact
       arms), p in {0.05, 0.08, 0.12}, shots 3e5 escalated x2 until
       >=30 errors, cap 5M;
  (iii) correlated-channel weighted-MAP LER (map_channel_ler instrument),
       eta in {0.90, 0.92, 0.93}, shots escalated x2 until >=100 errors,
       cap 640k.
Question: what does trained distance buy under each benchmark level?

Writes t2c_results.json (incremental, resume anchor). Feeds the weighted
distance trio of the paper.
Launch: nohup nice -n 19 python3 t2c_queue.py >> t2c.log 2>&1 &
"""
import os, sys, json, math, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
os.chdir(REPO)
import src.jax_backend  # x64 first
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 't2c_results.json')
STATUS = os.path.join(HERE, 't2c_status.json')
CODEDIR = os.path.join(HERE, 't2c_codes')
os.makedirs(CODEDIR, exist_ok=True)

D, N = 3, 7
DISTS = [2, 3, 4]
LS = [2, 4, 6, 8]
SEEDS = [100, 101, 102, 103]
STEPS = 3000
FRAC = 0.2
ETA = 0.9296
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]
MSBUDGET = lambda L: L * len(CONNS) * (D - 1)   # 42L
PROBE = [(1, 1001, 200_000), (2, 1002, 100_000), (3, 1003, 100_000),
         (4, 1004, 100_000)]
WB_PS = [0.05, 0.08, 0.12]
WB_START, WB_CAP, WB_MINERR = 300_000, 5_000_000, 30
MAP_ETAS = [0.90, 0.92, 0.93]
MAP_START = {0.90: 10_000, 0.92: 40_000, 0.93: 80_000}
MAP_CAP, MAP_MINERR = 640_000, 100


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


def wk_noise(w):
    from src.simulation import _apply_factored_op
    S = singles()
    def fn(state, rng):
        qs = rng.choice(N, size=w, replace=False)
        s = state
        for q in qs:
            s = _apply_factored_op(s, S[rng.integers(len(S))], int(q), N, D)
        nrm = np.linalg.norm(s)
        return s / nrm if nrm > 1e-15 else s
    return fn


def train_unit(res, dist, L, seed):
    from src.seed_race import run_race
    from src.decoders._common import encoder_forward
    from src.simulation import simulate_ler_with_correction_factored
    key = f'd{dist}_L{L}_s{seed}'
    if key in res['units']:
        return res['units'][key]
    set_status(state='running', unit=key)
    t0 = time.time()
    npz = os.path.join(CODEDIR, key + '.npz')
    r = run_race(d=D, n=N, distance=dist, layers=L, num_seeds=1,
                 steps=STEPS, target_loss=1e-9, sample_frac=FRAC,
                 sampler='importance', trigger_margin=0.25,
                 trigger_patience=5, out=npz, first_seed=seed,
                 verbose=False, weighted=True,
                 connectivity='all-to-all', entangler='ms')
    train_wall = time.time() - t0
    cs = np.asarray(encoder_forward(npz, CONNS)[0])
    row = {'dist': dist, 'L': L, 'seed': seed, 'npz': npz,
           'ms_budget': MSBUDGET(L),
           'train_loss_weighted': float(r['result']['full_loss']),
           'race_op_evs': int(r['race_op_evs']),
           'train_wall_s': round(train_wall, 1)}
    for w, sd, shots in PROBE:
        rr = simulate_ler_with_correction_factored(
            cs, wk_noise(w), singles(), N, D, n_shots=shots, seed=sd)
        k = int(round(float(rr['logical_error_rate']) * shots))
        row[f'ler_w{w}'] = {'k': k, 'n': shots, 'ler': k / shots,
                            'wilson95': wilson(k, shots)}
    row['wall_s'] = round(time.time() - t0, 1)
    res['units'][key] = row
    json.dump(res, open(OUT, 'w'), indent=2)
    print(f"[t2c] {key}: wloss={row['train_loss_weighted']:.3e} "
          f"w1={row['ler_w1']['k']} w2={row['ler_w2']['k']} "
          f"w3={row['ler_w3']['k']} w4={row['ler_w4']['k']} "
          f"[train {row['train_wall_s']}s total {row['wall_s']}s]",
          flush=True)
    return row


def probe_key(dist):
    return f'ler_w{dist - 1}'


def phase_a(res):
    for dist in DISTS:
        akey = f'd{dist}'
        if akey in res['minimal_L']:
            continue
        found = None
        for L in LS:
            rows = [train_unit(res, dist, L, s) for s in SEEDS]
            if all(r[probe_key(dist)]['k'] == 0 for r in rows):
                found = (L, False)
                break
        if found is None:
            for L in LS:
                ks = [res['units'].get(f'd{dist}_L{L}_s{s}',
                                       {}).get(probe_key(dist), {}).get('k')
                      for s in SEEDS]
                if any(k == 0 for k in ks if k is not None):
                    found = (L, True)
                    break
        if found is None:
            found = (LS[-1], True)
        Lmin, fallback = found
        # selection: (k_w(d-1), k_wd, seed)
        rows = [(res['units'][f'd{dist}_L{Lmin}_s{s}'][probe_key(dist)]['k'],
                 res['units'][f'd{dist}_L{Lmin}_s{s}'][f'ler_w{dist}']['k'],
                 s) for s in SEEDS]
        rows.sort()
        sel = rows[0][2]
        res['minimal_L'][akey] = {
            'L_min': Lmin, 'probe_selection_mandatory': fallback,
            'selected_seed': sel,
            'selected_npz': res['units'][f'd{dist}_L{Lmin}_s{sel}']['npz'],
            'ms_budget': MSBUDGET(Lmin),
            'selection': '(k_w(d-1), k_wd, seed) lexicographic'}
        json.dump(res, open(OUT, 'w'), indent=2)
        print(f"[t2c] {akey}: L_min={Lmin} fallback={fallback} sel s{sel}",
              flush=True)
    # budget-matched variant
    if 'budget_matched' not in res:
        lmax = max(res['minimal_L'][f'd{d}']['L_min'] for d in DISTS)
        bm = {'L_budget': lmax, 'arms': {}}
        for dist in DISTS:
            rows = [train_unit(res, dist, lmax, s) for s in SEEDS]
            key_rows = [(r[probe_key(dist)]['k'], r[f'ler_w{dist}']['k'],
                         r['seed']) for r in rows]
            key_rows.sort()
            sel = key_rows[0][2]
            bm['arms'][f'd{dist}'] = {
                'selected_seed': sel,
                'selected_npz': res['units'][f'd{dist}_L{lmax}_s{sel}']['npz']}
        res['budget_matched'] = bm
        json.dump(res, open(OUT, 'w'), indent=2)
        print(f"[t2c] budget-matched at L={lmax}", flush=True)


def phase_b(res):
    """Weighted-benchmark + correlated-channel evaluation of selected codes."""
    from src.decoders._common import encoder_forward
    from src.simulation import simulate_ler_with_correction_factored
    from src.errors import make_weighted_hardware_noise_fn
    from src.meth_weights import hardware_benchmark_weights
    from weighted_race_test import map_channel_ler
    W = {'meth': hardware_benchmark_weights(D, eta=ETA),
         'deph': hardware_benchmark_weights(D, eta=ETA,
                                            kind='dephasing-exact')}
    for dist in DISTS:
        akey = f'd{dist}'
        sel = res['minimal_L'][akey]
        cs = None
        for arm in ('meth', 'deph'):
            for p in WB_PS:
                key = f'wb_{akey}_{arm}_p{p}'
                if key in res['eval']:
                    continue
                set_status(state='running', unit=key)
                if cs is None:
                    cs = np.asarray(
                        encoder_forward(sel['selected_npz'], CONNS)[0])
                t0 = time.time()
                shots = WB_START
                while True:
                    rr = simulate_ler_with_correction_factored(
                        cs, make_weighted_hardware_noise_fn(D, N, p, W[arm]),
                        singles(), N, D, n_shots=shots, seed=2000)
                    k = int(round(float(rr['logical_error_rate']) * shots))
                    if k >= WB_MINERR or shots >= WB_CAP:
                        break
                    shots = min(WB_CAP, shots * 2)
                res['eval'][key] = {
                    'dist': dist, 'arm': arm, 'p': p, 'k': k, 'shots': shots,
                    'ler': k / shots, 'wilson95': wilson(k, shots),
                    'resolved': bool(k >= WB_MINERR),
                    'wall_s': round(time.time() - t0, 1)}
                json.dump(res, open(OUT, 'w'), indent=2)
                print(f"[t2c] {key}: k={k}/{shots} "
                      f"[{res['eval'][key]['wall_s']}s]", flush=True)
        for eta in MAP_ETAS:
            key = f'map_{akey}@{eta}'
            if key in res['eval']:
                continue
            set_status(state='running', unit=key)
            t0 = time.time()
            shots = MAP_START[eta]
            while True:
                ler, boot = map_channel_ler(sel['selected_npz'], D, N,
                                            eta=eta, n_shots=shots,
                                            seed=41 + int(eta * 1e4))
                k = int(round(ler * shots))
                if k >= MAP_MINERR or shots >= MAP_CAP:
                    break
                shots = min(MAP_CAP, shots * 2)
            res['eval'][key] = {
                'dist': dist, 'eta': eta, 'k': k, 'shots': shots,
                'ler': k / shots, 'wilson95': wilson(k, shots),
                'bootstrap_ci': list(boot),
                'resolved': bool(k >= MAP_MINERR),
                'wall_s': round(time.time() - t0, 1)}
            json.dump(res, open(OUT, 'w'), indent=2)
            print(f"[t2c] {key}: k={k}/{shots} LER={k/shots:.3e} "
                  f"[{res['eval'][key]['wall_s']}s]", flush=True)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {
        'protocol': __doc__, 'units': {}, 'minimal_L': {}, 'eval': {}}
    json.dump(res, open(OUT, 'w'), indent=2)
    phase_a(res)
    phase_b(res)
    set_status(state='done')
    print('[t2c] DONE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        set_status(state='FAILED', error=str(e))
        raise
