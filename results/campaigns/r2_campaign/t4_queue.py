#!/usr/bin/env python3
"""Register-size family: matched-depth correlated-channel family.

Phase A (locate + train): for n in {6,8,9}: 2-seed {100,101} mini-sweep,
L ascending {2,4,6,8,12,16}, weighted corrected-Meth importance@0.2 recipe,
3000 steps; L_min(n) = first L with 2/2 seeds w2-probe k=0/100k
(pre-registered probe-only rule). Then 2 more seeds {102,103} at
L_min(n). n=7 REUSES the t1a d=3 arm (4 seeds, weighted, same recipe):
L_min(7) = that arm's minimal viable L. A 50-step timing probe per (n,L)
is recorded before each batch (measure, don't assume — n=9 at low L).

Phase B (correlated family): weighted-MAP LER under the corrected
correlated channel, eta in {0.8642, 0.90, 0.92, 0.9296, 0.93}, shots
escalated x2 until >=100 errors else per-n cap (the r1_campaign/t2t4
instrument, map_channel_ler). Code per n = best-by-probe (k_w2, k_w3,
seed) at L_min(n). n=5 central-code + n=5 reference rows REUSED from
r1_campaign/t2t4_results.json (same instrument + seeds). NOTE: the channel
dose is ONE Kraus step per EDGE (build_gate_pairs), independent of L —
matched depth equalizes TRAINING depth, not channel dose; the per-edge
dose growth with n is the physical spectator effect.

Phase C (iid n-panel re-measure): per matched-L code, iid p-sweep
{0.02,0.05,0.08,0.10,0.12,0.15}, >=3e5 shots escalated x2 to >=30 errors
(per-n caps), so the main-text n panel shares the minimal-L protocol.

Writes t4_results.json (sequential, resumable). Feeds the register-size
family results of the paper.
Launch: nohup nice -n 19 python3 t4_queue.py >> t4.log 2>&1 &
"""
import os, sys, json, math, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
os.chdir(REPO)

import src.jax_backend  # x64 first
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODEDIR = os.path.join(HERE, 't4_codes')
OUT = os.path.join(HERE, 't4_results.json')
STATUS = os.path.join(HERE, 't4_status.json')
T1A = os.path.join(HERE, 't1a_results.json')
R1T4 = os.path.join(REPO, 'results/campaigns/r1_campaign',
                    't2t4_results.json')

D, STEPS, P, FRAC = 3, 3000, 0.05, 0.2
NS = [6, 8, 9]                      # n=7 reused from t1a d3 arm
LOCATE_SEEDS = [100, 101]
EXTRA_SEEDS = [102, 103]
LSWEEP = [2, 4, 6, 8, 12, 16]
ETAS = [0.8642, 0.90, 0.92, 0.9296, 0.93]
ETA_START_SHOTS = {0.8642: 10_000, 0.90: 10_000, 0.92: 20_000,
                   0.9296: 40_000, 0.93: 40_000}
SHOT_CAP_MAP = {5: 1_280_000, 6: 640_000, 7: 320_000, 8: 160_000,
                9: 160_000}
MIN_ERR_MAP = 100
PS = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15]
IID_START, MIN_ERR_IID = 300_000, 30
IID_CAP = {6: 5_000_000, 7: 5_000_000, 8: 2_000_000, 9: 2_000_000}


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


def probes(cs, n):
    from src.simulation import simulate_ler_with_correction_factored
    out = {}
    for w, sd, shots in ((1, 1001, 200_000), (2, 1002, 100_000),
                         (3, 1003, 100_000)):
        rr = simulate_ler_with_correction_factored(
            cs, wk_noise(n, w), singles(), n, D, n_shots=shots, seed=sd)
        ler = float(rr['logical_error_rate'])
        k = int(round(ler * shots))
        out[f'ler_w{w}'] = {'ler': ler, 'k': k, 'n': shots,
                            'wilson95': wilson(k, shots)}
    return out


def train_unit(res, n, L, seed):
    from src.seed_race import run_race
    from src.decoders._common import encoder_forward
    key = f'n{n}_L{L}_s{seed}'
    if key in res['units']:
        return res['units'][key]
    set_status(state='running', unit=key)
    t0 = time.time()
    npz = os.path.join(CODEDIR, key + '.npz')
    r = run_race(d=D, n=n, distance=3, layers=L, num_seeds=1, steps=STEPS,
                 target_loss=1e-9, sample_frac=FRAC, sampler='importance',
                 trigger_margin=0.25, trigger_patience=5, out=npz,
                 first_seed=seed, verbose=False, weighted=True,
                 connectivity='all-to-all', entangler='ms')
    row = {'n': n, 'L': L, 'seed': seed, 'npz': npz,
           'ms_budget': L * len(conns_of(n)) * (D - 1),
           'train_loss_weighted': float(r['result']['full_loss']),
           'race_op_evs': int(r['race_op_evs']),
           'train_wall_s': round(time.time() - t0, 1)}
    cs = np.asarray(encoder_forward(npz, conns_of(n))[0])
    row.update(probes(cs, n))
    row['wall_s'] = round(time.time() - t0, 1)
    res['units'][key] = row
    json.dump(res, open(OUT, 'w'), indent=2)
    print(f"[t4] {key}: wloss={row['train_loss_weighted']:.3e} "
          f"w2={row['ler_w2']['k']} w3={row['ler_w3']['k']} "
          f"[{row['wall_s']}s]", flush=True)
    return row


def timing_probe(res, n, L):
    from src.seed_race import run_race
    key = f'timing_n{n}_L{L}'
    if key in res['timing']:
        return res['timing'][key]
    npz = os.path.join(CODEDIR, f'probe_n{n}_L{L}.npz')
    t0 = time.time()
    run_race(d=D, n=n, distance=3, layers=L, num_seeds=1, steps=50,
             target_loss=1e-9, sample_frac=FRAC, sampler='importance',
             trigger_margin=0.25, trigger_patience=5, out=npz,
             first_seed=999, verbose=False, weighted=True,
             connectivity='all-to-all', entangler='ms')
    wall50 = time.time() - t0
    row = {'wall50_s': round(wall50, 1),
           'projected_seed_wall_s': round(wall50 * STEPS / 50, 1)}
    res['timing'][key] = row
    json.dump(res, open(OUT, 'w'), indent=2)
    print(f'[t4] {key}: projected {row["projected_seed_wall_s"]/3600:.2f} '
          f'h/seed', flush=True)
    return row


def phase_a(res):
    for n in NS:
        akey = f'family_n{n}'
        if akey in res['family']:
            continue
        lmin = None
        for L in LSWEEP:
            timing_probe(res, n, L)
            rows = [train_unit(res, n, L, s) for s in LOCATE_SEEDS]
            if all(r['ler_w2']['k'] == 0 for r in rows):
                lmin = L
                break
        if lmin is None:
            res['family'][akey] = {'L_min': None,
                                   'note': 'no L in sweep reached 2/2 '
                                           'w2-perfect'}
            json.dump(res, open(OUT, 'w'), indent=2)
            continue
        for s in EXTRA_SEEDS:
            train_unit(res, n, lmin, s)
        sel = sorted(
            (res['units'][f'n{n}_L{lmin}_s{s}']
             for s in LOCATE_SEEDS + EXTRA_SEEDS),
            key=lambda r: (r['ler_w2']['k'], r['ler_w3']['k'], r['seed']))[0]
        res['family'][akey] = {'L_min': lmin, 'selected_seed': sel['seed'],
                               'selected_npz': sel['npz'],
                               'ms_budget': sel['ms_budget'],
                               'selection': '(k_w2, k_w3, seed) at L_min'}
        json.dump(res, open(OUT, 'w'), indent=2)
        print(f'[t4] family n={n}: L_min={lmin} seed={sel["seed"]}',
              flush=True)
    # n=7 from t1a d3 arm
    if 'family_n7' not in res['family'] and os.path.exists(T1A):
        a = json.load(open(T1A))
        if a.get('arm_d3', {}).get('complete'):
            lmin = a['arm_d3']['minimal_viable_L']
            units = [u for u in a['units'].values()
                     if u['d'] == 3 and u['L'] == lmin]
            sel = sorted(units, key=lambda r: (r['ler_w2']['k'],
                                               r['ler_w3']['k'],
                                               r['seed']))[0]
            res['family']['family_n7'] = {
                'L_min': lmin, 'selected_seed': sel['seed'],
                'selected_npz': sel['npz'], 'ms_budget': sel['ms_budget'],
                'source': 't1a_arm_d3 (4 seeds, same recipe)'}
            json.dump(res, open(OUT, 'w'), indent=2)


def phase_b(res):
    from weighted_race_test import map_channel_ler
    # reuse the r1_campaign n=5 rows (same instrument + seeds)
    if 'r1_reused' not in res:
        r1 = json.load(open(R1T4))
        res['r1_reused'] = {k: v for k, v in r1['units'].items()
                            if k.startswith('t4_t1_headline') or
                            k.startswith('t4_n5')}
        json.dump(res, open(OUT, 'w'), indent=2)
    for n in (6, 7, 8, 9):
        fam = res['family'].get(f'family_n{n}')
        if not fam or fam.get('L_min') is None:
            continue
        for eta in ETAS:
            key = f'map_n{n}@{eta}'
            if key in res['map']:
                continue
            set_status(state='running', unit=key)
            t0 = time.time()
            shots = ETA_START_SHOTS[eta]
            cap = SHOT_CAP_MAP[n]
            while True:
                ler, boot = map_channel_ler(fam['selected_npz'], D, n,
                                            eta=eta, n_shots=shots,
                                            seed=41 + int(eta * 1e4))
                k = int(round(ler * shots))
                if k >= MIN_ERR_MAP or shots >= cap:
                    break
                shots = min(cap, shots * 2)
            lo, hi = wilson(k, shots)
            res['map'][key] = {
                'n': n, 'eta': eta, 'L_min': fam['L_min'],
                'ms_budget': fam['ms_budget'], 'shots': shots, 'k': k,
                'ler': k / shots, 'wilson95': [lo, hi],
                'bootstrap_ci': list(boot),
                'resolved': bool(k >= MIN_ERR_MAP),
                'below_detection': bool(k == 0),
                'wall_s': round(time.time() - t0, 1)}
            json.dump(res, open(OUT, 'w'), indent=2)
            print(f"[t4] {key}: k={k}/{shots} LER={k/shots:.3e} "
                  f"[{res['map'][key]['wall_s']}s]", flush=True)


def phase_c(res):
    from src.simulation import simulate_ler_with_correction_factored
    from src.errors import make_hardware_noise_fn
    from src.decoders._common import encoder_forward
    for n in (6, 7, 8, 9):
        fam = res['family'].get(f'family_n{n}')
        if not fam or fam.get('L_min') is None:
            continue
        cs = np.asarray(encoder_forward(fam['selected_npz'],
                                        conns_of(n))[0])
        for pi, p in enumerate(PS):
            key = f'iid_n{n}@p{p:g}'
            if key in res['iid']:
                continue
            set_status(state='running', unit=key)
            t0 = time.time()
            shots = IID_START
            while True:
                rr = simulate_ler_with_correction_factored(
                    cs, make_hardware_noise_fn(D, n, p), singles(), n, D,
                    n_shots=shots, seed=7000 + pi)
                k = int(round(rr['logical_error_rate'] * shots))
                if k >= MIN_ERR_IID or shots >= IID_CAP[n]:
                    break
                shots = min(IID_CAP[n], shots * 2)
            res['iid'][key] = {
                'n': n, 'p': p, 'L_min': fam['L_min'],
                'ms_budget': fam['ms_budget'], 'shots': shots, 'k': k,
                'ler': k / shots, 'wilson95': wilson(k, shots),
                'resolved': bool(k >= MIN_ERR_IID),
                'below_detection': bool(k == 0),
                'wall_s': round(time.time() - t0, 1)}
            json.dump(res, open(OUT, 'w'), indent=2)
            print(f"[t4] {key}: k={k}/{shots} ler={k/shots:.3e} "
                  f"[{res['iid'][key]['wall_s']}s]", flush=True)


def main():
    os.makedirs(CODEDIR, exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {
        'protocol': 'Register-size family: matched minimal-depth '
                    'family, pre-registered L_min rule = first L with 2/2 '
                    'w2-probe-perfect; channel dose is per-edge '
                    '(L-independent) by protocol.',
        'units': {}, 'timing': {}, 'family': {}, 'map': {}, 'iid': {}}
    phase_a(res)
    phase_b(res)
    phase_c(res)
    set_status(state='done')
    print('[t4] QUEUE DONE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        set_status(state='FAILED', error=str(e))
        raise
