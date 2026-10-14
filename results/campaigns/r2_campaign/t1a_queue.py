#!/usr/bin/env python3
"""Distance trio: n=7 per-d mini depth sweeps at minimal budget.

Setting: n=7, K=3, q=3, a2a (|E|=21, MS budget 42L), MS entangler,
3000 steps, seeds {100..103}, run_race num_seeds=1.
Arms:
  d=2  weighted corrected-Meth KL importance@0.2, L ascent {2,4,6,8}
  d=3  weighted corrected-Meth KL importance@0.2, L ascent {2,4,8,16}
       (swept in FULL - doubles as the n=7 depth sweep)
  d=4  UNWEIGHTED hw weight-3 closure full batch (weighted machinery
       rejects w3 ops), L ascent {2,4,6,8,12,16}
Stop rule (d=2/d=4): stop at first L with (i) 4/4 seeds k=0 on the
weight-(d-1) probe AND (ii) min-seed train floor improved <1.5x vs the
previous L. Minimal viable L = smallest L with 4/4 perfect w(d-1);
fallback = smallest with >=1 perfect seed, flagged.
Per unit: train floor, cross-basis losses (d3 arm: hw + full), probes
w1@200k/w2@100k/w3@100k (+w4@100k on d4), iid 1M @p=0.05, walls.
Cost control: 50-step timing probe per (d,L) recorded first; projected
d=4 seed wall >5.5h -> that L runs seeds {100,101} only, flagged.
Writes t1a_results.json (sequential, resumable). Feeds the distance-trio
results of the paper.
Launch: nohup nice -n 19 python3 t1a_queue.py >> t1a.log 2>&1 &
"""
import os, sys, json, math, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import src.jax_backend  # x64 first (must precede any jnp array creation)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODEDIR = os.path.join(HERE, 't1a_codes')
OUT = os.path.join(HERE, 't1a_results.json')
STATUS = os.path.join(HERE, 't1a_status.json')

D, N, STEPS, P, FRAC = 3, 7, 3000, 0.05, 0.2
SEEDS = [100, 101, 102, 103]
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]
MSBUDGET = lambda L: L * len(CONNS) * (D - 1)   # 42L
ARMS = {
    2: {'L': [2, 4, 6, 8],        'weighted': True,  'stop_rule': True},
    3: {'L': [2, 4, 8, 16],       'weighted': True,  'stop_rule': False},
    4: {'L': [2, 4, 6, 8, 12, 16], 'weighted': False, 'stop_rule': True},
}
D4_SEED_WALL_CAP_S = 5.5 * 3600


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


_SINGLES = None
def singles():
    global _SINGLES
    if _SINGLES is None:
        from src.errors import qudit_hardware_error_basis
        _SINGLES = [np.asarray(E, complex)
                    for E in qudit_hardware_error_basis(D)]
    return _SINGLES


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


def measure_code(cs, with_w4=False):
    from src.simulation import simulate_ler_with_correction_factored
    from src.errors import make_hardware_noise_fn
    out = {}
    t0 = time.time()
    probes = [(1, 1001, 200_000), (2, 1002, 100_000), (3, 1003, 100_000)]
    if with_w4:
        probes.append((4, 1004, 100_000))
    for w, sd, shots in probes:
        rr = simulate_ler_with_correction_factored(
            cs, wk_noise(w), singles(), N, D, n_shots=shots, seed=sd)
        ler = float(rr['logical_error_rate'])
        k = int(round(ler * shots))
        out[f'ler_w{w}'] = {'ler': ler, 'k': k, 'n': shots,
                            'wilson95': wilson(k, shots)}
    out['probe_wall_s'] = round(time.time() - t0, 1)
    t0 = time.time()
    rr = simulate_ler_with_correction_factored(
        cs, make_hardware_noise_fn(D, N, P), singles(), N, D,
        n_shots=1_000_000, seed=2000)
    ler = float(rr['logical_error_rate'])
    k = int(round(ler * 1_000_000))
    out['ler_iid_p05'] = {'ler': ler, 'k': k, 'n': 1_000_000,
                          'wilson95': wilson(k, 1_000_000)}
    out['iid_wall_s'] = round(time.time() - t0, 1)
    return out


def race_kwargs(dist, L, seed, steps, npz, weighted):
    kw = dict(d=D, n=N, distance=dist, layers=L, num_seeds=1, steps=steps,
              target_loss=1e-9, out=npz, first_seed=seed, verbose=False,
              connectivity='all-to-all', entangler='ms')
    if weighted:
        kw.update(weighted=True, sample_frac=FRAC, sampler='importance',
                  trigger_margin=0.25, trigger_patience=5)
    else:
        kw.update(noise='hw')
    return kw


def timing_probe(res, dist, L, weighted):
    """50-step run_race wall (compile included) -> conservative projection."""
    key = f'timing_d{dist}_L{L}'
    if key in res['timing']:
        return res['timing'][key]
    from src.seed_race import run_race
    set_status(state='running', unit=key)
    npz = os.path.join(CODEDIR, f'probe_d{dist}_L{L}.npz')
    t0 = time.time()
    run_race(**race_kwargs(dist, L, 999, 50, npz, weighted))
    wall50 = time.time() - t0
    row = {'wall50_s': round(wall50, 1),
           'projected_seed_wall_s': round(wall50 * STEPS / 50, 1)}
    res['timing'][key] = row
    json.dump(res, open(OUT, 'w'), indent=2)
    print(f'[t1a] {key}: 50 steps in {row["wall50_s"]}s -> projected '
          f'{row["projected_seed_wall_s"] / 3600:.2f} h/seed', flush=True)
    return row


def xbasis_loss_fns(cache, dist, L, which):
    """which: list of noise tags ('hw','full') -> {tag: loss_fn}."""
    from src.seed_race import build_race
    key = (dist, L)
    if key not in cache:
        fns = {}
        for tag in which:
            _, fn, *_ = build_race(D, N, dist, L, noise=tag)
            fns[tag] = fn
        cache[key] = fns
    return cache[key]


def run_arm(res, dist):
    import jax.numpy as jnp
    from src.seed_race import run_race
    from src.decoders._common import encoder_forward
    arm = ARMS[dist]
    wprobe = dist - 1                 # the certifying probe weight
    xcache = {}
    prev_floor = None
    arm_key = f'arm_d{dist}'
    if res.get(arm_key, {}).get('complete'):
        return
    for L in arm['L']:
        tp = timing_probe(res, dist, L, arm['weighted'])
        seeds = SEEDS
        if dist == 4 and tp['projected_seed_wall_s'] > D4_SEED_WALL_CAP_S:
            seeds = SEEDS[:2]
            print(f'[t1a] d4 L{L}: projected seed wall '
                  f'{tp["projected_seed_wall_s"]/3600:.2f} h > 5.5 h -> '
                  f'2 seeds only', flush=True)
        for seed in seeds:
            key = f'd{dist}_L{L}_s{seed}'
            if key in res['units']:
                continue
            set_status(state='running', unit=key, done=len(res['units']))
            t0 = time.time()
            npz = os.path.join(CODEDIR, key + '.npz')
            r = run_race(**race_kwargs(dist, L, seed, STEPS, npz,
                                       arm['weighted']))
            train_wall = time.time() - t0
            row = {'d': dist, 'L': L, 'seed': seed, 'npz': npz,
                   'weighted': arm['weighted'],
                   'ms_budget': MSBUDGET(L),
                   'train_loss': float(r['result']['full_loss']),
                   'race_op_evs': int(r['race_op_evs']),
                   'n_seeds_at_L': len(seeds),
                   'train_wall_s': round(train_wall, 1)}
            theta = jnp.asarray(r['result']['theta'])
            if arm['weighted']:
                tags = ['hw', 'full'] if dist == 3 else ['hw']
                fns = xbasis_loss_fns(xcache, dist, L, tags)
                for tag in tags:
                    row[f'{tag}_basis_loss'] = float(fns[tag](theta))
            cs = np.asarray(encoder_forward(npz, CONNS)[0])
            row.update(measure_code(cs, with_w4=(dist == 4)))
            row['wall_s'] = round(time.time() - t0, 1)
            res['units'][key] = row
            json.dump(res, open(OUT, 'w'), indent=2)
            print(f"[t1a] {key}: loss={row['train_loss']:.3e} "
                  f"w{wprobe}={row[f'ler_w{wprobe}']['k']} "
                  f"iid={row['ler_iid_p05']['ler']:.3e} "
                  f"[train {row['train_wall_s']}s]", flush=True)
        # arm bookkeeping after each L
        ks = [res['units'][f'd{dist}_L{L}_s{s}'][f'ler_w{wprobe}']['k']
              for s in seeds]
        floor = min(res['units'][f'd{dist}_L{L}_s{s}']['train_loss']
                    for s in seeds)
        probes_pass = all(k == 0 for k in ks)
        floor_stalled = (prev_floor is not None
                         and prev_floor / max(floor, 1e-300) < 1.5)
        print(f'[t1a] d{dist} L{L}: floor={floor:.3e} '
              f'k_w{wprobe}={ks} pass={probes_pass} '
              f'stalled={floor_stalled}', flush=True)
        prev_floor = floor
        if arm['stop_rule'] and probes_pass and floor_stalled:
            break
    # minimal viable L per pre-registration
    swept = sorted({res['units'][k]['L'] for k in res['units']
                    if res['units'][k]['d'] == dist})
    mvl, fallback = None, False
    for L in swept:
        ks = [u[f'ler_w{wprobe}']['k'] for u in res['units'].values()
              if u['d'] == dist and u['L'] == L]
        if len(ks) >= 2 and all(k == 0 for k in ks):
            mvl = L
            break
    if mvl is None:
        for L in swept:
            ks = [u[f'ler_w{wprobe}']['k'] for u in res['units'].values()
                  if u['d'] == dist and u['L'] == L]
            if any(k == 0 for k in ks):
                mvl, fallback = L, True
                break
    res[arm_key] = {'complete': True, 'swept_L': swept,
                    'minimal_viable_L': mvl,
                    'probe_selection_mandatory': fallback,
                    'probe_weight': wprobe}
    json.dump(res, open(OUT, 'w'), indent=2)
    print(f'[t1a] ARM d={dist} done: minimal viable L = {mvl} '
          f'(fallback={fallback})', flush=True)


def main():
    os.makedirs(CODEDIR, exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {
        'protocol': 'Distance trio: n=7 per-d mini depth '
                    'sweeps, arms d=2/3/4, pre-registered stop + minimal-'
                    'viable-L rules.',
        'units': {}, 'timing': {}}
    for dist in (2, 3, 4):
        run_arm(res, dist)
    set_status(state='done', done=len(res['units']))
    print('[t1a] QUEUE DONE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        set_status(state='FAILED', error=str(e))
        raise
