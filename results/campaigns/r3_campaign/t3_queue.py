#!/usr/bin/env python3
"""Consistency reruns at minimal depth on the weighted objective.

(A) Entangler study rerun at ((6,3,3))_3 MINIMAL L, NOISE-WEIGHTED objective,
    4 seeds/gate (the earlier entangler study was L=16 on the
    unweighted-loss protocol). Gates {ms, zz, ls, csum, cex}, run_race
    weighted importance@0.2 recipe, L=2, 3000 steps, seeds {100..103}. Per
    unit: weighted loss floor, w1@200k / w2@100k / w3@100k conditioned
    probes, iid LER @p=0.05 1M (uniform instrument, comparable with the
    earlier entangler study).
    Question: do the loss ordering ms<zz<ls<csum<cex, the universal
    weight-<=2 perfection, and the iid-inversion-is-seed-noise finding
    survive at minimal depth on the weighted objective?

(B) Sampler-study confirmation arm at L=2 (launched by the queue wrapper as
    scripts/fig8_weighted_train.py --n 9 --layers 2 --steps 1500
    --outdir r3_campaign/t3_sampler_L2): does the 5-strategy ranking /
    op-EV story hold at minimal depth?  compare_samplers() then writes the
    ranking comparison vs the committed L=16 traces into t3_results.json.

Writes t3_results.json (incremental). Feeds the weighted benchmarks of
the paper; the L=2 sampler codes are scored again by r10_sampler.
Launch: nohup nice -n 19 python3 t3_queue.py >> t3.log 2>&1 &
"""
import os, sys, json, math, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)
import src.jax_backend  # x64 first
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 't3_results.json')
STATUS = os.path.join(HERE, 't3_status.json')
CODEDIR = os.path.join(HERE, 't3_codes')
SAMPLER_DIR = os.path.join(HERE, 't3_sampler_L2')
FIG8_L16 = os.path.join(REPO, 'results/campaigns/fig8_weighted')
os.makedirs(CODEDIR, exist_ok=True)

D, N, L = 3, 6, 2
GATES = ['ms', 'zz', 'ls', 'csum', 'cex']
SEEDS = [100, 101, 102, 103]
STEPS = 3000
FRAC = 0.2
P = 0.05
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


def part_a(res):
    from src.seed_race import run_race
    from src.decoders._common import encoder_forward
    from src.simulation import simulate_ler_with_correction_factored
    from src.errors import make_hardware_noise_fn
    for gate in GATES:
        for seed in SEEDS:
            key = f'{gate}_s{seed}'
            if key in res['units']:
                continue
            set_status(state='running', unit=key)
            t0 = time.time()
            npz = os.path.join(CODEDIR, key + '.npz')
            r = run_race(d=D, n=N, distance=3, layers=L, num_seeds=1,
                         steps=STEPS, target_loss=1e-9, sample_frac=FRAC,
                         sampler='importance', trigger_margin=0.25,
                         trigger_patience=5, out=npz, first_seed=seed,
                         verbose=False, weighted=True,
                         connectivity='all-to-all', entangler=gate)
            train_wall = time.time() - t0
            cs = np.asarray(encoder_forward(npz, CONNS)[0])
            row = {'gate': gate, 'seed': seed, 'npz': npz, 'L': L,
                   'train_loss_weighted': float(r['result']['full_loss']),
                   'race_op_evs': int(r['race_op_evs']),
                   'train_wall_s': round(train_wall, 1)}
            for w, sd, shots in ((1, 1001, 200_000), (2, 1002, 100_000),
                                 (3, 1003, 100_000)):
                rr = simulate_ler_with_correction_factored(
                    cs, wk_noise(w), singles(), N, D, n_shots=shots, seed=sd)
                k = int(round(float(rr['logical_error_rate']) * shots))
                row[f'ler_w{w}'] = {'k': k, 'n': shots, 'ler': k / shots,
                                    'wilson95': wilson(k, shots)}
            rr = simulate_ler_with_correction_factored(
                cs, make_hardware_noise_fn(D, N, P), singles(), N, D,
                n_shots=1_000_000, seed=2000)
            k = int(round(float(rr['logical_error_rate']) * 1_000_000))
            row['ler_iid_p05'] = {'k': k, 'n': 1_000_000, 'ler': k / 1e6,
                                  'wilson95': wilson(k, 1_000_000)}
            row['wall_s'] = round(time.time() - t0, 1)
            res['units'][key] = row
            json.dump(res, open(OUT, 'w'), indent=2)
            print(f"[t3] {key}: wloss={row['train_loss_weighted']:.3e} "
                  f"w2={row['ler_w2']['k']}/100k "
                  f"iid={row['ler_iid_p05']['ler']:.3e} "
                  f"[{row['wall_s']}s]", flush=True)
    # per-gate summary: median loss / iid, probe tally
    summ = {}
    for gate in GATES:
        rows = [res['units'][f'{gate}_s{s}'] for s in SEEDS]
        summ[gate] = {
            'loss_median': float(np.median(
                [r['train_loss_weighted'] for r in rows])),
            'loss_range': [float(min(r['train_loss_weighted'] for r in rows)),
                           float(max(r['train_loss_weighted'] for r in rows))],
            'iid_median': float(np.median(
                [r['ler_iid_p05']['ler'] for r in rows])),
            'iid_range': [float(min(r['ler_iid_p05']['ler'] for r in rows)),
                          float(max(r['ler_iid_p05']['ler'] for r in rows))],
            'w2_perfect': sum(1 for r in rows if r['ler_w2']['k'] == 0),
            'w3_perfect': sum(1 for r in rows if r['ler_w3']['k'] == 0)}
    res['entangler_summary'] = summ
    res['loss_ordering'] = sorted(GATES,
                                  key=lambda g: summ[g]['loss_median'])
    json.dump(res, open(OUT, 'w'), indent=2)
    print('[t3] part A summary:', res['loss_ordering'], flush=True)


def compare_samplers(res):
    """Rank the L=2 confirmation traces vs the committed L=16 fig8 traces."""
    names = ['full_batch', 'stratified_0.2', 'importance_0.2',
             'importance_0.35', 'full_basis_is_0.1']
    if 'sampler_comparison' in res:
        return
    if not all(os.path.exists(os.path.join(SAMPLER_DIR, f'{n}.npz'))
               for n in names):
        print('[t3] sampler L=2 traces incomplete — comparison skipped',
              flush=True)
        return
    comp = {'L2': {}, 'L16': {}}
    for tag, ddir in (('L2', SAMPLER_DIR), ('L16', FIG8_L16)):
        for n in names:
            f = os.path.join(ddir, f'{n}.npz')
            if not os.path.exists(f):
                comp[tag][n] = None
                continue
            z = np.load(f, allow_pickle=True)
            comp[tag][n] = {'min_loss': float(np.min(z['loss'])),
                            'final_op_evs': int(np.asarray(z['op_evs'])[-1])}
    for tag in ('L2', 'L16'):
        have = {k: v for k, v in comp[tag].items() if v}
        comp[f'{tag}_loss_ranking'] = sorted(
            have, key=lambda n: have[n]['min_loss'])
        comp[f'{tag}_opev_ranking'] = sorted(
            have, key=lambda n: have[n]['final_op_evs'])
    comp['ranking_match'] = bool(
        comp['L2_loss_ranking'] == comp['L16_loss_ranking'])
    res['sampler_comparison'] = comp
    json.dump(res, open(OUT, 'w'), indent=2)
    print(f"[t3] sampler ranking L2={comp['L2_loss_ranking']} "
          f"match_L16={comp['ranking_match']}", flush=True)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {
        'protocol': __doc__, 'units': {}}
    json.dump(res, open(OUT, 'w'), indent=2)
    part_a(res)
    compare_samplers(res)
    set_status(state='done')
    print('[t3] DONE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        set_status(state='FAILED', error=str(e))
        raise
