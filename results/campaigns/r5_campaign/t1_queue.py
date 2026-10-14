#!/usr/bin/env python3
"""Entangling-gate study at L=2 with twelve runs per gate (Fig. 9):
retuned-schedule entangler ensemble.

Diagnosis: the r3_campaign/t3 4-seed L=2 trajectories' end-of-training
oscillation is ~100% importance-sampler ESTIMATOR noise at stationary
parameters (fixed-theta estimator spread reproduces the observed tail
spread per gate: ms 1.44x vs 1.41x ... csum 46x vs 57x), and the saved
codes were picked by the noisy sampled argmin (winner's curse; csum sampled
min 2.6e-6 vs full-batch 4.3e-4). No LR-driven parameter oscillation exists
(the loss-keyed LR staircase is one-way), but its final-stage entry is a
first-passage event of the noisy estimator, i.e. gate/seed-arbitrary.

Retune (options in src/seed_race.py, defaults off elsewhere):
  - monitor_every=25: full-batch loss every 25 steps -> de-noised trace in
    the npz ('monitor_steps'/'monitor_losses'), selection of uncertified
    codes by monitored full-batch minimum (kills the winner's curse).
  - anneal_steps=500, final_lr=1e-4: step-keyed cosine anneal of the final
    LR (1e-3 -> 1e-4) over the last 500 steps, identical for every gate/seed
    (replaces the noisy loss-keyed stage entry at the tail; at L=2 the
    plateau is reached by ~step 500, so the tail is pure refinement).

Ensemble: gates {ms, zz, ls, csum, cex} x seeds {100..111} (12/gate),
((6,3,3))_3 L=2 a2a, weighted importance@0.2, 3000 steps (t3 protocol
otherwise unchanged). Per unit:
  - w1@200k / w2@100k / w3@100k conditioned probes (seeds 1001-1003);
  - iid LER @p=0.05, shots escalated 1M x2 until k>=300 or 8M cap
    (bars must reflect seed spread, not shot noise);
  - weighted-MAP LER under the corrected correlated channel @ eta=0.8642,
    20k shots (the MAP-bar instrument).

Writes t1_results.json (incremental, resume anchor). Feeds Fig. 9.
Launch: nohup nice -n 19 python3 t1_queue.py >> t1.log 2>&1 &
"""
import os, sys, json, math, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
os.chdir(REPO)
import src.jax_backend  # x64 first
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 't1_results.json')
STATUS = os.path.join(HERE, 't1_status.json')
CODEDIR = os.path.join(HERE, 't1_codes')
os.makedirs(CODEDIR, exist_ok=True)

D, N, L = 3, 6, 2
GATES = ['ms', 'zz', 'ls', 'csum', 'cex']
SEEDS = list(range(100, 112))  # 12 seeds/gate
STEPS = 3000
FRAC = 0.2
P = 0.05
ETA_MAP = 0.8642
MAP_SHOTS = 20_000
IID_START, IID_CAP, IID_KMIN = 1_000_000, 8_000_000, 300
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


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {
        'protocol': {
            'code': f'(({N},3,3))_{D}', 'L': L, 'steps': STEPS,
            'objective': 'channel-weighted KL, importance@0.2',
            'retune': {'monitor_every': 25, 'anneal_steps': 500,
                       'final_lr': 1e-4,
                       'selection': 'monitored full-batch minimum'},
            'probes': 'w1@200k w2@100k w3@100k (seeds 1001-1003)',
            'iid': f'p={P}, 1M x2 escalation to k>={IID_KMIN} or {IID_CAP}',
            'map': f'weighted MAP @ eta={ETA_MAP}, {MAP_SHOTS} shots',
            'seeds': SEEDS},
        'units': {}}
    from src.seed_race import run_race
    from src.decoders._common import encoder_forward
    from src.simulation import simulate_ler_with_correction_factored
    from src.errors import make_hardware_noise_fn
    from weighted_race_test import map_channel_ler

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
                         connectivity='all-to-all', entangler=gate,
                         monitor_every=25, anneal_steps=500, final_lr=1e-4)
            train_wall = time.time() - t0
            f = np.load(npz)
            cs = np.asarray(encoder_forward(npz, CONNS)[0])
            row = {'gate': gate, 'seed': seed, 'npz': npz, 'L': L,
                   'train_loss_weighted': float(r['result']['full_loss']),
                   'selected_by': str(f['selected_by']),
                   'monitor_final': float(f['monitor_losses'][-1]),
                   'race_op_evs': int(r['race_op_evs']),
                   'train_wall_s': round(train_wall, 1)}
            for w, sd, shots in ((1, 1001, 200_000), (2, 1002, 100_000),
                                 (3, 1003, 100_000)):
                rr = simulate_ler_with_correction_factored(
                    cs, wk_noise(w), singles(), N, D, n_shots=shots, seed=sd)
                k = int(round(float(rr['logical_error_rate']) * shots))
                row[f'ler_w{w}'] = {'k': k, 'n': shots, 'ler': k / shots,
                                    'wilson95': wilson(k, shots)}
            # escalated iid LER @ p=0.05
            shots = IID_START
            while True:
                rr = simulate_ler_with_correction_factored(
                    cs, make_hardware_noise_fn(D, N, P), singles(), N, D,
                    n_shots=shots, seed=2000)
                k = int(round(float(rr['logical_error_rate']) * shots))
                if k >= IID_KMIN or shots >= IID_CAP:
                    break
                shots = min(2 * shots, IID_CAP)
            row['ler_iid_p05'] = {'k': k, 'n': shots, 'ler': k / shots,
                                  'wilson95': wilson(k, shots)}
            # weighted-MAP under the corrected correlated channel
            t1 = time.time()
            ler, boot = map_channel_ler(npz, D, N, eta=ETA_MAP,
                                        n_shots=MAP_SHOTS, seed=7)
            row['map_eta08642'] = {'ler': float(ler),
                                   'boot95': [float(boot[0]), float(boot[1])],
                                   'n': MAP_SHOTS,
                                   'wall_s': round(time.time() - t1, 1)}
            row['wall_s'] = round(time.time() - t0, 1)
            res['units'][key] = row
            json.dump(res, open(OUT, 'w'), indent=2)
            print(f"[t1] {key}: wloss={row['train_loss_weighted']:.3e} "
                  f"({row['selected_by']}) w2={row['ler_w2']['k']}/100k "
                  f"iid={row['ler_iid_p05']['k']}/{row['ler_iid_p05']['n']} "
                  f"map={row['map_eta08642']['ler']:.3e} "
                  f"[{row['wall_s']}s]", flush=True)

    # per-gate summary (median + IQR: the figure statistics)
    summ = {}
    for gate in GATES:
        rows = [res['units'][f'{gate}_s{s}'] for s in SEEDS]
        losses = [r['train_loss_weighted'] for r in rows]
        iids = [r['ler_iid_p05']['ler'] for r in rows]
        maps = [r['map_eta08642']['ler'] for r in rows]
        q = lambda v: [float(np.percentile(v, x)) for x in (25, 50, 75)]
        summ[gate] = {
            'loss_iqr': q(losses), 'iid_iqr': q(iids), 'map_iqr': q(maps),
            'w2_perfect': sum(1 for r in rows if r['ler_w2']['k'] == 0),
            'w3_perfect': sum(1 for r in rows if r['ler_w3']['k'] == 0),
            'n_seeds': len(rows)}
    res['gate_summary'] = summ
    json.dump(res, open(OUT, 'w'), indent=2)
    set_status(state='done')
    print('[t1] DONE', flush=True)


if __name__ == '__main__':
    main()
