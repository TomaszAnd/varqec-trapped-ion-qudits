#!/usr/bin/env python3
"""Early-stopped loss selector: does early stopping rescue loss-based seed
selection?

Reruns the 8 full-basis loss_ler_diag/t1g seeds (100-107) through the
dual-snapshot trainer of r1_campaign/t1_queue.py (train_full_basis_dual:
run_race theta-init convention, 3-stage LR, residual-plateau rule 3x<1%
100-step windows; plateau + convergence snapshots), n=5 L=16 a2a MS,
UNWEIGHTED full closure basis, full batch, 3000 steps. Each snapshot gets
the full instrument battery (w1@200k/w2@100k/w3@100k + 1M iid p=0.05).

Validation: the convergence snapshots re-walk the t1g trajectories (same
init convention, deterministic full-batch loop) - conv losses are compared
against the stored t1g_results.json residuals.

Scoring (exact subset enumeration, best-of-N method, ties averaged):
E[best-of-N] for N in {1,2,4,8} over all C(8,N) subsets, selectors:
  plateau_loss  -> score = selected seed's PLATEAU-snapshot iid LER
  conv_loss     -> score = selected seed's CONV-snapshot iid LER (control)
  w2_plateau    -> plateau w2-probe k, score = plateau iid LER
  oracle_plateau-> plateau iid LER itself (floor)
Pooled 12-seed cross-check adds the Part-B seeds 200-203 of t1_queue.py.
Writes t6d_results.json (resumable).
"""
import os, sys, json, time
from itertools import combinations
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
R1DIR = os.path.join(REPO, 'results/campaigns/r1_campaign')
sys.path.insert(0, REPO)
sys.path.insert(0, R1DIR)
os.chdir(REPO)

import src.jax_backend  # x64 first
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODEDIR = os.path.join(HERE, 't6d_codes')
OUT = os.path.join(HERE, 't6d_results.json')
SEEDS = list(range(100, 108))


def main():
    from t1_queue import train_full_basis_dual, measure_code, wilson, \
        CONNS, MSBUDGET, LB, D
    from src.loss import save_varqec_result
    from src.decoders._common import encoder_forward
    os.makedirs(CODEDIR, exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {'units': {}}

    t1g = json.load(open(os.path.join(
        REPO, 'results/campaigns/loss_ler_diag/t1g_results.json')))

    for seed in SEEDS:
        key = f'F_s{seed}'
        if key in res['units']:
            continue
        t0 = time.time()
        tr = train_full_basis_dual(seed)
        row = {'seed': seed, 'ms_budget': MSBUDGET(LB),
               'plateau_step': tr['plateau_step'],
               'plateau_never_detected': tr['plateau_never_detected'],
               'conv_step': tr['conv_step'],
               'train_wall_s': round(time.time() - t0, 1),
               'snapshots': {}}
        for tag in ('plateau', 'conv'):
            npz = os.path.join(CODEDIR, f'{key}_{tag}.npz')
            save_varqec_result(
                npz, tr[f'{tag}_theta'], [tr[f'{tag}_loss']],
                f'native_d{D}', 3, LB,
                metadata={'K': D, 'n_qudit': 5, 'd': D, 'seed': seed,
                          'entangler': 'ms', 'connectivity': 'all-to-all',
                          'noise': 'full', 'snapshot': tag,
                          'backend': 'jax'})
            cs = np.asarray(encoder_forward(npz, CONNS)[0])
            snap = {'npz': npz, 'full_basis_loss': tr[f'{tag}_loss']}
            snap.update(measure_code(cs))
            row['snapshots'][tag] = snap
            print(f"[t6d] {key}/{tag}: loss={snap['full_basis_loss']:.3e} "
                  f"w2={snap['ler_w2']['k']} "
                  f"iid={snap['ler_iid_p05']['ler']:.3e}", flush=True)
        # trajectory-identity check vs the stored t1g full-arm residual
        t1g_key = f'full_s{seed}'
        if t1g_key in t1g.get('units', {}):
            stored = t1g['units'][t1g_key].get('train_loss',
                     t1g['units'][t1g_key].get('final_loss'))
            row['t1g_stored_residual'] = stored
        row['wall_s'] = round(time.time() - t0, 1)
        res['units'][key] = row
        json.dump(res, open(OUT, 'w'), indent=2)

    # ---------------- scoring ----------------
    def bestofn(rows, keyfn, scorefn, N):
        vals = []
        for sub in combinations(rows, N):
            m = min(keyfn(r) for r in sub)
            picks = [scorefn(r) for r in sub if keyfn(r) == m]
            vals.append(sum(picks) / len(picks))
        return float(np.mean(vals))

    def score_block(rows, label):
        sel = {
            'plateau_loss': (
                lambda r: r['snapshots']['plateau']['full_basis_loss'],
                lambda r: r['snapshots']['plateau']['ler_iid_p05']['ler']),
            'conv_loss': (
                lambda r: r['snapshots']['conv']['full_basis_loss'],
                lambda r: r['snapshots']['conv']['ler_iid_p05']['ler']),
            'w2_plateau': (
                lambda r: r['snapshots']['plateau']['ler_w2']['k'],
                lambda r: r['snapshots']['plateau']['ler_iid_p05']['ler']),
            'oracle_plateau': (
                lambda r: r['snapshots']['plateau']['ler_iid_p05']['ler'],
                lambda r: r['snapshots']['plateau']['ler_iid_p05']['ler']),
        }
        out = {}
        for name, (kf, sf) in sel.items():
            out[name] = {f'N{N}': bestofn(rows, kf, sf, N)
                         for N in (1, 2, 4, 8) if N <= len(rows)}
        return out

    rows8 = [res['units'][f'F_s{s}'] for s in SEEDS]
    res['scoring_8seed'] = score_block(rows8, '8seed')

    # pooled 12-seed cross-check with the t1_queue.py Part-B seeds
    t1r = json.load(open(os.path.join(R1DIR, 't1_results.json')))
    rowsB = [t1r['units'][f'B_s{s}'] for s in (200, 201, 202, 203)]
    res['scoring_pooled12'] = score_block(rows8 + rowsB, 'pooled12')
    json.dump(res, open(OUT, 'w'), indent=2)
    print('[t6d] scoring:', json.dumps(res['scoring_8seed'], indent=1),
          flush=True)
    print('[t6d] DONE', flush=True)


if __name__ == '__main__':
    main()
