#!/usr/bin/env python3
"""((7,3,4))_3 pilot: one 500-step timing run.

Recipe: the weighted importance@0.2 line at distance=4 (hw closure
extended to weight-3 products, 2661 ops — t5c_closure.json), n=7, L=16 a2a
MS, single seed. Falls back to the unweighted hw-closure arm if the
weighted machinery rejects distance=4. Records s/step and the projected
3000-step per-seed wall; the go/no-go gate is <=6 h/seed. If GO, the
2-seed train (t5c_train.py) is launched separately; this script only
measures. Writes t5c_timing.json.
"""
import os, sys, json, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import src.jax_backend  # x64 first
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 't5c_timing.json')
STEPS = 500


def main():
    from src.seed_race import run_race
    t0 = time.time()
    npz = os.path.join(HERE, 't1_codes', 't5c_n7_d4_timing.npz')
    arm, err = 'weighted_importance_0.2', None
    try:
        r = run_race(d=3, n=7, distance=4, layers=16, num_seeds=1,
                     steps=STEPS, target_loss=1e-9, sample_frac=0.2,
                     sampler='importance', trigger_margin=0.25,
                     trigger_patience=5, out=npz, first_seed=100,
                     verbose=False, weighted=True,
                     connectivity='all-to-all', entangler='ms')
    except Exception as e:
        err = f'weighted arm failed: {e!r} -- falling back to unweighted hw'
        print('[t5c]', err, flush=True)
        arm = 'unweighted_hw_fullbatch'
        t0 = time.time()
        r = run_race(d=3, n=7, distance=4, layers=16, num_seeds=1,
                     steps=STEPS, target_loss=1e-9, noise='hw',
                     out=npz, first_seed=100, verbose=False,
                     connectivity='all-to-all', entangler='ms')
    wall = time.time() - t0
    s_per_step = wall / STEPS   # incl. build+compile amortized; noted
    proj_3000 = 3000 * s_per_step / 3600
    doc = {'arm': arm, 'fallback_error': err, 'steps': STEPS,
           'wall_s': round(wall, 1),
           's_per_step_incl_build': round(s_per_step, 3),
           'projected_3000step_hours': round(proj_3000, 2),
           'go_full_train_le_6h': bool(proj_3000 <= 6.0),
           'loss_at_500': float(r['result']['full_loss']),
           'race_op_evs': int(r['race_op_evs']), 'npz': npz,
           'note': 'shared box (nice -19); s/step includes one-time '
                   'build+compile amortized over 500 steps'}
    json.dump(doc, open(OUT, 'w'), indent=2)
    print('[t5c]', json.dumps(doc), flush=True)


if __name__ == '__main__':
    main()
