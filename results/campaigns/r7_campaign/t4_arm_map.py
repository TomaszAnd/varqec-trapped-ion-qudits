#!/usr/bin/env python3
"""Arm-wide statistics for the correlated-channel distance-trio ordering
(d3 < d4 ~ d2).

Existing cells (r3_campaign/t2c_results.json 'eval'): the SELECTED code per arm
(d2_L2_s100, d3_L2_s100, d4_L2_s102) at eta in {0.90, 0.92, 0.93}, >=100 errors.
This queue measures the 9 NON-selected t2c codes (3 arms x remaining 3 seeds)
with the identical instrument (weighted_race_test.map_channel_ler: corrected
Gaussian Meth channel + corrected w<=2 MAP prior, n=7 d=3), FIXED shots per eta
(20k / 40k / 80k, chosen from the existing cells' LERs to land >=~10 errors per
cell and >=~50 pooled per arm-eta), one escalation x2 only if k < 10.

Input to t4_verdict.py: per-seed cells + arm-pooled counts (fresh-only and
fresh+existing). Writes t4_{arm}_results.json.

Launch (one worker per arm, nice -19, detached):
  for a in d2 d3 d4; do nohup nice -n 19 python3 t4_arm_map.py --arm $a \
      >> t4_$a.log 2>&1 & echo $! > t4_$a.pid; done
Resumable: completed cells are skipped on relaunch.
"""
import os, sys, json, math, time, argparse

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
os.chdir(REPO)
import src.jax_backend  # x64 first
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T2C = os.path.join(REPO, 'results/campaigns/r3_campaign/t2c_codes')

D, N = 3, 7
ETAS = [0.90, 0.92, 0.93]
SHOTS = {0.90: 20_000, 0.92: 40_000, 0.93: 80_000}
ESC_MIN_ERR = 10          # escalate x2 (once) only below this
SELECTED = {'d2': 100, 'd3': 100, 'd4': 102}   # already measured in t2c
ALL_SEEDS = [100, 101, 102, 103]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    dn = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / dn
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / dn
    return (max(0.0, c - h), c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['d2', 'd3', 'd4'])
    args = ap.parse_args()
    arm = args.arm
    out_path = os.path.join(HERE, f't4_{arm}_results.json')
    status_path = os.path.join(HERE, f't4_{arm}_status.json')

    def set_status(**kw):
        kw['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        json.dump(kw, open(status_path, 'w'), indent=2)

    res = json.load(open(out_path)) if os.path.exists(out_path) else {
        'protocol': __doc__, 'arm': arm, 'cells': {}}

    from weighted_race_test import map_channel_ler

    seeds = [s for s in ALL_SEEDS if s != SELECTED[arm]]
    for seed in seeds:
        npz = os.path.join(T2C, f'{arm}_L2_s{seed}.npz')
        assert os.path.exists(npz), npz
        for eta in ETAS:
            key = f's{seed}@{eta}'
            if key in res['cells']:
                continue
            set_status(state='running', cell=key)
            t0 = time.time()
            shots = SHOTS[eta]
            while True:
                ler, boot = map_channel_ler(npz, D, N, eta=eta, n_shots=shots,
                                            seed=71 + seed * 13 + int(eta * 1e4))
                k = int(round(ler * shots))
                if k >= ESC_MIN_ERR or shots >= 2 * SHOTS[eta]:
                    break
                shots *= 2
            res['cells'][key] = {
                'seed': seed, 'eta': eta, 'k': k, 'shots': shots,
                'ler': k / shots, 'wilson95': wilson(k, shots),
                'bootstrap_ci': list(boot) if boot is not None else None,
                'wall_s': round(time.time() - t0, 1)}
            json.dump(res, open(out_path, 'w'), indent=2)
            print(f'[t4 {arm}] {key}: k={k}/{shots} = {k/shots:.3e} '
                  f'[{res["cells"][key]["wall_s"]}s]', flush=True)
    set_status(state='done')
    print(f'[t4 {arm}] DONE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        json.dump({'state': 'FAILED', 'error': str(e),
                   'updated': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(os.path.join(HERE, f't4_FAILED.json'), 'w'))
        raise
