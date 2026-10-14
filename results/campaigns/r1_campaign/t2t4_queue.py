#!/usr/bin/env python3
"""Budget levers (single-transition MS, pruning frontier) and the correlated-
channel register family, run after the central code has been selected.

Requires t1_results.json to contain the selected central code (L*, code).

T2a — single-transition MS pilot: 4 seeds {300..303} at L=L*, weighted
      importance@0.2 recipe, entangler='ms1' (one MS per edge, transition
      (0,1) only; budget L*|E| = 10L* vs 20L*). Full instrument battery.
      Question: does halving the per-layer budget retain the loss floor +
      perfect w2?
T2b — pruning frontier: the central code; entangler MS gates with
      wrapped-|theta| below the {10,25,50}% percentile (over all (conn,
      transition) gates, theta wrapped to (-pi,pi]) are pruned by setting
      (phi,theta)=(0,0) and FREEZING them (gradient mask) through a
      300-step weighted fine-tune (importance@0.2 stream, Adam lr 0.01).
      Battery per threshold -> MS-budget->LER frontier.
T4  — weighted-MAP LER under the corrected correlated channel for
      ((n,3,3))_3 n=5..9 (reference codes) + the central code, eta in
      {0.8642, 0.90, 0.92, 0.9296, 0.93}; >=100 errors else escalate x2
      to a per-n shot cap; below-cap cells reported below-detection with
      Wilson UB (never zero).

Writes t2t4_results.json (sequential, resumable, incremental). Feeds the
pruning results of the paper.
Launch: nohup nice -n 19 python3 t2t4_queue.py >> t2t4.log 2>&1 &
"""
import os, sys, json, math, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import src.jax_backend  # x64 first
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
OUT = os.path.join(HERE, 't2t4_results.json')
STATUS = os.path.join(HERE, 't2t4_status.json')
CODEDIR = os.path.join(HERE, 't2_codes')
D, N, P, FRAC, STEPS = 3, 5, 0.05, 0.2, 3000
SEEDS_2A = [300, 301, 302, 303]
PRUNE_PCTS = [10, 25, 50]
ETAS = [0.8642, 0.90, 0.92, 0.9296, 0.93]
ETA_START_SHOTS = {0.8642: 10_000, 0.90: 10_000, 0.92: 20_000,
                   0.9296: 40_000, 0.93: 40_000}
SHOT_CAP = {5: 1_280_000, 6: 640_000, 7: 320_000, 8: 160_000, 9: 160_000}
MIN_ERRORS = 100

from t1_queue import wilson, measure_code, singles, CONNS  # instruments


def set_status(**kw):
    kw['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    json.dump(kw, open(STATUS, 'w'), indent=2)


def headline():
    t1 = json.load(open(os.path.join(HERE, 't1_results.json')))
    return t1['headline']


def t2a(res):
    import jax.numpy as jnp
    from src.seed_race import run_race
    from src.decoders._common import encoder_forward
    L = headline()['L_star']
    for seed in SEEDS_2A:
        key = f'2a_ms1_L{L}_s{seed}'
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
                     connectivity='all-to-all', entangler='ms1')
        row = {'part': '2a', 'L': L, 'seed': seed, 'npz': npz,
               'entangler': 'ms1',
               'ms_budget': L * len(CONNS),   # ONE gate per edge
               'train_loss_weighted': float(r['result']['full_loss']),
               'race_op_evs': int(r['race_op_evs']),
               'train_wall_s': round(time.time() - t0, 1)}
        cs = np.asarray(encoder_forward(npz, CONNS)[0])
        row.update(measure_code(cs))
        row['wall_s'] = round(time.time() - t0, 1)
        res['units'][key] = row
        json.dump(res, open(OUT, 'w'), indent=2)
        print(f"[t2a] {key}: wloss={row['train_loss_weighted']:.3e} "
              f"w2={row['ler_w2']['k']}/100k iid="
              f"{row['ler_iid_p05']['ler']:.3e} [{row['wall_s']}s]",
              flush=True)


def t2b(res):
    import jax, jax.numpy as jnp, optax
    from src.seed_race import build_race
    from src.loss import save_varqec_result
    from src.decoders._common import encoder_forward
    hd = headline()
    L = hd['L_star']
    src_npz = hd['selected_npz']
    data = np.load(src_npz, allow_pickle=True)
    theta0 = np.array(data['params'], dtype=float)
    nsing = 4 * N * (D - 1)                      # 40 at n=5 d=3
    nc = len(CONNS)
    ngates = L * nc * (D - 1)                    # (layer, conn, transition)
    # wrapped |theta| per entangler gate: theta at nsing + 4c + 2t + 1
    mags = np.empty((L, nc, D - 1))
    for c in range(nc):
        for t in range(D - 1):
            th = theta0[:, nsing + 4 * c + 2 * t + 1]
            mags[:, c, t] = np.abs((th + np.pi) % (2 * np.pi) - np.pi)
    if '2b_gate_ranking' not in res:
        res['2b_gate_ranking'] = {
            'ms_budget_unpruned': 2 * L * nc,
            'wrapped_theta_percentiles': {
                str(p): float(np.percentile(mags, p)) for p in PRUNE_PCTS}}
    (vg, full_wloss_fn, gs, ppl, _, fw, sgw) = build_race(
        D, N, 3, L, weighted=True)
    from src.sampling.stratified_importance import (
        make_stratified_importance_weights)
    total_ops = int(sum(gs))
    for pct in PRUNE_PCTS:
        key = f'2b_prune{pct}'
        if key in res['units']:
            continue
        set_status(state='running', unit=key)
        t0 = time.time()
        thr = np.percentile(mags, pct)
        pruned = mags <= thr
        theta = theta0.copy()
        gmask = np.ones_like(theta)              # 1 = trainable
        for (l, c, t) in zip(*np.nonzero(pruned)):
            base = nsing + 4 * c + 2 * t
            theta[l, base:base + 2] = 0.0        # gate -> identity
            gmask[l, base:base + 2] = 0.0        # frozen
        n_pruned = int(pruned.sum())
        th = jnp.asarray(theta)
        gm = jnp.asarray(gmask)
        opt = optax.adam(learning_rate=0.01)
        st = opt.init(th)
        rng = np.random.default_rng(9000 + pct)
        budget = max(1, int(round(FRAC * total_ops)))
        for step in range(300):                   # fine-tune, pruned frozen
            w = make_stratified_importance_weights(sgw, budget, rng)
            _, g = vg(th, w)
            g = jax.tree_util.tree_map(lambda x: x * gm, g)
            upd, st = opt.update(g, st, th)
            th = optax.apply_updates(th, upd)
            th = jnp.where(gm > 0, th, 0.0)   # keep pruned gates at identity
        wloss = float(full_wloss_fn(th))
        npz = os.path.join(CODEDIR, f'headline_prune{pct}.npz')
        save_varqec_result(npz, np.asarray(th), [wloss], f'native_d{D}', 3,
                           L, metadata={'K': D, 'n_qudit': N, 'd': D,
                                        'entangler': 'ms',
                                        'pruned_pct': pct,
                                        'n_gates_pruned': n_pruned,
                                        'source': src_npz})
        cs = np.asarray(encoder_forward(npz, CONNS)[0])
        row = {'part': '2b', 'pct': pct, 'threshold_wrapped_theta': float(thr),
               'n_gates_pruned': n_pruned,
               'ms_budget': 2 * L * nc - n_pruned,
               'wloss_after_finetune': wloss, 'npz': npz}
        row.update(measure_code(cs))
        row['wall_s'] = round(time.time() - t0, 1)
        res['units'][key] = row
        json.dump(res, open(OUT, 'w'), indent=2)
        print(f"[t2b] {key}: pruned {n_pruned}/{ngates} gates "
              f"budget={row['ms_budget']} wloss={wloss:.3e} "
              f"w2={row['ler_w2']['k']}/100k iid="
              f"{row['ler_iid_p05']['ler']:.3e} [{row['wall_s']}s]",
              flush=True)


def t4(res):
    from weighted_race_test import map_channel_ler
    hd = headline()
    codes = [(f'n{n}', f'results/campaigns/campaign/'
                       f'd3_n{n}_dist3_16L.npz', n) for n in range(5, 10)]
    codes.append(('t1_headline', hd['selected_npz'], 5))
    for name, npz, n in codes:
        for eta in ETAS:
            key = f't4_{name}@{eta}'
            if key in res['units']:
                continue
            set_status(state='running', unit=key)
            t0 = time.time()
            shots = ETA_START_SHOTS[eta]
            cap = SHOT_CAP[n]
            while True:
                ler, boot = map_channel_ler(npz, D, n, eta=eta,
                                            n_shots=shots,
                                            seed=41 + int(eta * 1e4))
                k = int(round(ler * shots))
                if k >= MIN_ERRORS or shots >= cap:
                    break
                shots = min(cap, shots * 2)
            lo, hi = wilson(k, shots)
            row = {'part': '4', 'code': name, 'n': n, 'npz': npz,
                   'eta': eta, 'shots': shots, 'k': k, 'ler': k / shots,
                   'wilson95': [lo, hi], 'bootstrap_ci': list(boot),
                   'resolved': bool(k >= MIN_ERRORS),
                   'below_detection': bool(k == 0),
                   'wall_s': round(time.time() - t0, 1)}
            res['units'][key] = row
            json.dump(res, open(OUT, 'w'), indent=2)
            print(f"[t4] {key}: k={k}/{shots} LER={k/shots:.3e} "
                  f"[{lo:.2e},{hi:.2e}] [{row['wall_s']}s]", flush=True)


def main():
    os.makedirs(CODEDIR, exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {
        'protocol': 'Budget levers (T2a single-transition MS, T2b pruning frontier) + T4 correlated-channel family', 'units': {}}
    t2a(res)
    t2b(res)
    t4(res)
    set_status(state='done', done=len(res['units']))
    print('[t2t4] QUEUE DONE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        set_status(state='FAILED', error=str(e))
        raise
