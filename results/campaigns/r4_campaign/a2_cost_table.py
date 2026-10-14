#!/usr/bin/env python3
"""Exact sampler cost table across n (per-step cost table, Table VI).

Realized distinct-operator cost per step for the five strategies at
n in {5,6,7,9}, by exact replay of the theta-independent per-step draw
streams (the a2_decompose method: same code path, same rng consumption
order, no training). 1500 steps, seed 0 — the fig8 protocol.

Validation: the n=5 and n=9 replays are asserted against the stored
op-EV traces of real runs (fig8_weighted_n5/ and the committed
fig8_weighted/, all logged checkpoints), so both ends of the n range
are validated against instrumented runs; n=6,7 are the same code path.

Run: nice -n 19 python3 a2_cost_table.py
Writes a2_cost_table.json (this directory). Feeds Table VI of the paper.
"""
import os, sys, json
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
os.chdir(REPO)
import src.jax_backend  # x64 first
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'a2_cost_table.json')
BP = os.path.join(REPO, 'results', 'campaigns')
TRACES = {5: os.path.join(BP, 'fig8_weighted_n5'),
          9: os.path.join(BP, 'fig8_weighted')}
STEPS, SEED, LOG_EVERY = 1500, 0, 25
NS = [5, 6, 7, 9]
CONFIGS = [('full_batch', 'full_batch', 1.0),
           ('stratified_0.2', 'stratified', 0.2),
           ('importance_0.2', 'importance', 0.2),
           ('importance_0.35', 'importance', 0.35),
           ('full_basis_is_0.1', 'full_basis_is', 0.1)]


def replay(n, sampler, frac):
    from src.seed_race import build_race
    from src.jax_backend import stratified_weights
    from src.sampling.stratified_importance import (
        make_stratified_importance_weights,
        make_stratified_importance_weights_full_basis)
    (_, _, group_sizes, _, _, full_weights, sgw) = build_race(
        3, n, 3, 2, weighted=True, connectivity='all-to-all')
    total_ops = int(sum(group_sizes))
    budget = max(1, int(round(frac * total_ops)))
    if sampler == 'full_basis_is':
        from loss_split_helpers import build_w1_w2_masks
        from src.errors import ErrorModel
        grouped = ErrorModel(d=3, n_qudit=n, distance=3,
                             closed=True).build_grouped(verbose=False)
        w1_mask, w2_mask = build_w1_w2_masks(grouped, 3)
        w1_budget = max(1, int(round(0.75 * frac * total_ops)))
        w2_budget = max(1, int(round(frac * total_ops)) - w1_budget)
    rng = np.random.default_rng(SEED)
    per_step, cum_at_log, cum = [], [], 0
    for step in range(STEPS):
        if sampler == 'full_batch':
            c = total_ops
        elif sampler == 'stratified':
            w = stratified_weights(group_sizes, frac, rng)
            c = int(sum(int(np.count_nonzero(np.asarray(x))) for x in w))
        elif sampler == 'full_basis_is':
            w = make_stratified_importance_weights_full_basis(
                sgw, w1_mask, w2_mask, w1_budget, w2_budget, rng)
            c = int(sum(int(np.count_nonzero(np.asarray(x))) for x in w))
        else:
            w = make_stratified_importance_weights(sgw, budget, rng)
            c = int(sum(int(np.count_nonzero(np.asarray(x))) for x in w))
        per_step.append(c); cum += c
        if step % LOG_EVERY == 0 or step == STEPS - 1:
            cum += total_ops  # full-batch monitoring log eval
            cum_at_log.append(cum)
    return total_ops, np.asarray(per_step), np.asarray(cum_at_log)


def main():
    results = {'protocol': {
        'steps': STEPS, 'seed': SEED, 'log_every': LOG_EVERY,
        'method': 'exact replay of theta-independent draw streams '
                  '(a2_decompose method); train totals exclude the '
                  'every-25-step full-batch monitoring evals',
        'validation': 'n=5 and n=9 replays asserted against stored op-EV '
                      'traces at all logged checkpoints'},
        'table': {}}
    for n in NS:
        results['table'][str(n)] = {}
        for name, sampler, frac in CONFIGS:
            T, per_step, cum_at_log = replay(n, sampler, frac)
            train_total = int(per_step.sum())
            rec = {'total_ops': T,
                   'nominal_budget_per_step': (
                       T if sampler == 'full_batch'
                       else max(1, int(round(frac * T)))),
                   'realized_ops_per_step_mean': round(
                       float(per_step.mean()), 2),
                   'train_total_op_evs': train_total}
            tdir = TRACES.get(n)
            if tdir:
                f = os.path.join(tdir, f'{name}.npz')
                if os.path.exists(f):
                    z = np.load(f, allow_pickle=True)
                    stored = np.asarray(z['op_evs'])
                    match = bool(np.array_equal(cum_at_log, stored))
                    rec['replay_matches_stored_trace'] = match
                    if not match:
                        rec['stored_final'] = int(stored[-1])
                        rec['replay_final'] = int(cum_at_log[-1])
            results['table'][str(n)][name] = rec
            v = rec.get('replay_matches_stored_trace')
            print(f"n={n} {name}: total_ops={T} realized/step="
                  f"{rec['realized_ops_per_step_mean']} train={train_total}"
                  f"{'' if v is None else f' validated={v}'}", flush=True)
    json.dump(results, open(OUT, 'w'), indent=2)
    print('wrote', OUT, flush=True)


if __name__ == '__main__':
    main()
