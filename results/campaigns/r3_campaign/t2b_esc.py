#!/usr/bin/env python3
"""q-panel probe escalation — tighten conditioned probes so the q panel
carries a p-DEPENDENT reconstructed curve/bound instead of a flat Wilson
floor.

q5 selected code: w2/w3 at 1M shots, w4/w5 at 2M shots (seeds 2002..2005).
If still zero, the panel reports the reconstructed UPPER BOUND
sum_w P(w|p) * WilsonUB(0, shots_w) — p-dependent by construction.
q4 selected code: w4 at 1M (seed 2004) to tighten the p=0.15 validation
factor (was 28/100k).

Appends to t2b_results.json under 'escalated_probes'. Feeds the q = 4, 5
code results of the paper.
"""
import os, sys, json, math, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)
import src.jax_backend  # x64 first
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 't2b_results.json')
N = 5


def wilson_ub(k, n, z=1.96):
    if n == 0:
        return 1.0
    ph = k / n
    dn = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / dn
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / dn
    return c + h


def main():
    import t2b_queue as t
    res = json.load(open(OUT))
    esc = res.setdefault('escalated_probes', {})
    jobs = [('q5', 2, 2002, 1_000_000), ('q5', 3, 2003, 1_000_000),
            ('q5', 4, 2004, 2_000_000), ('q5', 5, 2005, 2_000_000),
            ('q4', 4, 2004, 1_000_000)]
    from src.simulation import simulate_ler_with_correction_factored
    for qkey, w, seed, shots in jobs:
        key = f'{qkey}_w{w}_esc'
        if key in esc:
            continue
        d = int(qkey[1])
        sel = res['minimal_L'][qkey]
        cs = t.code_states_from(sel['selected_npz'])
        t0 = time.time()
        rr = simulate_ler_with_correction_factored(
            cs, t.wk_noise(d, w), t.singles(d), N, d,
            n_shots=shots, seed=seed)
        k = int(round(float(rr['logical_error_rate']) * shots))
        esc[key] = {'w': w, 'seed': seed, 'shots': shots, 'k': k,
                    'ler': k / shots, 'wilson_ub': wilson_ub(k, shots),
                    'wall_s': round(time.time() - t0, 1)}
        json.dump(res, open(OUT, 'w'), indent=2)
        print(f'[t2b_esc] {key}: k={k}/{shots} '
              f'[{esc[key]["wall_s"]}s]', flush=True)
    # reconstructed bound table for q5 (and updated q4 reconstruction)
    for qkey in ('q5', 'q4'):
        probes = {}
        for w in (1, 2, 3, 4, 5):
            e = esc.get(f'{qkey}_w{w}_esc')
            if e is None:
                pr = res['panel'][f'panel_{qkey}']['probes'][f'ler_w{w}']
                probes[w] = (pr['k'], pr['n'])
            else:
                probes[w] = (e['k'], e['shots'])
        tab = {}
        for p in t.P_GRID:
            pt = sum(math.comb(N, w) * p ** w * (1 - p) ** (N - w)
                     * (probes[w][0] / probes[w][1]) for w in probes)
            ub = sum(math.comb(N, w) * p ** w * (1 - p) ** (N - w)
                     * wilson_ub(*probes[w]) for w in probes)
            tab[str(p)] = {'reconstructed': pt, 'reconstructed_ub': ub}
        res[f'reconstruction_{qkey}_escalated'] = {
            'probes': {str(w): probes[w] for w in probes}, 'cells': tab}
        json.dump(res, open(OUT, 'w'), indent=2)
        print(f'[t2b_esc] {qkey} reconstruction table written', flush=True)
    print('[t2b_esc] DONE', flush=True)


if __name__ == '__main__':
    main()
