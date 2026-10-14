#!/usr/bin/env python3
"""Best-of-N seed selection, offline: expected best-of-N curves from the
loss_ler_diag/t1g dataset (no new training; the 16 t1g codes ARE the race).

For each basis (hw / full, 8 single-seed codes each) and each N=1..8,
enumerate ALL C(8,N) seed subsets exactly and select one code per subset by:
  (a) final training loss        — the old selector, refuted by t1g;
  (b) conditioned LER|w=2 probe  — 100k shots/code, the candidate cheap
                                   selector (ties broken uniformly: tied
                                   winners contribute their mean truth);
  (c) true iid LER @300k         — oracle (winner's-curse caveat: selection
                                   and scoring share the same measurement).
Score = the selected code's measured iid LER @p=0.05 (300k, seed 2000 — the
t1g instrument; the three full-basis standouts re-measured at 1M agree within
CI). Budgets: train op-EVs = N x per-seed race_op_evs (hw 663,221 / full
2,403,801 — full-batch, zero cert checks, from the npz metadata); probe cost
reported SEPARATELY in shots ((b): N x 100k conditioned; (c): N x 300k iid).

Selector-(b) robustness: parametric bootstrap of the probe measurement
(k* ~ Binom(100k, p_hat) per code, B=2000) — does probe noise change the
selection?

Writes c1_bestofn.json + c1_bestofn.png (also re-renderable via
scripts/make_figures.py). Pure numpy; seconds. Feeds the best-of-N
results of the paper.
"""
import os, sys, json
from itertools import combinations
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T1G = os.path.join(REPO, 'results', 'campaigns', 'loss_ler_diag',
                   't1g_results.json')
OUT = os.path.join(HERE, 'c1_bestofn.json')
PNG = os.path.join(HERE, 'c1_bestofn.png')
CLASS_UB = 2.8e-4   # the "~2.5e-4 class" (standouts' Wilson UBs are ~2.8e-4)
DISASTER = 1.0e-3   # the generic bad-weight-2 floor class
PROBE_SHOTS, IID_SHOTS = 100_000, 300_000


def subset_expectation(vals_sel, vals_truth, N, extra=None):
    """Exact E[truth of argmin(sel)] over all C(8,N) subsets, uniform
    tie-breaking (tied winners contribute their mean). Returns expectation,
    subset-level distribution, and P(truth in class) style probabilities."""
    idx = range(len(vals_sel))
    per_subset, p_class, p_dis = [], [], []
    for sub in combinations(idx, N):
        s = np.asarray([vals_sel[i] for i in sub])
        t = np.asarray([vals_truth[i] for i in sub])
        winners = np.flatnonzero(s == s.min())
        tw = t[winners]
        per_subset.append(float(tw.mean()))
        p_class.append(float((tw <= CLASS_UB).mean()))
        p_dis.append(float((tw >= DISASTER).mean()))
    ps = np.asarray(per_subset)
    return {'expected_ler': float(ps.mean()),
            'subset_q10': float(np.quantile(ps, 0.10)),
            'subset_q50': float(np.quantile(ps, 0.50)),
            'subset_q90': float(np.quantile(ps, 0.90)),
            'p_in_2.8e-4_class': float(np.mean(p_class)),
            'p_disaster_ge_1e-3': float(np.mean(p_dis)),
            'n_subsets': len(ps)}


def probe_bootstrap(w2_hat, truth, N, B=2000, rng=None):
    """Selector-(b) with probe measurement noise: per replica redraw each
    code's probe count k* ~ Binom(100k, p_hat), select min k*, tie-average."""
    rng = rng or np.random.default_rng(0)
    idx = list(range(len(w2_hat)))
    subs = list(combinations(idx, N))
    out = np.empty(B)
    truth = np.asarray(truth)
    w2 = np.asarray(w2_hat)
    for b in range(B):
        kstar = rng.binomial(PROBE_SHOTS, w2)
        sub = subs[rng.integers(len(subs))]
        ks = kstar[list(sub)]
        winners = np.flatnonzero(ks == ks.min())
        out[b] = truth[list(sub)][winners].mean()
    return {'expected_ler_boot': float(out.mean()),
            'boot_q10': float(np.quantile(out, 0.10)),
            'boot_q90': float(np.quantile(out, 0.90))}


def main():
    d = json.load(open(T1G))
    units = d['units']
    data = {}
    for basis in ('hw', 'full'):
        rows = sorted((u for u in units.values() if u['basis'] == basis),
                      key=lambda u: u['seed'])
        assert len(rows) == 8
        data[basis] = {
            'seeds': [u['seed'] for u in rows],
            'loss': [u['train_loss'] for u in rows],
            'w2': [u['ler_w2']['ler'] for u in rows],
            'iid': [u['ler_iid_p05']['ler'] for u in rows]}
    # per-seed train op-EVs from the npz metadata (identical within a basis)
    per_seed_ev = {}
    for basis in ('hw', 'full'):
        z = np.load(os.path.join(REPO, 'results', 'campaigns',
                                 'loss_ler_diag', 't1g',
                                 f'{basis}_s100.npz'), allow_pickle=True)
        per_seed_ev[basis] = int(z['race_op_evs'])

    res = {'inputs': {'per_seed_train_op_evs': per_seed_ev,
                      'probe_shots_per_code': PROBE_SHOTS,
                      'iid_shots_per_code': IID_SHOTS,
                      'class_ub': CLASS_UB, 'disaster': DISASTER,
                      'data': data},
           'curves': {}}
    selectors = {'loss': ('loss', 0),
                 'w2_probe': ('w2', PROBE_SHOTS),
                 'oracle_iid': ('iid', IID_SHOTS)}
    for basis in ('hw', 'full'):
        for sel, (field, shots_per) in selectors.items():
            pts = []
            for N in range(1, 9):
                e = subset_expectation(data[basis][field],
                                       data[basis]['iid'], N)
                e['N'] = N
                e['train_op_evs'] = N * per_seed_ev[basis]
                e['probe_shots'] = N * shots_per
                if sel == 'w2_probe':
                    e['probe_boot'] = probe_bootstrap(
                        data[basis]['w2'], data[basis]['iid'], N,
                        rng=np.random.default_rng(1234 + N))
                pts.append(e)
            res['curves'][f'{basis}|{sel}'] = pts

    # headline comparisons
    hw1 = res['curves']['hw|loss'][0]
    fw4 = res['curves']['full|w2_probe'][3]
    fw8 = res['curves']['full|w2_probe'][7]
    res['headline'] = {
        '1seed_hw_recipe': {'expected_ler': hw1['expected_ler'],
                            'train_op_evs': hw1['train_op_evs']},
        'bestof4_full_by_w2': fw4, 'bestof8_full_by_w2': fw8,
        'budget_ratio_bestof4full_vs_1seedhw':
            round(fw4['train_op_evs'] / hw1['train_op_evs'], 2)}
    json.dump(res, open(OUT, 'w'), indent=2)
    print(json.dumps(res['headline'], indent=2))

    # figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    styles = {'hw|loss': ('tab:blue', 'o', '-'),
              'hw|w2_probe': ('tab:blue', 's', '--'),
              'hw|oracle_iid': ('tab:blue', '^', ':'),
              'full|loss': ('tab:red', 'o', '-'),
              'full|w2_probe': ('tab:red', 's', '--'),
              'full|oracle_iid': ('tab:red', '^', ':')}
    for key, pts in res['curves'].items():
        c, m, ls = styles[key]
        x = [p['train_op_evs'] for p in pts]
        y = [p['expected_ler'] for p in pts]
        lo = [p['subset_q10'] for p in pts]
        hi = [p['subset_q90'] for p in pts]
        ax.plot(x, y, marker=m, ls=ls, color=c, label=key.replace('|', ' / '),
                ms=4, lw=1.3)
        ax.fill_between(x, lo, hi, color=c, alpha=0.08, lw=0)
    for pts in (res['curves']['full|w2_probe'],):
        for p in pts:
            if p['N'] in (1, 4, 8):
                ax.annotate(f"N={p['N']}", (p['train_op_evs'],
                                            p['expected_ler']),
                            textcoords='offset points', xytext=(4, -10),
                            fontsize=7, color='tab:red')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('total TRAIN op-EVs (N seeds x per-seed race cost)')
    ax.set_ylabel('expected iid LER @ p=0.05 of the SELECTED code')
    ax.set_title('Best-of-N seed selection: expected LER vs training budget\n'
                 '(t1g codes, n=5 L=16; probe shot costs reported '
                 'separately)', fontsize=10)
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(PNG, dpi=160)
    print('wrote', OUT, 'and', PNG)


if __name__ == '__main__':
    main()
