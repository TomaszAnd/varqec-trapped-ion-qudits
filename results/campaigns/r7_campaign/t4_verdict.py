#!/usr/bin/env python3
"""Arm-wide statistics for the correlated distance-trio ordering.

Merges t4_{d2,d3,d4}_results.json (9 fresh codes, from t4_arm_map.py) with
the existing selected-code cells (r3_campaign/t2c_results.json 'eval') and
tests, per eta: does d3 < d4 ~ d2 survive 4 seeds per arm?
Writes t4_verdict.json + printed table.
"""
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
T2C = os.path.join(HERE, '../r3_campaign/t2c_results.json')
SELECTED = {'d2': 100, 'd3': 100, 'd4': 102}
ETAS = ['0.9', '0.92', '0.93']


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    dn = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / dn
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / dn
    return (max(0.0, c - h), c + h)


def ztest(k1, n1, k2, n2):
    """Two-proportion z (pooled)."""
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return ((k1 / n1 - k2 / n2) / se) if se > 0 else 0.0


cells = {a: {} for a in ('d2', 'd3', 'd4')}   # arm -> eta -> list[(seed,k,n)]
for arm in cells:
    r = json.load(open(os.path.join(HERE, f't4_{arm}_results.json')))
    for c in r['cells'].values():
        cells[arm].setdefault(str(c['eta']), []).append(
            (c['seed'], c['k'], c['shots'], 'fresh'))
t2c = json.load(open(T2C))['eval']
for key, c in t2c.items():
    if not key.startswith('map_'):
        continue
    arm = f"d{c['dist']}"
    cells[arm].setdefault(str(c['eta']), []).append(
        (SELECTED[arm], c['k'], c['shots'], 'selected'))

out = {'per_eta': {}}
for eta in ETAS:
    row = {}
    for arm in ('d2', 'd3', 'd4'):
        lst = sorted(cells[arm].get(eta, []))
        pooled_k = sum(k for _, k, _, _ in lst)
        pooled_n = sum(n for _, _, n, _ in lst)
        row[arm] = {
            'per_seed': [{'seed': s, 'k': k, 'n': n, 'ler': k / n, 'src': src,
                          'wilson95': wilson(k, n)} for s, k, n, src in lst],
            'pooled_k': pooled_k, 'pooled_n': pooled_n,
            'pooled_ler': pooled_k / pooled_n,
            'pooled_wilson95': wilson(pooled_k, pooled_n),
            'seed_ler_range': [min(k / n for _, k, n, _ in lst),
                               max(k / n for _, k, n, _ in lst)]}
    z_32 = ztest(row['d3']['pooled_k'], row['d3']['pooled_n'],
                 row['d2']['pooled_k'], row['d2']['pooled_n'])
    z_34 = ztest(row['d3']['pooled_k'], row['d3']['pooled_n'],
                 row['d4']['pooled_k'], row['d4']['pooled_n'])
    z_24 = ztest(row['d2']['pooled_k'], row['d2']['pooled_n'],
                 row['d4']['pooled_k'], row['d4']['pooled_n'])
    row['z_d3_vs_d2'] = z_32
    row['z_d3_vs_d4'] = z_34
    row['z_d2_vs_d4'] = z_24
    row['ordering_holds'] = bool(z_32 < -1.96 and z_34 < -1.96
                                 and abs(z_24) < 1.96)
    out['per_eta'][eta] = row
    print(f"eta={eta}: " + " | ".join(
        f"{a} {row[a]['pooled_ler']:.3e} ({row[a]['pooled_k']}/{row[a]['pooled_n']})"
        for a in ('d2', 'd3', 'd4')) +
        f"  z(d3<d2)={z_32:.2f} z(d3<d4)={z_34:.2f} z(d2~d4)={z_24:.2f}"
        f"  ordering={'HOLDS' if row['ordering_holds'] else 'NOT SIG'}")

json.dump(out, open(os.path.join(HERE, 't4_verdict.json'), 'w'), indent=2)
print('[t4 verdict] written')
