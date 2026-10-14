#!/usr/bin/env python3
"""Decoder-saturation check + weight-conditioned LER decomposition
(residual-vs-LER diagnostics, Table IV rows).

A. LER conditioned on exact error weight w (1,2,3), lookup decoder.
B. p-sweep p in {0.01,0.02,0.05,0.10,0.15}, lookup decoder, >=300k-2M shots.
C. reconstruction: LER(p) ~= sum_w Binom(5,p,w)*LER(w) vs measured.
D. weighted-MAP decoder with i.i.d.-channel priors and weight<=2 corrections
   at p=0.05 (does a stronger decoder un-floor / re-order the codes?).

Writes t1b_results.json.
x64 footgun: import src.jax_backend before any jnp array creation.
"""
import os, sys, json, math, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import src.jax_backend  # x64 first
import numpy as np
from src.errors import (ErrorModel, qudit_hardware_error_basis,
                        make_hardware_noise_fn)
from src.simulation import (simulate_ler_with_correction_factored,
                            _apply_factored_op)
from src.decoders._common import encoder_forward
from src.decoders.weighted_map import (build_weighted_correction_set,
                                       simulate_ler_with_weighted_map)
from src.catalog import load_code

D, N, K = 3, 5, 3
SCRATCH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRATCH, 't1b_results.json')
SINGLES = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]
NE = len(SINGLES)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    dn = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / dn
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / dn
    return (max(0.0, c - h), c + h)


def get_codes():
    codes = {}
    for name, path in [
            ('L16', 'results/campaigns/headline_p05_Lsweep/n5_L16.npz'),
            ('L20', 'results/campaigns/headline_p05_Lsweep/n5_L20.npz'),
            ('L24', 'results/campaigns/headline_p05_Lsweep/n5_L24.npz'),
            ('L28', 'results/campaigns/headline_p05_Lsweep/n5_L28.npz'),
            ('stageH', 'results/campaigns/stage_h/race_fullbatch_d3_n5_dist3.npz')]:
        data = np.load(path, allow_pickle=True)
        conns = [list(map(int, c)) for c in data['connections']]
        cs, *_ = encoder_forward(path, conns)
        codes[name] = np.asarray(cs)
    codes['published'] = np.asarray(load_code('qutrit_d3')['code_states'])
    codes['513'] = np.asarray(load_code('five_qudit_d3')['code_states'])
    return codes


def make_fixed_weight_noise_fn(w):
    def noise_fn(state, rng):
        qs = rng.choice(N, size=w, replace=False)
        s = state
        for q in qs:
            E = SINGLES[rng.integers(NE)]
            s = _apply_factored_op(s, E, int(q), N, D)
        nrm = np.linalg.norm(s)
        return s / nrm if nrm > 1e-15 else s
    return noise_fn


def binom_pw(p, w):
    return math.comb(N, w) * p**w * (1 - p)**(N - w)


def iid_prior_weights(grouped, p):
    """Flat prior per op in grouped order under the i.i.d. hardware channel.
    Singles: (p/4)(1-p)^4 on the 4 channel ops, 0 for closure products;
    weight-2 pairs: (p/4)^2(1-p)^3; identity sentinel: (1-p)^5 per row."""
    pri = []
    for g in grouped:
        wl = len(tuple(g['wires']))
        mats = np.asarray(g['matrices'])
        for e in range(mats.shape[0]):
            if wl == 0:
                pri.append((1 - p)**N)
            elif wl == 1:
                is_channel = any(np.allclose(mats[e], S, atol=1e-12)
                                 for S in SINGLES)
                pri.append((p / NE) * (1 - p)**(N - 1) if is_channel else 0.0)
            elif wl == 2:
                pri.append((p / NE)**2 * (1 - p)**(N - 2))
            else:
                pri.append(0.0)
    return np.asarray(pri)


def main():
    codes = get_codes()
    res = {'weight_conditioned': {}, 'p_sweep': {}, 'reconstruction': {},
           'map_p05': {}}

    # A. weight-conditioned LER
    for name, cs in codes.items():
        res['weight_conditioned'][name] = {}
        for w, shots in [(1, 200_000), (2, 200_000), (3, 100_000)]:
            t = time.time()
            r = simulate_ler_with_correction_factored(
                cs, make_fixed_weight_noise_fn(w), SINGLES, N, D,
                n_shots=shots, seed=1000 + w)
            ler = float(r['logical_error_rate'])
            kerr = int(round(ler * shots))
            res['weight_conditioned'][name][f'w{w}'] = {
                'ler': ler, 'k': kerr, 'n': shots, 'wilson95': wilson(kerr, shots)}
            print(f"A {name} w={w}: LER|w={ler:.4e} ({kerr}/{shots}) "
                  f"[{time.time()-t:.0f}s]", flush=True)
        json.dump(res, open(OUT, 'w'), indent=2)

    # B. p-sweep, lookup decoder
    PGRID = [(0.01, 2_000_000), (0.02, 1_000_000), (0.05, 1_000_000),
             (0.10, 300_000), (0.15, 300_000)]
    for name, cs in codes.items():
        res['p_sweep'][name] = {}
        for p, shots in PGRID:
            t = time.time()
            r = simulate_ler_with_correction_factored(
                cs, make_hardware_noise_fn(D, N, p), SINGLES, N, D,
                n_shots=shots, seed=2000)
            ler = float(r['logical_error_rate'])
            kerr = int(round(ler * shots))
            res['p_sweep'][name][str(p)] = {
                'ler': ler, 'k': kerr, 'n': shots, 'wilson95': wilson(kerr, shots)}
            print(f"B {name} p={p}: LER={ler:.4e} ({kerr}/{shots}) "
                  f"[{time.time()-t:.0f}s]", flush=True)
        json.dump(res, open(OUT, 'w'), indent=2)

    # C. reconstruction check
    for name in codes:
        wc = res['weight_conditioned'][name]
        res['reconstruction'][name] = {}
        for p, _ in PGRID:
            pred = sum(binom_pw(p, w) * wc[f'w{w}']['ler'] for w in (1, 2, 3))
            meas = res['p_sweep'][name][str(p)]['ler']
            res['reconstruction'][name][str(p)] = {
                'predicted': pred, 'measured': meas,
                'ratio': meas / pred if pred > 0 else None}
    json.dump(res, open(OUT, 'w'), indent=2)
    print("C reconstruction done", flush=True)

    # D. weighted-MAP with iid priors, weight<=2 corrections, p=0.05
    model = ErrorModel(d=D, n_qudit=N, distance=3, closed=True, basis=None)
    grouped = model.build_grouped(verbose=False)
    pri = iid_prior_weights(grouped, 0.05)
    print(f"D: {int(np.sum(pri > 0))} nonzero-prior corrections "
          f"of {len(pri)} ops", flush=True)
    corr = build_weighted_correction_set(grouped, pri, N, D, max_weight=2)
    for name, cs in codes.items():
        t = time.time()
        shots = 200_000
        outcomes = simulate_ler_with_weighted_map(
            cs, make_hardware_noise_fn(D, N, 0.05), corr, N, D,
            n_shots=shots, seed=3000)
        kerr = int(outcomes.sum())
        res['map_p05'][name] = {'ler': kerr / shots, 'k': kerr, 'n': shots,
                                'wilson95': wilson(kerr, shots)}
        print(f"D {name} MAP p=0.05: LER={kerr/shots:.4e} ({kerr}/{shots}) "
              f"[{time.time()-t:.0f}s]", flush=True)
        json.dump(res, open(OUT, 'w'), indent=2)

    print("DONE", flush=True)


if __name__ == '__main__':
    main()
