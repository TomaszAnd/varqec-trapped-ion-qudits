#!/usr/bin/env python3
"""Continuous weight-1 probe: mean infidelity after correction,
conditioned on exactly one qudit erring. Maps KL residual -> weight-1 damage
(LER|w1 is 0/200k for every code, so a rate can't resolve the threshold
question; infidelity can). Also includes the two hw-retrained codes of
t1c_hw_retrain.py. Writes t1e_results.json.
"""
import os, sys, json
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import src.jax_backend  # x64 first
import numpy as np
from src.errors import qudit_hardware_error_basis
from src.simulation import (simulate_ler_with_correction_factored,
                            _apply_factored_op)
from src.decoders._common import encoder_forward
from src.catalog import load_code

D, N = 3, 5
SCRATCH = os.path.dirname(os.path.abspath(__file__))
SINGLES = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]


def w1_noise(state, rng):
    q = int(rng.integers(N))
    s = _apply_factored_op(state, SINGLES[rng.integers(len(SINGLES))], q, N, D)
    nrm = np.linalg.norm(s)
    return s / nrm if nrm > 1e-15 else s


def states(path):
    data = np.load(path, allow_pickle=True)
    conns = [list(map(int, c)) for c in data['connections']]
    cs, *_ = encoder_forward(path, conns)
    return np.asarray(cs)


CODES = {
    'L16': states('results/campaigns/headline_p05_Lsweep/n5_L16.npz'),
    'L20': states('results/campaigns/headline_p05_Lsweep/n5_L20.npz'),
    'L24': states('results/campaigns/headline_p05_Lsweep/n5_L24.npz'),
    'L28': states('results/campaigns/headline_p05_Lsweep/n5_L28.npz'),
    'stageH': states('results/campaigns/stage_h/race_fullbatch_d3_n5_dist3.npz'),
    'published': np.asarray(load_code('qutrit_d3')['code_states']),
    '513': np.asarray(load_code('five_qudit_d3')['code_states']),
}
for L in (16, 24):
    p = f'results/campaigns/loss_ler_diag/n5_hw_L{L}.npz'
    if os.path.exists(p):
        CODES[f'hwL{L}'] = states(p)

def wk_noise(w):
    def fn(state, rng):
        qs = rng.choice(N, size=w, replace=False)
        s = state
        for q in qs:
            s = _apply_factored_op(s, SINGLES[rng.integers(len(SINGLES))],
                                   int(q), N, D)
        nrm = np.linalg.norm(s)
        return s / nrm if nrm > 1e-15 else s
    return fn


out = {}
for name, cs in CODES.items():
    r = simulate_ler_with_correction_factored(
        cs, w1_noise, SINGLES, N, D, n_shots=100_000, seed=777)
    out[name] = {'mean_infid_w1': 1.0 - float(r['mean_fidelity']),
                 'ler_w1': float(r['logical_error_rate'])}
    print(f"{name}: mean 1-F |w1 = {out[name]['mean_infid_w1']:.3e}  "
          f"LER|w1={out[name]['ler_w1']:.1e}", flush=True)

# w2/w3 decomposition for the two hw-retrained codes (missing from t1b part A)
for name in ('hwL16', 'hwL24'):
    if name not in CODES:
        continue
    for w, shots in ((2, 200_000), (3, 100_000)):
        r = simulate_ler_with_correction_factored(
            CODES[name], wk_noise(w), SINGLES, N, D,
            n_shots=shots, seed=1000 + w)
        out[name][f'ler_w{w}'] = float(r['logical_error_rate'])
        print(f"{name} w={w}: LER|w={out[name][f'ler_w{w}']:.4e}", flush=True)

json.dump(out, open(os.path.join(SCRATCH, 't1e_results.json'), 'w'), indent=2)
print('DONE')
