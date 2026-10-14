#!/usr/bin/env python3
"""Conditioned weight-1 probe on the dist-2 panel code
(fig6_panels/d3_n5_dist2_16L.npz), closing the unmeasured premise behind the
claim "operational weight-1 correction without the dist-3 certificate":
LER|w=1 at 200k conditioned shots + mean infidelity|w=1 at 100k
(t1b/t1e machinery, same seeds/instrument). LER|w=2 (200k) added for the
beyond-distance context. NOTE: the dist-2 objective omits the diagonal
codeword-independence term (create_jax_loss_vmap_weighted_v2 gates Term 2 on
distance >= 3), so exact off-diagonal detection does NOT imply the full KL
weight-1 condition — this probe measures whether that matters operationally.
Writes t1h_results.json.

x64 footgun: import src.jax_backend before any jnp array creation.
"""
import os, sys, json, math
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import src.jax_backend  # x64 first
import numpy as np
from src.errors import qudit_hardware_error_basis
from src.simulation import (simulate_ler_with_correction_factored,
                            _apply_factored_op)
from src.decoders._common import encoder_forward

D, N = 3, 5
SCRATCH = os.path.dirname(os.path.abspath(__file__))
SINGLES = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]
NPZ = 'results/campaigns/fig6_panels/d3_n5_dist2_16L.npz'


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    dn = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / dn
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / dn
    return (max(0.0, c - h), c + h)


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


conns = [[i, j] for i in range(N) for j in range(i + 1, N)]
cs, *_ = encoder_forward(NPZ, conns)
cs = np.asarray(cs)

out = {'code': NPZ, 'protocol': 'CORRECTION (lookup decoder, hw singles), '
       'NOT Appendix-G dist-2 detection-and-postselect'}

# LER|w=1, 200k conditioned shots (t1b seed convention 1000+w)
r = simulate_ler_with_correction_factored(
    cs, wk_noise(1), SINGLES, N, D, n_shots=200_000, seed=1001)
k1 = int(round(float(r['logical_error_rate']) * 200_000))
out['w1'] = {'ler': float(r['logical_error_rate']), 'k': k1, 'n': 200_000,
             'wilson95': wilson(k1, 200_000)}
print(f"LER|w=1 = {out['w1']['ler']:.4e} ({k1}/200k)", flush=True)

# mean infidelity|w=1, 100k shots (t1e seed 777)
r = simulate_ler_with_correction_factored(
    cs, wk_noise(1), SINGLES, N, D, n_shots=100_000, seed=777)
out['w1_infidelity'] = {'mean_infid': 1.0 - float(r['mean_fidelity']),
                        'ler': float(r['logical_error_rate']), 'n': 100_000}
print(f"mean 1-F |w=1 = {out['w1_infidelity']['mean_infid']:.3e}  "
      f"(LER {out['w1_infidelity']['ler']:.2e})", flush=True)

# LER|w=2 for the beyond-distance context (t1b seed 1002)
r = simulate_ler_with_correction_factored(
    cs, wk_noise(2), SINGLES, N, D, n_shots=200_000, seed=1002)
k2 = int(round(float(r['logical_error_rate']) * 200_000))
out['w2'] = {'ler': float(r['logical_error_rate']), 'k': k2, 'n': 200_000,
             'wilson95': wilson(k2, 200_000)}
print(f"LER|w=2 = {out['w2']['ler']:.4e} ({k2}/200k)", flush=True)

json.dump(out, open(os.path.join(SCRATCH, 't1h_results.json'), 'w'), indent=2)
print('DONE', flush=True)
