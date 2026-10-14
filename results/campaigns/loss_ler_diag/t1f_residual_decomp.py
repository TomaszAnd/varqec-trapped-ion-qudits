#!/usr/bin/env python3
"""Decompose the hardware-basis KL residual per E_det op class:
weight-1 singles, same-qudit closure products, cross-qudit weight-2 pairs.
Explains how a code with residual 0.1-0.4 still corrects weight-1 exactly.
Prints the decomposition (residual-vs-LER diagnostics).
"""
import os, sys, json
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import src.jax_backend  # x64 first
import numpy as np
from src.errors import ErrorModel
from src.decoders._common import encoder_forward
from src.catalog import load_code

D, N, K = 3, 5, 3

model = ErrorModel(d=D, n_qudit=N, distance=3, closed=True, basis=None)
grouped = model.build_grouped(verbose=False)


def states(path):
    data = np.load(path, allow_pickle=True)
    conns = [list(map(int, c)) for c in data['connections']]
    cs, *_ = encoder_forward(path, conns)
    return np.asarray(cs)


CODES = {
    'L24': states('results/campaigns/headline_p05_Lsweep/n5_L24.npz'),
    'stageH': states('results/campaigns/stage_h/race_fullbatch_d3_n5_dist3.npz'),
    'published': np.asarray(load_code('qutrit_d3')['code_states']),
    'hwL16': states('results/campaigns/loss_ler_diag/n5_hw_L16.npz'),
}

from src.decoders._common import apply_local_np

for name, cs in CODES.items():
    acc = {'w1_single': 0.0, 'w1_closure': 0.0, 'w2_pair': 0.0}
    nsingle = len(model.single_errors)
    for g in grouped:
        wires = tuple(int(w) for w in g['wires'])
        inv = tuple(int(p) for p in g['inverse_perm'])
        mats = np.asarray(g['matrices'])
        if len(wires) == 0:
            continue
        for e in range(mats.shape[0]):
            Ecs = np.stack([apply_local_np(cs[k], mats[e], wires, inv, D, N)
                            for k in range(K)])
            ov = cs.conj() @ Ecs.T  # ov[i,j] = <psi_i|E|psi_j>
            off = float(sum(abs(ov[i, j])**2
                            for i in range(K) for j in range(i + 1, K)))
            diag = np.diag(ov)
            var = float(np.real(np.var(diag)))
            r = off + (K / 4) * var
            if len(wires) == 2:
                acc['w2_pair'] += r
            else:
                is_single = any(np.allclose(mats[e], S, atol=1e-12)
                                for S in model.single_errors)
                acc['w1_single' if is_single else 'w1_closure'] += r
    tot = sum(acc.values())
    print(f"{name}: total={tot:.6e}  singles={acc['w1_single']:.3e}  "
          f"closure={acc['w1_closure']:.3e}  w2_pairs={acc['w2_pair']:.3e}",
          flush=True)
