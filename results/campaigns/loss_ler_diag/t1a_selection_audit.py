#!/usr/bin/env python3
"""Selection audit of the depth-sweep codes + hardware-basis loss of the
existing codes (residual-vs-LER diagnostics). Writes t1a_results.json.

x64 footgun: import src.jax_backend BEFORE any jnp array.
"""
import os, sys, json
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import src.jax_backend  # flips jax_enable_x64 FIRST
import numpy as np
import jax.numpy as jnp
from src.jax_backend import create_jax_loss_vmap_weighted_v2
from src.errors import ErrorModel
from src.seed_race import build_race
from src.decoders._common import encoder_forward
from src.catalog import load_code

D, N, DIST, K = 3, 5, 3, 3


def make_state_loss(basis):
    model = ErrorModel(d=D, n_qudit=N, distance=DIST, closed=True, basis=basis)
    grouped = model.build_grouped(verbose=False)
    core = create_jax_loss_vmap_weighted_v2(grouped, K, D, N, DIST)
    ones = tuple(jnp.ones((g['matrices'].shape[0],), dtype=jnp.float64)
                 for g in grouped)
    n_ops = sum(int(g['matrices'].shape[0]) for g in grouped)
    return (lambda cs: float(core(jnp.asarray(cs), ones))), n_ops


loss_full, nops_full = make_state_loss('full')
loss_hw, nops_hw = make_state_loss(None)
print(f"E_det ops: full={nops_full}  hw={nops_hw}")

rows = []

NPZ_CODES = [
    ('L16', 'results/campaigns/headline_p05_Lsweep/n5_L16.npz'),
    ('L20', 'results/campaigns/headline_p05_Lsweep/n5_L20.npz'),
    ('L24', 'results/campaigns/headline_p05_Lsweep/n5_L24.npz'),
    ('L28', 'results/campaigns/headline_p05_Lsweep/n5_L28.npz'),
    ('stageH', 'results/campaigns/stage_h/race_fullbatch_d3_n5_dist3.npz'),
]

for name, path in NPZ_CODES:
    data = np.load(path, allow_pickle=True)
    L = int(data['n_layers'])
    conns = [list(map(int, c)) for c in data['connections']]
    stored = float(data['certified_full_loss'])
    a2a = [[i, j] for i in range(N) for j in range(i + 1, N)]
    conn_ok = (conns == a2a)
    params_ok = (data['params'].shape[0] == L)

    # theta-loss via the SAME machinery training used
    _, full_loss_fn, _, ppl, bconns, _, _ = build_race(
        D, N, DIST, L, noise='full', entangler='ms', connectivity='all-to-all')
    theta_loss = float(full_loss_fn(jnp.asarray(data['params'])))

    # states via the SAME rebuild the LER benchmark used
    cs, _, _, _ = encoder_forward(path, conns)
    ortho = float(np.max(np.abs(cs @ cs.conj().T - np.eye(K))))
    sfull = loss_full(cs)
    shw = loss_hw(cs)
    rows.append(dict(name=name, L=L, stored=stored, theta_loss=theta_loss,
                     states_full=sfull, hw_loss=shw, ortho=ortho,
                     conn_ok=bool(conn_ok), params_ok=bool(params_ok),
                     ppl_match=bool(ppl == data['params'].shape[1])))
    print(f"{name}: L={L} conn_ok={conn_ok} params_ok={params_ok} "
          f"ppl={ppl}=={data['params'].shape[1]} ortho_dev={ortho:.2e}\n"
          f"   stored={stored:.6e} theta_recomputed={theta_loss:.6e} "
          f"rel={(abs(theta_loss-stored)/stored):.2e}\n"
          f"   states_full={sfull:.6e} hw_basis={shw:.6e}", flush=True)

for name, cat in [('published_L10_ring', 'qutrit_d3'),
                  ('[[5,1,3]]', 'five_qudit_d3')]:
    cs = np.asarray(load_code(cat)['code_states'])
    ortho = float(np.max(np.abs(cs @ cs.conj().T - np.eye(K))))
    sfull = loss_full(cs)
    shw = loss_hw(cs)
    rows.append(dict(name=name, stored=None, states_full=sfull,
                     hw_loss=shw, ortho=ortho))
    print(f"{name}: ortho_dev={ortho:.2e} states_full={sfull:.6e} "
          f"hw_basis={shw:.6e}", flush=True)

out = os.path.join(os.environ.get('SCRATCH', '.'), 't1a_results.json')
json.dump(rows, open(out, 'w'), indent=2)
print('saved', out)
