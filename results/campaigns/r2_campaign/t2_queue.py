#!/usr/bin/env python3
"""K study and q-panel completions.

(a) K panel at the minimal-L recipe (L=2, weighted corrected-Meth KL on
the hw closure, FULL batch, K-explicit):
  - K=1 (((5,1,3))_3 and ((7,1,3))_3): the KL loss is IDENTICALLY ZERO at
    K=1 (Term 1 has no off-diagonal pairs; Term 2's codeword-independence
    is vacuous) — VERIFIED numerically at run time and recorded. Any
    state satisfies the K=1 KL conditions, so the row is the seed-100
    init-theta state, measured with the standing battery; reported as the
    degenerate "stores nothing" reference point of the K panel.
  - K=2: 2 seeds {100,101}, n=5 and n=7, L=2, 3000 steps, K-explicit
    weighted full-batch (fig6_panel_train lineage + Meth weights).
  - K=3 rows: pointers to the central code (n=5, r1_campaign) and the
    t1a d3 L=2 best-by-probe (n=7) — same recipe, no retrain.
(b) q panel — ((5,3,3))_5 resolution: measure the q=5 shot cost (20k
shots), then escalate cells to >=30 errors or the 1e7 cap, in DESCENDING
p order over the committed fig6 grid {0.02,0.03,0.05,0.08,0.12,0.15}
(seed 2240+idx, the committed instrument seeds); if a cell hits the cap
with k=0, lower-p cells inherit its Wilson UB by LER monotonicity in p
(stated, not re-simulated). Writes t2_results.json (resumable). Feeds
the K study of the paper.
"""
import os, sys, json, math, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import src.jax_backend  # x64 first
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODEDIR = os.path.join(HERE, 't2_codes')
OUT = os.path.join(HERE, 't2_results.json')
T1A = os.path.join(HERE, 't1a_results.json')
STEPS, LR, P = 3000, 0.05, 0.05
D5NPZ = os.path.join(REPO, 'results/campaigns/fig6_panels',
                     'd5K3_n5_dist3_16L.npz')
Q5_GRID = [0.02, 0.03, 0.05, 0.08, 0.12, 0.15]
Q5_SEED0, Q5_CAP, Q5_MINERR = 2240, 10_000_000, 30
HEADLINE_NPZ = os.path.join(REPO, 'results/campaigns/r1_campaign',
                            't1_codes', 'A_L2_s100.npz')


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    dn = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / dn
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / dn
    return (max(0.0, c - h), c + h)


def battery(cs, n, d):
    from src.simulation import (simulate_ler_with_correction_factored,
                                _apply_factored_op)
    from src.errors import qudit_hardware_error_basis, make_hardware_noise_fn
    S = [np.asarray(E, complex) for E in qudit_hardware_error_basis(d)]
    def wk(w):
        def fn(state, rng):
            qs = rng.choice(n, size=w, replace=False)
            s = state
            for q in qs:
                s = _apply_factored_op(s, S[rng.integers(len(S))], int(q),
                                       n, d)
            nrm = np.linalg.norm(s)
            return s / nrm if nrm > 1e-15 else s
        return fn
    out = {}
    for w, sd, shots in ((1, 1001, 200_000), (2, 1002, 100_000),
                         (3, 1003, 100_000)):
        rr = simulate_ler_with_correction_factored(
            cs, wk(w), S, n, d, n_shots=shots, seed=sd)
        k = int(round(float(rr['logical_error_rate']) * shots))
        out[f'ler_w{w}'] = {'k': k, 'n': shots, 'ler': k / shots,
                            'wilson95': wilson(k, shots)}
    rr = simulate_ler_with_correction_factored(
        cs, make_hardware_noise_fn(d, n, P), S, n, d,
        n_shots=1_000_000, seed=2000)
    k = int(round(float(rr['logical_error_rate']) * 1_000_000))
    out['ler_iid_p05'] = {'k': k, 'n': 1_000_000, 'ler': k / 1_000_000,
                          'wilson95': wilson(k, 1_000_000)}
    return out


def kexp_pack(n, K):
    import jax, jax.numpy as jnp
    from src.errors import ErrorModel
    from src.jax_backend import create_jax_encoder, build_varqec_loss_weighted
    from src.meth_weights import compute_meth_group_weights
    conns = [[i, j] for i in range(n) for j in range(i + 1, n)]
    enc, _, ppl = create_jax_encoder(n, 3, connections=conns, use_scan=True)
    grouped = ErrorModel(d=3, n_qudit=n, distance=3,
                         closed=True).build_grouped(verbose=False)
    loss_w = build_varqec_loss_weighted(encoder_fn=enc, K=K, d=3, n_qudit=n,
                                        distance=3, E_det_grouped=grouped)
    gw, _ = compute_meth_group_weights(3, n, distance=3)
    w = tuple(jnp.asarray(x, dtype=jnp.float64) for x in gw)
    ops = sum(len(x) for x in gw)
    return conns, ppl, loss_w, w, ops


def init_theta(L, ppl, seed):
    import jax
    init_key, _ = jax.random.split(jax.random.PRNGKey(seed))
    return jax.random.uniform(init_key, (L, ppl), minval=0.0,
                              maxval=2 * np.pi)


def train_kexp(pack, L, seed):
    import jax, optax
    conns, ppl, loss_w, w, ops = pack
    val_grad = jax.jit(jax.value_and_grad(lambda th: loss_w(th, w)))
    theta = init_theta(L, ppl, seed)
    opt = optax.inject_hyperparams(optax.adam)(learning_rate=LR)
    st = opt.init(theta)
    stage = 0
    best = (np.inf, theta)
    for step in range(STEPS):
        lv, g = val_grad(theta)
        lvf = float(lv)
        if lvf < best[0]:
            best = (lvf, theta)
        upd, st = opt.update(g, st, theta)
        theta = optax.apply_updates(theta, upd)
        if stage == 0 and lvf < 0.5:
            st.hyperparams['learning_rate'] = LR / 5; stage = 1
        if stage <= 1 and lvf < 1e-3:
            st.hyperparams['learning_rate'] = LR / 50; stage = 2
    return best


def part_a(res):
    from src.loss import save_varqec_result
    from src.decoders._common import encoder_forward
    for n in (5, 7):
        # ---- K=1 degeneracy check + row ----
        key = f'K1_n{n}'
        if key not in res['units']:
            t0 = time.time()
            pack = kexp_pack(n, 1)
            conns, ppl, loss_w, w, ops = pack
            import jax.numpy as jnp
            vals = [float(loss_w(jnp.asarray(init_theta(2, ppl, s)), w))
                    for s in (100, 101, 102)]
            theta = np.asarray(init_theta(2, ppl, 100))
            npz = os.path.join(CODEDIR, key + '.npz')
            save_varqec_result(npz, theta, [vals[0]], 'native_d3', 3, 2,
                               metadata={'K': 1, 'n_qudit': n, 'd': 3,
                                         'seed': 100, 'entangler': 'ms',
                                         'connectivity': 'all-to-all',
                                         'backend': 'jax',
                                         'note': 'K=1 KL loss degenerate',
                                         'connections': np.asarray(
                                             conns, dtype=int)})
            row = {'part': 'Ka', 'n': n, 'K': 1, 'L': 2, 'npz': npz,
                   'ms_budget': 2 * len(conns) * 2,
                   'kl_loss_at_3_random_thetas': vals,
                   'kl_degenerate': bool(max(abs(v) for v in vals) < 1e-12),
                   'protocol': 'untrained seed-100 init state (any state '
                               'satisfies the K=1 KL conditions)'}
            cs = np.asarray(encoder_forward(npz, conns)[0])
            row.update(battery(cs, n, 3))
            row['wall_s'] = round(time.time() - t0, 1)
            res['units'][key] = row
            json.dump(res, open(OUT, 'w'), indent=2)
            print(f"[t2] {key}: kl_vals={vals} "
                  f"iid={row['ler_iid_p05']['ler']:.3e}", flush=True)
        # ---- K=2, 2 seeds ----
        pack = None
        for seed in (100, 101):
            key = f'K2_n{n}_L2_s{seed}'
            if key in res['units']:
                continue
            if pack is None:
                pack = kexp_pack(n, 2)
            t0 = time.time()
            fl, theta = train_kexp(pack, 2, seed)
            conns = pack[0]
            npz = os.path.join(CODEDIR, key + '.npz')
            save_varqec_result(npz, np.array(theta), [fl], 'native_d3',
                               3, 2,
                               metadata={'K': 2, 'n_qudit': n, 'd': 3,
                                         'seed': seed, 'entangler': 'ms',
                                         'connectivity': 'all-to-all',
                                         'backend': 'jax',
                                         'connections': np.asarray(
                                             conns, dtype=int)})
            row = {'part': 'Ka', 'n': n, 'K': 2, 'L': 2, 'seed': seed,
                   'npz': npz, 'ms_budget': 2 * len(conns) * 2,
                   'weighted_loss': fl,
                   'op_evs_full_batch': pack[4] * STEPS}
            cs = np.asarray(encoder_forward(npz, conns)[0])
            row.update(battery(cs, n, 3))
            row['wall_s'] = round(time.time() - t0, 1)
            res['units'][key] = row
            json.dump(res, open(OUT, 'w'), indent=2)
            print(f"[t2] {key}: wloss={fl:.3e} w2={row['ler_w2']['k']} "
                  f"iid={row['ler_iid_p05']['ler']:.3e} "
                  f"[{row['wall_s']}s]", flush=True)
    # ---- K=3 pointer rows ----
    if 'K3_pointers' not in res:
        ptr = {'n5': {'npz': HEADLINE_NPZ,
                      'source': 'central code A_L2_s100, r1_campaign (same recipe, '
                                'importance@0.2)'}}
        if os.path.exists(T1A):
            a = json.load(open(T1A))
            units = sorted((u for u in a['units'].values()
                            if u['d'] == 3 and u['L'] == 2),
                           key=lambda u: (u['ler_w2']['k'],
                                          u['ler_w3']['k'], u['seed']))
            if units:
                ptr['n7'] = {'npz': units[0]['npz'],
                             'source': 't1a d3 L=2 best-by-probe'}
        res['K3_pointers'] = ptr
        json.dump(res, open(OUT, 'w'), indent=2)


def part_b(res):
    from src.simulation import simulate_ler_with_correction_factored
    from src.errors import qudit_hardware_error_basis, make_hardware_noise_fn
    from src.decoders._common import encoder_forward
    n, d = 5, 5
    S = [np.asarray(E, complex) for E in qudit_hardware_error_basis(d)]
    conns = [[i, j] for i in range(n) for j in range(i + 1, n)]
    cs = np.asarray(encoder_forward(D5NPZ, conns)[0])
    if 'q5_shot_cost' not in res:
        t0 = time.time()
        simulate_ler_with_correction_factored(
            cs, make_hardware_noise_fn(d, n, 0.05), S, n, d,
            n_shots=20_000, seed=1)
        per_shot = (time.time() - t0) / 20_000
        res['q5_shot_cost'] = {'s_per_shot': per_shot,
                               'projected_cap_cell_h':
                                   per_shot * Q5_CAP / 3600}
        json.dump(res, open(OUT, 'w'), indent=2)
        print(f"[t2] q5 shot cost {per_shot*1e3:.3f} ms/shot -> "
              f"{per_shot*Q5_CAP/3600:.2f} h per cap cell", flush=True)
    res.setdefault('q5_cells', {})
    monotone_ub = None
    for pi in reversed(range(len(Q5_GRID))):   # descending p
        p = Q5_GRID[pi]
        pk = f'{p:g}'
        if pk in res['q5_cells']:
            cell = res['q5_cells'][pk]
            if cell.get('k') == 0 and cell.get('shots', 0) >= Q5_CAP:
                monotone_ub = cell
            continue
        if monotone_ub is not None:
            res['q5_cells'][pk] = {
                'p': p, 'inherited_from_p': monotone_ub['p'],
                'wilson95_ub': monotone_ub['wilson95'][1],
                'note': 'k=0 at cap at a HIGHER p; bound inherited by '
                        'LER monotonicity in p (stated assumption)'}
            json.dump(res, open(OUT, 'w'), indent=2)
            print(f"[t2] q5 p={p}: inherited UB from p="
                  f"{monotone_ub['p']}", flush=True)
            continue
        t0 = time.time()
        shots = Q5_CAP  # committed data: 0/3M everywhere -> start at cap
        rr = simulate_ler_with_correction_factored(
            cs, make_hardware_noise_fn(d, n, p), S, n, d,
            n_shots=shots, seed=Q5_SEED0 + pi)
        k = int(round(rr['logical_error_rate'] * shots))
        cell = {'p': p, 'shots': shots, 'k': k, 'ler': k / shots,
                'wilson95': wilson(k, shots),
                'resolved': bool(k >= Q5_MINERR),
                'below_detection': bool(k == 0),
                'wall_s': round(time.time() - t0, 1)}
        res['q5_cells'][pk] = cell
        json.dump(res, open(OUT, 'w'), indent=2)
        print(f"[t2] q5 p={p}: k={k}/{shots} [{cell['wall_s']}s]",
              flush=True)
        if k == 0:
            monotone_ub = cell
    print('[t2] DONE', flush=True)


def main():
    os.makedirs(CODEDIR, exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {
        'protocol': 'K study: K panel minimal-L '
                    'K-explicit weighted; q5 escalation descending-p '
                    'with monotonicity shortcut.', 'units': {}}
    part_a(res)
    part_b(res)
    print('[t2] QUEUE DONE', flush=True)


if __name__ == '__main__':
    main()
