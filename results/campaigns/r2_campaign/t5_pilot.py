#!/usr/bin/env python3
"""Weight-2 objective pilot: same-support surrogate (option (ii)).

Objective = weighted corrected-Meth KL on the hw closure basis (the
standard recipe's loss) + lambda * same-support weight-2 correction term:
off-diagonal + diagonal-uniformity KL terms over products E^dag F where
E, F are weight-2 ops on the SAME qudit pair (dedup, identity-prop
dropped), uniform weights within the new set scaled so its total mass =
lambda * (total base Meth mass). FULL BATCH for the pilot (gradient
quality over budget; the sampler question is production-side).
lambda = 1.0 primary (0.3 fallback if training destabilises).

Arms: n=5 L=2 seeds {100..103}; n=7 d=3 L=L_min(from t1a arm_d3) seeds
{100,101} (skipped with a note if t1a is not complete yet).
Per code: base weighted loss + (ii)-term residual (also evaluated for
the central code A_L2_s100 and the baseline codes at the same setting,
so "trained vs incidental" is a measured comparison), probe battery
w1/w2/w3 + iid 1M @p=0.05, op-EV accounting (full-batch ops/step x steps
vs the baseline 113.5k).
Writes t5_results.json (resumable).
Launch: nohup nice -n 19 python3 t5_pilot.py >> t5.log 2>&1 &
"""
import os, sys, json, math, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import src.jax_backend  # x64 first
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODEDIR = os.path.join(HERE, 't5_codes')
OUT = os.path.join(HERE, 't5_results.json')
T1A = os.path.join(HERE, 't1a_results.json')
D, DIST, STEPS, P, LR, LAM = 3, 3, 3000, 0.05, 0.05, 1.0
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


def same_support_groups(n):
    """Per qudit pair: dedup'd E^dag F products of same-support weight-2
    ops (16 per pair from the 4 hardware singles), identity-prop dropped.
    Returns list of grouped-format dicts."""
    from src.errors import qudit_hardware_error_basis
    from src.errors import _is_proportional_to_identity, _is_in_set
    S = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]
    w2ops = [np.kron(a, b) for a in S for b in S]      # 16 ops on (a,b)
    prods = []
    for E in w2ops:
        for F in w2ops:
            M = E.conj().T @ F
            if _is_proportional_to_identity(M, D * D):
                continue
            if _is_in_set(M, prods, D * D):
                continue
            prods.append(M)
    stack = np.stack(prods)
    groups = []
    for a in range(n):
        for b in range(a + 1, n):
            groups.append({'wires': (a, b), 'matrices': stack.copy()})
    return groups


def build_pilot_loss(n, lam):
    import jax, jax.numpy as jnp
    from src.errors import ErrorModel
    from src.jax_backend import create_jax_encoder, build_varqec_loss_weighted
    from src.meth_weights import compute_meth_group_weights
    conns = [[i, j] for i in range(n) for j in range(i + 1, n)]
    enc, _, ppl = create_jax_encoder(n, D, connections=conns, use_scan=True)
    base = ErrorModel(d=D, n_qudit=n, distance=DIST,
                      closed=True).build_grouped(verbose=False)
    # the loss builders accept any iterable of ints for 'wires'
    aug = same_support_groups(n)
    union = list(base) + aug
    loss_w = build_varqec_loss_weighted(encoder_fn=enc, K=D, d=D, n_qudit=n,
                                        distance=DIST, E_det_grouped=union)
    bw, _ = compute_meth_group_weights(D, n, distance=DIST)
    base_mass = float(sum(np.sum(w) for w in bw))
    n_new = sum(g['matrices'].shape[0] for g in aug)
    unit = lam * base_mass / n_new
    w_train = tuple([jnp.asarray(w, dtype=jnp.float64) for w in bw] +
                    [jnp.full((g['matrices'].shape[0],), unit,
                              dtype=jnp.float64) for g in aug])
    w_base_only = tuple([jnp.asarray(w, dtype=jnp.float64) for w in bw] +
                        [jnp.zeros((g['matrices'].shape[0],),
                                   dtype=jnp.float64) for g in aug])
    w_aug_only = tuple([jnp.zeros((len(w),), dtype=jnp.float64)
                        for w in bw] +
                       [jnp.ones((g['matrices'].shape[0],),
                                 dtype=jnp.float64) for g in aug])
    ops_per_step = sum(len(w) for w in w_train)
    return (enc, ppl, conns, loss_w, w_train, w_base_only, w_aug_only,
            ops_per_step)


def train_one(loss_pack, L, seed):
    import jax, optax
    enc, ppl, conns, loss_w, w_train, *_ = loss_pack
    val_grad = jax.jit(jax.value_and_grad(lambda th: loss_w(th, w_train)))
    init_key, _ = jax.random.split(jax.random.PRNGKey(seed))
    theta = jax.random.uniform(init_key, (L, ppl), minval=0.0,
                               maxval=2 * np.pi)
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


def battery(cs, n):
    from src.simulation import (simulate_ler_with_correction_factored,
                                _apply_factored_op)
    from src.errors import qudit_hardware_error_basis, make_hardware_noise_fn
    S = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]
    def wk(w):
        def fn(state, rng):
            qs = rng.choice(n, size=w, replace=False)
            s = state
            for q in qs:
                s = _apply_factored_op(s, S[rng.integers(len(S))], int(q),
                                       n, D)
            nrm = np.linalg.norm(s)
            return s / nrm if nrm > 1e-15 else s
        return fn
    out = {}
    for w, sd, shots in ((1, 1001, 200_000), (2, 1002, 100_000),
                         (3, 1003, 100_000)):
        rr = simulate_ler_with_correction_factored(
            cs, wk(w), S, n, D, n_shots=shots, seed=sd)
        k = int(round(float(rr['logical_error_rate']) * shots))
        out[f'ler_w{w}'] = {'k': k, 'n': shots,
                            'ler': k / shots, 'wilson95': wilson(k, shots)}
    rr = simulate_ler_with_correction_factored(
        cs, make_hardware_noise_fn(D, n, P), S, n, D,
        n_shots=1_000_000, seed=2000)
    k = int(round(float(rr['logical_error_rate']) * 1_000_000))
    out['ler_iid_p05'] = {'k': k, 'n': 1_000_000, 'ler': k / 1_000_000,
                          'wilson95': wilson(k, 1_000_000)}
    return out


def eval_residuals(loss_pack, theta):
    import jax.numpy as jnp
    _, _, _, loss_w, _, w_base, w_aug, _ = loss_pack
    th = jnp.asarray(theta)
    return {'base_weighted_loss': float(loss_w(th, w_base)),
            'same_support_residual': float(loss_w(th, w_aug))}


def theta_from_npz(npz):
    dat = np.load(npz, allow_pickle=True)
    return np.asarray(dat['params'])


def run_arm(res, arm, n, L, seeds):
    from src.loss import save_varqec_result
    from src.decoders._common import encoder_forward
    pack = build_pilot_loss(n, LAM)
    conns = pack[2]
    ops = pack[7]
    for seed in seeds:
        key = f'{arm}_s{seed}'
        if key in res['units']:
            continue
        t0 = time.time()
        fl, theta = train_one(pack, L, seed)
        npz = os.path.join(CODEDIR, key + '.npz')
        save_varqec_result(
            npz, np.array(theta), [fl], f'native_d{D}', DIST, L,
            metadata={'K': D, 'n_qudit': n, 'd': D, 'seed': seed,
                      'objective': f'meth_weighted+{LAM}*same_support_w2',
                      'entangler': 'ms', 'connectivity': 'all-to-all',
                      'backend': 'jax',
                      'connections': np.asarray(conns, dtype=int)})
        row = {'arm': arm, 'n': n, 'L': L, 'seed': seed, 'npz': npz,
               'lambda': LAM, 'train_loss_union': fl,
               'ops_per_step_full_batch': ops,
               'op_evs': ops * STEPS,
               'train_wall_s': round(time.time() - t0, 1)}
        row.update(eval_residuals(pack, theta))
        cs = np.asarray(encoder_forward(npz, conns)[0])
        row.update(battery(cs, n))
        row['wall_s'] = round(time.time() - t0, 1)
        res['units'][key] = row
        json.dump(res, open(OUT, 'w'), indent=2)
        print(f"[t5] {key}: union={fl:.3e} "
              f"ss_resid={row['same_support_residual']:.3e} "
              f"w2={row['ler_w2']['k']} "
              f"iid={row['ler_iid_p05']['ler']:.3e} [{row['wall_s']}s]",
              flush=True)
    # baseline residual comparison rows (evaluated once per arm)
    bkey = f'{arm}_baselines'
    if bkey not in res:
        base_rows = {}
        if arm == 'n5_L2':
            for tag, npz in [('headline_A_L2_s100', HEADLINE_NPZ)] + [
                    (f'A_L2_s{s}',
                     HEADLINE_NPZ.replace('s100', f's{s}'))
                    for s in (101, 102, 103)]:
                if os.path.exists(npz):
                    base_rows[tag] = eval_residuals(pack,
                                                    theta_from_npz(npz))
        elif os.path.exists(T1A):
            a = json.load(open(T1A))
            for k, u in a['units'].items():
                if u['d'] == 3 and u['L'] == L:
                    base_rows[k] = eval_residuals(pack,
                                                  theta_from_npz(u['npz']))
        res[bkey] = base_rows
        json.dump(res, open(OUT, 'w'), indent=2)


def main():
    os.makedirs(CODEDIR, exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {
        'protocol': 'Weight-2 objective pilot (option '
                    '(ii)): full-batch union objective, lambda=1.0, '
                    'uniform-within-new-set weights scaled to '
                    'lambda*base mass.', 'units': {}}
    run_arm(res, 'n5_L2', 5, 2, [100, 101, 102, 103])
    if os.path.exists(T1A):
        a = json.load(open(T1A))
        if a.get('arm_d3', {}).get('complete') and \
                a['arm_d3']['minimal_viable_L'] is not None:
            run_arm(res, 'n7_d3', 7, a['arm_d3']['minimal_viable_L'],
                    [100, 101])
        else:
            res['n7_note'] = 't1a arm_d3 incomplete at run time'
            json.dump(res, open(OUT, 'w'), indent=2)
    print('[t5] DONE', flush=True)


if __name__ == '__main__':
    main()
