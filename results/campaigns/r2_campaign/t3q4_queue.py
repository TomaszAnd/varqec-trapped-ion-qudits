#!/usr/bin/env python3
"""q=4 depth arm: ((5,3,3))_4, L in {2,4,8,16}, 2 seeds.

K=3 != q=4 is not expressible via build_race, so this mirrors the
fig6_panel_train K-explicit trainer EXACTLY (build_varqec_loss_weighted
with K explicit, unweighted full basis full batch, Adam 3-stage LR,
theta init uniform(0,2pi) from split(PRNGKey(seed))[0]) — the protocol of
the committed q-panel code d4K3_n5_dist3_16L — extended to 3000 steps and
run per-seed (no best-of collapse; per-seed floors are needed).

Per code: full-basis loss floor; probes LER|w1 @200k / w2 @100k /
w3 @100k; iid LER @p=0.05 1M (q=4 hardware basis); MS budget = L*|E|*(q-1)
= 30L at n=5 a2a q=4. Writes t3q4_results.json (resumable).
"""
import os, sys, json, math, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import src.jax_backend  # x64 first
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODEDIR = os.path.join(HERE, 't3_codes')
OUT = os.path.join(HERE, 't3q4_results.json')
D, K, N, DIST, STEPS, P, LR = 4, 3, 5, 3, 3000, 0.05, 0.05
LSWEEP = [2, 4, 8, 16]
SEEDS = [0, 1]
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]
MSBUDGET = lambda L: L * len(CONNS) * (D - 1)   # 30L


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    dn = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / dn
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / dn
    return (max(0.0, c - h), c + h)


def train_one(L, seed, loss_pack):
    import jax, optax
    val_grad, loss_full, ppl = loss_pack
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
    return float(loss_full(best[1])), np.array(best[1])


def main():
    import jax, jax.numpy as jnp
    from src.errors import (ErrorModel, qudit_hardware_error_basis,
                            make_hardware_noise_fn)
    from src.jax_backend import create_jax_encoder, build_varqec_loss_weighted
    from src.loss import save_varqec_result
    from src.decoders._common import encoder_forward
    from src.simulation import (simulate_ler_with_correction_factored,
                                _apply_factored_op)
    os.makedirs(CODEDIR, exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {'units': {}}

    singles = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]

    def wk_noise(w):
        def fn(state, rng):
            qs = rng.choice(N, size=w, replace=False)
            s = state
            for q in qs:
                s = _apply_factored_op(s, singles[rng.integers(len(singles))],
                                       int(q), N, D)
            nrm = np.linalg.norm(s)
            return s / nrm if nrm > 1e-15 else s
        return fn

    grouped = ErrorModel(d=D, n_qudit=N, distance=DIST, closed=True,
                         basis="full").build_grouped(verbose=False)
    full_w = tuple(jnp.ones((g['matrices'].shape[0],), dtype=jnp.float64)
                   for g in grouped)
    enc, _, ppl = create_jax_encoder(N, D, connections=CONNS, use_scan=True)
    loss_w = build_varqec_loss_weighted(encoder_fn=enc, K=K, d=D, n_qudit=N,
                                        distance=DIST, E_det_grouped=grouped)
    val_grad = jax.jit(jax.value_and_grad(lambda th: loss_w(th, full_w)))
    loss_full = lambda th: float(loss_w(th, full_w))

    for L in LSWEEP:
        for seed in SEEDS:
            key = f'q4_L{L}_s{seed}'
            if key in res['units']:
                continue
            t0 = time.time()
            fl, theta = train_one(L, seed, (val_grad, loss_full, ppl))
            npz = os.path.join(CODEDIR, key + '.npz')
            save_varqec_result(
                npz, theta, [fl], f'native_d{D}', DIST, L,
                metadata={'K': K, 'n_qudit': N, 'd': D, 'seed': seed,
                          'noise': 'full', 'entangler': 'ms',
                          'connectivity': 'all-to-all', 'backend': 'jax',
                          'connections': np.asarray(CONNS, dtype=int)})
            row = {'L': L, 'seed': seed, 'npz': npz,
                   'ms_budget': MSBUDGET(L), 'full_basis_loss': fl,
                   'train_wall_s': round(time.time() - t0, 1)}
            cs = np.asarray(encoder_forward(npz, CONNS)[0])
            for w, sd, shots in ((1, 1001, 200_000), (2, 1002, 100_000),
                                 (3, 1003, 100_000)):
                rr = simulate_ler_with_correction_factored(
                    cs, wk_noise(w), singles, N, D, n_shots=shots, seed=sd)
                ler = float(rr['logical_error_rate'])
                k = int(round(ler * shots))
                row[f'ler_w{w}'] = {'ler': ler, 'k': k, 'n': shots,
                                    'wilson95': wilson(k, shots)}
            rr = simulate_ler_with_correction_factored(
                cs, make_hardware_noise_fn(D, N, P), singles, N, D,
                n_shots=1_000_000, seed=2000)
            ler = float(rr['logical_error_rate'])
            k = int(round(ler * 1_000_000))
            row['ler_iid_p05'] = {'ler': ler, 'k': k, 'n': 1_000_000,
                                  'wilson95': wilson(k, 1_000_000)}
            row['wall_s'] = round(time.time() - t0, 1)
            res['units'][key] = row
            json.dump(res, open(OUT, 'w'), indent=2)
            print(f"[t3q4] {key}: loss={fl:.3e} w2={row['ler_w2']['k']} "
                  f"iid={row['ler_iid_p05']['ler']:.3e} "
                  f"[{row['wall_s']}s]", flush=True)
    print('[t3q4] DONE', flush=True)


if __name__ == '__main__':
    main()
