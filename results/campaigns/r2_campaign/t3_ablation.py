#!/usr/bin/env python3
"""Per-layer entangler ablation at minimal L: how few entanglers per layer
survive with intact probes?

Extends the magnitude-pruning frontier of r1_campaign/t2t4_queue.py (flat
to 50%) on the central code A_L2_s100 (n=5, L=2, 40 MS gates) to
percentiles {60,70,80,90} (keep 16/12/8/4 gates). Identical machinery:
wrapped-|theta| percentile threshold over (layer, conn, transition) gates;
pruned gates set to identity and FROZEN through a 300-step weighted
importance@0.2 fine-tune (Adam lr 0.01, rng 9000+pct); full battery per
point. Also records surviving-gates-per-layer. Writes t3abl_results.json
(resumable).
"""
import os, sys, json, math, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import src.jax_backend  # x64 first
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODEDIR = os.path.join(HERE, 't3_codes')
OUT = os.path.join(HERE, 't3abl_results.json')
D, N, L, FRAC = 3, 5, 2, 0.2
PCTS = [60, 70, 80, 90]
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]
SRC_NPZ = os.path.join(REPO, 'results/campaigns/r1_campaign',
                       't1_codes', 'A_L2_s100.npz')


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    dn = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / dn
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / dn
    return (max(0.0, c - h), c + h)


def battery(cs):
    from src.simulation import (simulate_ler_with_correction_factored,
                                _apply_factored_op)
    from src.errors import qudit_hardware_error_basis, make_hardware_noise_fn
    S = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]
    def wk(w):
        def fn(state, rng):
            qs = rng.choice(N, size=w, replace=False)
            s = state
            for q in qs:
                s = _apply_factored_op(s, S[rng.integers(len(S))], int(q),
                                       N, D)
            nrm = np.linalg.norm(s)
            return s / nrm if nrm > 1e-15 else s
        return fn
    out = {}
    for w, sd, shots in ((1, 1001, 200_000), (2, 1002, 100_000),
                         (3, 1003, 100_000)):
        rr = simulate_ler_with_correction_factored(
            cs, wk(w), S, N, D, n_shots=shots, seed=sd)
        k = int(round(float(rr['logical_error_rate']) * shots))
        out[f'ler_w{w}'] = {'k': k, 'n': shots, 'ler': k / shots,
                            'wilson95': wilson(k, shots)}
    rr = simulate_ler_with_correction_factored(
        cs, make_hardware_noise_fn(D, N, 0.05), S, N, D,
        n_shots=1_000_000, seed=2000)
    k = int(round(float(rr['logical_error_rate']) * 1_000_000))
    out['ler_iid_p05'] = {'k': k, 'n': 1_000_000, 'ler': k / 1_000_000,
                          'wilson95': wilson(k, 1_000_000)}
    return out


def main():
    import jax, jax.numpy as jnp, optax
    from src.seed_race import build_race
    from src.loss import save_varqec_result
    from src.decoders._common import encoder_forward
    from src.sampling.stratified_importance import (
        make_stratified_importance_weights)
    os.makedirs(CODEDIR, exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {'units': {}}

    data = np.load(SRC_NPZ, allow_pickle=True)
    theta0 = np.array(data['params'], dtype=float)
    nsing = 4 * N * (D - 1)
    nc = len(CONNS)
    ngates = L * nc * (D - 1)                    # 40
    mags = np.empty((L, nc, D - 1))
    for c in range(nc):
        for t in range(D - 1):
            th = theta0[:, nsing + 4 * c + 2 * t + 1]
            mags[:, c, t] = np.abs((th + np.pi) % (2 * np.pi) - np.pi)

    (vg, full_wloss_fn, gs, ppl, _, fw, sgw) = build_race(
        D, N, 3, L, weighted=True)
    total_ops = int(sum(gs))
    for pct in PCTS:
        key = f'abl_prune{pct}'
        if key in res['units']:
            continue
        t0 = time.time()
        thr = np.percentile(mags, pct)
        pruned = mags <= thr
        theta = theta0.copy()
        gmask = np.ones_like(theta)
        for (l, c, t) in zip(*np.nonzero(pruned)):
            base = nsing + 4 * c + 2 * t
            theta[l, base:base + 2] = 0.0
            gmask[l, base:base + 2] = 0.0
        n_pruned = int(pruned.sum())
        per_layer = [int((~pruned[l]).sum()) for l in range(L)]
        th = jnp.asarray(theta)
        gm = jnp.asarray(gmask)
        opt = optax.adam(learning_rate=0.01)
        st = opt.init(th)
        rng = np.random.default_rng(9000 + pct)
        budget = max(1, int(round(FRAC * total_ops)))
        for step in range(300):
            w = make_stratified_importance_weights(sgw, budget, rng)
            _, g = vg(th, w)
            g = jax.tree_util.tree_map(lambda x: x * gm, g)
            upd, st = opt.update(g, st, th)
            th = optax.apply_updates(th, upd)
            th = jnp.where(gm > 0, th, 0.0)
        wloss = float(full_wloss_fn(th))
        npz = os.path.join(CODEDIR, f'abl_prune{pct}.npz')
        save_varqec_result(npz, np.asarray(th), [wloss], f'native_d{D}',
                           3, L,
                           metadata={'K': D, 'n_qudit': N, 'd': D,
                                     'entangler': 'ms', 'pruned_pct': pct,
                                     'n_gates_pruned': n_pruned,
                                     'source': SRC_NPZ})
        cs = np.asarray(encoder_forward(npz, CONNS)[0])
        row = {'pct': pct, 'threshold_wrapped_theta': float(thr),
               'n_gates_pruned': n_pruned,
               'ms_budget': ngates - n_pruned,
               'surviving_gates_per_layer': per_layer,
               'wloss_after_finetune': wloss, 'npz': npz}
        row.update(battery(cs))
        row['wall_s'] = round(time.time() - t0, 1)
        res['units'][key] = row
        json.dump(res, open(OUT, 'w'), indent=2)
        print(f"[t3abl] prune{pct}: budget={row['ms_budget']} "
              f"per_layer={per_layer} wloss={wloss:.3e} "
              f"w2={row['ler_w2']['k']} "
              f"iid={row['ler_iid_p05']['ler']:.3e} [{row['wall_s']}s]",
              flush=True)
    print('[t3abl] DONE', flush=True)


if __name__ == '__main__':
    main()
