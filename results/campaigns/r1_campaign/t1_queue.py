#!/usr/bin/env python3
"""Central code selection: all-to-all depth sweep, full-basis stopping protocol,
and selection of the central ((5,3,3))_3 code.

Part A: weighted corrected-Meth KL objective, hardware-closure basis,
        importance@0.2 sampler, MS entangler, all-to-all connectivity, n=5,
        3000 steps, run_race num_seeds=1; L in {2,4,6,8,10,12,16} x seeds
        {100..103}. Per unit: weighted + cross-basis losses, LER|w1 @200k
        (seed 1001), LER|w2 @100k (1002), LER|w3 @100k (1003), i.i.d. LER
        p=0.05 @1M (2000).
Part B: 4 seeds {200..203}, UNWEIGHTED full closure basis, full batch,
        L=16, 3000 steps, run_race theta-init + 3-stage LR; TWO snapshots
        per seed (plateau rule: 100-step windows, 3 consecutive <1% rel
        improvements; convergence: loss-best theta over 3000). Both get the
        Part-A instrument battery.
Part C: central-code selection (min L with 4/4 k_w2=0, fallback 16; then
        (k_w2, k_w3, seed) lexicographic) + 2e6-shot p=0.05 measurement of
        the selected code AND [[5,1,3]], seed 3000.

Writes t1_results.json (sequential, resumable: done units skipped;
incremental JSON after every unit). Feeds the central-code selection and
the depth-sweep results of the paper.
Launch: nohup nice -n 19 python3 t1_queue.py >> t1.log 2>&1 &
"""
import os, sys, json, math, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import src.jax_backend  # x64 first (must precede any jnp array creation)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODEDIR = os.path.join(HERE, 't1_codes')
OUT = os.path.join(HERE, 't1_results.json')
STATUS = os.path.join(HERE, 't1_status.json')

D, N, STEPS, P, FRAC = 3, 5, 3000, 0.05, 0.2
LSWEEP = [2, 4, 6, 8, 10, 12, 16]
SEEDS_A = [100, 101, 102, 103]
SEEDS_B = [200, 201, 202, 203]
LB = 16                       # Part-B depth (same as loss_ler_diag/t1g_queue.py)
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]
MSBUDGET = lambda L: L * len(CONNS) * (D - 1)   # 20L


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    dn = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / dn
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / dn
    return (max(0.0, c - h), c + h)


def set_status(**kw):
    kw['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    json.dump(kw, open(STATUS, 'w'), indent=2)


# ---------------- instruments (t1g/c2 lineage) ----------------
_SINGLES = None
def singles():
    global _SINGLES
    if _SINGLES is None:
        from src.errors import qudit_hardware_error_basis
        _SINGLES = [np.asarray(E, complex)
                    for E in qudit_hardware_error_basis(D)]
    return _SINGLES


def wk_noise(w):
    from src.simulation import _apply_factored_op
    S = singles()
    def fn(state, rng):
        qs = rng.choice(N, size=w, replace=False)
        s = state
        for q in qs:
            s = _apply_factored_op(s, S[rng.integers(len(S))], int(q), N, D)
        nrm = np.linalg.norm(s)
        return s / nrm if nrm > 1e-15 else s
    return fn


def measure_code(cs, iid_shots=1_000_000, iid_seed=2000):
    """Full instrument battery on code states cs -> dict."""
    from src.simulation import simulate_ler_with_correction_factored
    from src.errors import make_hardware_noise_fn
    out = {}
    t0 = time.time()
    for w, sd, shots in ((1, 1001, 200_000), (2, 1002, 100_000),
                         (3, 1003, 100_000)):
        rr = simulate_ler_with_correction_factored(
            cs, wk_noise(w), singles(), N, D, n_shots=shots, seed=sd)
        ler = float(rr['logical_error_rate'])
        k = int(round(ler * shots))
        out[f'ler_w{w}'] = {'ler': ler, 'k': k, 'n': shots,
                            'wilson95': wilson(k, shots)}
    out['probe_wall_s'] = round(time.time() - t0, 1)
    t0 = time.time()
    rr = simulate_ler_with_correction_factored(
        cs, make_hardware_noise_fn(D, N, P), singles(), N, D,
        n_shots=iid_shots, seed=iid_seed)
    ler = float(rr['logical_error_rate'])
    k = int(round(ler * iid_shots))
    out['ler_iid_p05'] = {'ler': ler, 'k': k, 'n': iid_shots,
                          'wilson95': wilson(k, iid_shots)}
    out['iid_wall_s'] = round(time.time() - t0, 1)
    return out


# ---------------- Part A ----------------
def part_a(res):
    import jax.numpy as jnp
    from src.seed_race import run_race, build_race
    from src.decoders._common import encoder_forward
    xloss = {}   # L -> (hw_fn, full_fn), built lazily per L

    for L in LSWEEP:
        for seed in SEEDS_A:
            key = f'A_L{L}_s{seed}'
            if key in res['units']:
                continue
            set_status(state='running', unit=key, done=len(res['units']))
            t0 = time.time()
            npz = os.path.join(CODEDIR, key + '.npz')
            r = run_race(d=D, n=N, distance=3, layers=L, num_seeds=1,
                         steps=STEPS, target_loss=1e-9, sample_frac=FRAC,
                         sampler='importance', trigger_margin=0.25,
                         trigger_patience=5, out=npz, first_seed=seed,
                         verbose=False, weighted=True,
                         connectivity='all-to-all', entangler='ms')
            train_wall = time.time() - t0
            if L not in xloss:
                _, hw_fn, *_ = build_race(D, N, 3, L, noise='hw')
                _, full_fn, *_ = build_race(D, N, 3, L, noise='full')
                xloss[L] = (hw_fn, full_fn)
            theta = jnp.asarray(r['result']['theta'])
            row = {'part': 'A', 'L': L, 'seed': seed, 'npz': npz,
                   'ms_budget': MSBUDGET(L),
                   'train_loss_weighted': float(r['result']['full_loss']),
                   'hw_basis_loss': float(xloss[L][0](theta)),
                   'full_basis_loss': float(xloss[L][1](theta)),
                   'race_op_evs': int(r['race_op_evs']),
                   'train_wall_s': round(train_wall, 1)}
            cs = np.asarray(encoder_forward(npz, CONNS)[0])
            row.update(measure_code(cs))
            row['wall_s'] = round(time.time() - t0, 1)
            res['units'][key] = row
            json.dump(res, open(OUT, 'w'), indent=2)
            print(f"[t1] {key}: wloss={row['train_loss_weighted']:.3e} "
                  f"w2={row['ler_w2']['k']}/100k w3={row['ler_w3']['k']}/100k "
                  f"iid={row['ler_iid_p05']['ler']:.3e} "
                  f"[train {row['train_wall_s']}s total {row['wall_s']}s]",
                  flush=True)


# ---------------- Part B ----------------
def train_full_basis_dual(seed):
    """Full-batch full-basis training, 3000 steps, plateau + convergence
    snapshots. Returns dict with both thetas + traces."""
    import jax, jax.numpy as jnp, optax
    from src.seed_race import build_race
    (vg, full_fn, gs, ppl, _, fw, _) = build_race(D, N, 3, LB, noise='full')
    base_key = jax.random.PRNGKey(seed)
    init_key, _ = jax.random.split(base_key)
    theta = jax.random.uniform(init_key, (LB, ppl),
                               minval=0.0, maxval=2 * np.pi)
    lr = 0.05
    opt = optax.inject_hyperparams(optax.adam)(learning_rate=lr)
    st = opt.init(theta)
    stage = 0
    losses = []
    best_loss, best_theta, best_step = np.inf, theta, 0
    win_means, consec = [], 0
    plateau = None   # (step, theta, loss)
    for step in range(STEPS):
        lv, g = vg(theta, fw)
        lvf = float(lv)
        losses.append(lvf)
        if lvf < best_loss:
            best_loss, best_theta, best_step = lvf, theta, step
        if plateau is None and (step + 1) % 100 == 0:
            m = float(np.mean(losses[-100:]))
            if win_means and (win_means[-1] - m) / max(win_means[-1],
                                                      1e-300) < 0.01:
                consec += 1
            else:
                consec = 0
            win_means.append(m)
            if consec >= 3:
                plateau = (step, theta, lvf)
        upd, st = opt.update(g, st, theta)
        theta = optax.apply_updates(theta, upd)
        if stage == 0 and lvf < 0.5:
            st.hyperparams['learning_rate'] = lr / 5; stage = 1
        if stage <= 1 and lvf < 1e-3:
            st.hyperparams['learning_rate'] = lr / 50; stage = 2
    never = plateau is None
    if never:
        plateau = (STEPS - 1, theta, losses[-1])
    return {'plateau_step': int(plateau[0]),
            'plateau_theta': np.asarray(plateau[1]),
            'plateau_loss': float(plateau[2]),
            'plateau_never_detected': never,
            'conv_theta': np.asarray(best_theta),
            'conv_loss': float(best_loss), 'conv_step': int(best_step),
            'losses': losses}


def part_b(res):
    import jax
    from src.loss import save_varqec_result
    from src.decoders._common import encoder_forward
    for seed in SEEDS_B:
        key = f'B_s{seed}'
        if key in res['units']:
            continue
        set_status(state='running', unit=key, done=len(res['units']))
        t0 = time.time()
        tr = train_full_basis_dual(seed)
        train_wall = time.time() - t0
        row = {'part': 'B', 'L': LB, 'seed': seed,
               'ms_budget': MSBUDGET(LB),
               'protocol': 'unweighted full closure basis, full batch, '
                           '3000 steps, paired plateau+convergence snapshots',
               'plateau_step': tr['plateau_step'],
               'plateau_never_detected': tr['plateau_never_detected'],
               'conv_step': tr['conv_step'],
               'train_wall_s': round(train_wall, 1),
               'loss_trace_every25': tr['losses'][::25],
               'snapshots': {}}
        for tag in ('plateau', 'conv'):
            npz = os.path.join(CODEDIR, f'{key}_{tag}.npz')
            save_varqec_result(
                npz, tr[f'{tag}_theta'], [tr[f'{tag}_loss']],
                f'native_d{D}', 3, LB,
                metadata={'K': D, 'n_qudit': N, 'd': D, 'seed': seed,
                          'entangler': 'ms', 'connectivity': 'all-to-all',
                          'noise': 'full', 'snapshot': tag,
                          'snapshot_step': tr[f'{tag}_step']
                          if tag == 'conv' else tr['plateau_step'],
                          'backend': 'jax'})
            cs = np.asarray(encoder_forward(npz, CONNS)[0])
            snap = {'npz': npz, 'full_basis_loss': tr[f'{tag}_loss']}
            snap.update(measure_code(cs))
            row['snapshots'][tag] = snap
            print(f"[t1] {key}/{tag}: loss={snap['full_basis_loss']:.3e} "
                  f"w2={snap['ler_w2']['k']}/100k "
                  f"iid={snap['ler_iid_p05']['ler']:.3e}", flush=True)
        row['wall_s'] = round(time.time() - t0, 1)
        res['units'][key] = row
        json.dump(res, open(OUT, 'w'), indent=2)


# ---------------- Part C ----------------
def part_c(res):
    from src.simulation import simulate_ler_with_correction_factored
    from src.errors import make_hardware_noise_fn
    from src.decoders._common import encoder_forward
    from src.catalog import load_code
    if 'headline' in res:
        return
    set_status(state='running', unit='C_headline', done=len(res['units']))
    # 1. minimal L with 4/4 perfect w2
    lstar, perfect_by_L = None, {}
    for L in LSWEEP:
        ks = [res['units'][f'A_L{L}_s{s}']['ler_w2']['k'] for s in SEEDS_A]
        perfect_by_L[L] = ks
        if lstar is None and all(k == 0 for k in ks):
            lstar = L
    fallback = lstar is None
    if fallback:
        lstar = 16
    # 2. select by (k_w2, k_w3, seed)
    rank = sorted(SEEDS_A, key=lambda s: (
        res['units'][f'A_L{lstar}_s{s}']['ler_w2']['k'],
        res['units'][f'A_L{lstar}_s{s}']['ler_w3']['k'], s))
    sel = rank[0]
    sel_unit = res['units'][f'A_L{lstar}_s{sel}']
    # 3. 2e6-shot head-to-head
    shots = 2_000_000
    rows = {}
    cs_sel = np.asarray(encoder_forward(sel_unit['npz'], CONNS)[0])
    cs_513 = np.asarray(load_code('five_qudit_d3')['code_states'])
    for name, cs in (('t1_headline', cs_sel), ('513', cs_513)):
        t0 = time.time()
        rr = simulate_ler_with_correction_factored(
            cs, make_hardware_noise_fn(D, N, P), singles(), N, D,
            n_shots=shots, seed=3000)
        ler = float(rr['logical_error_rate'])
        k = int(round(ler * shots))
        rows[name] = {'ler': ler, 'k': k, 'n': shots,
                      'wilson95': wilson(k, shots),
                      'wall_s': round(time.time() - t0, 1)}
    ratio = (rows['513']['ler'] / rows['t1_headline']['ler']
             if rows['t1_headline']['ler'] > 0 else None)
    res['headline'] = {
        'selection_protocol': 'min L with k_w2=0 for all 4 seeds '
                              '(fallback 16), then lexicographic '
                              '(k_w2, k_w3, seed); probe-based only',
        'perfect_w2_by_L': perfect_by_L, 'L_star': lstar,
        'L_star_fallback_used': fallback, 'selected_seed': sel,
        'selected_npz': sel_unit['npz'], 'ms_budget': MSBUDGET(lstar),
        'p': P, 'measurement': rows, 'ratio_513_over_headline': ratio}
    json.dump(res, open(OUT, 'w'), indent=2)
    print(f"[t1] HEADLINE: L*={lstar} seed={sel} "
          f"LER={rows['t1_headline']['ler']:.3e} vs 513 "
          f"{rows['513']['ler']:.3e} ratio={ratio}", flush=True)


def main():
    os.makedirs(CODEDIR, exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {
        'protocol': 'Central code selection: Part A weighted '
                    'importance@0.2 hw-closure L-sweep x 4 seeds; Part B '
                    'full-basis paired stopping protocol; Part C '
                    'pre-registered central-code selection + 2e6-shot '
                    'head-to-head vs [[5,1,3]].',
        'units': {}}
    part_a(res)
    part_b(res)
    part_c(res)
    set_status(state='done', done=len(res['units']))
    print('[t1] QUEUE DONE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        set_status(state='FAILED', error=str(e))
        raise
