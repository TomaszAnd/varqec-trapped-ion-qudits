#!/usr/bin/env python3
"""q = 4, 5 codes: q-panel retrains ((5,3,3))_4 and ((5,3,3))_5,
noise-weighted, minimal L (replaces the L=16 unweighted panel codes
d4K3/d5K3).

Protocol (pre-registered):
- K-explicit weighted trainer (r2_campaign/t2_queue.py kexp lineage,
  generalized to d): build_varqec_loss_weighted(K=3) over the
  hardware-closure grouped basis at d in {4,5}, weights =
  compute_meth_group_weights(d, 5, distance=3) — the q=4/5 channel weights
  — full batch, 3000 steps, three-stage Adam LR (0.05, /5 below 0.5,
  /50 below 1e-3), theta-init PRNGKey(seed)->split->uniform[0,2pi).
- L ascent {2,4,8}, 4 seeds {100..103}; minimal L = first with 4/4 seeds
  w2-probe-perfect (k=0/100k); fallback: smallest L with >=1 perfect seed,
  flagged probe_selection_mandatory.
- Selection at minimal L: lexicographic (k_w2, k_w3, seed).
- Panel measurement (selected code per q): conditioned probes w=1..5
  (w1 200k seed 1001; w2..w5 100k seeds 1002..1005) + direct MC on the
  standing q-panel grid p in {0.02,0.03,0.05,0.08,0.12,0.15}, shots 3e5
  escalated x2 until >=30 errors, cap 5M. Cells below the cap floor get the
  RECONSTRUCTED LER sum_{w=1..5} C(5,w) p^w (1-p)^{5-w} * LER|w
  (reconstructed points labelled; validation factor = direct/reconstructed
  on every resolved cell).

Writes t2b_results.json (incremental, resume anchor). Feeds the q = 4, 5
code results of the paper.
Launch: nohup nice -n 19 python3 t2b_queue.py >> t2b.log 2>&1 &
"""
import os, sys, json, math, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)
import src.jax_backend  # x64 first
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 't2b_results.json')
STATUS = os.path.join(HERE, 't2b_status.json')
CODEDIR = os.path.join(HERE, 't2b_codes')
os.makedirs(CODEDIR, exist_ok=True)

N, K = 5, 3
QS = [4, 5]
LS = [2, 4, 8]
SEEDS = [100, 101, 102, 103]
STEPS = 3000
LR = 0.05
P_GRID = [0.02, 0.03, 0.05, 0.08, 0.12, 0.15]
MC_START, MC_CAP, MC_MINERR = 300_000, 5_000_000, 30
PROBE = [(1, 1001, 200_000), (2, 1002, 100_000), (3, 1003, 100_000),
         (4, 1004, 100_000), (5, 1005, 100_000)]
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]
MSBUDGET = lambda L, d: L * len(CONNS) * (d - 1)


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


_PACKS = {}
def kexp_pack(d):
    """K-explicit weighted loss pack at dimension d (r2_campaign/t2_queue.py lineage)."""
    if d in _PACKS:
        return _PACKS[d]
    import jax.numpy as jnp
    from src.errors import ErrorModel
    from src.jax_backend import create_jax_encoder, build_varqec_loss_weighted
    from src.meth_weights import compute_meth_group_weights
    enc, _, ppl = create_jax_encoder(N, d, connections=CONNS, use_scan=True)
    grouped = ErrorModel(d=d, n_qudit=N, distance=3,
                         closed=True).build_grouped(verbose=False)
    loss_w = build_varqec_loss_weighted(encoder_fn=enc, K=K, d=d, n_qudit=N,
                                        distance=3, E_det_grouped=grouped)
    gw, _ = compute_meth_group_weights(d, N, distance=3)
    w = tuple(jnp.asarray(x, dtype=jnp.float64) for x in gw)
    _PACKS[d] = (enc, ppl, loss_w, w)
    return _PACKS[d]


def train_one(d, L, seed):
    import jax, optax
    enc, ppl, loss_w, w = kexp_pack(d)
    val_grad = jax.jit(jax.value_and_grad(lambda th: loss_w(th, w)))
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


_SING = {}
def singles(d):
    if d not in _SING:
        from src.errors import qudit_hardware_error_basis
        _SING[d] = [np.asarray(E, complex)
                    for E in qudit_hardware_error_basis(d)]
    return _SING[d]


def wk_noise(d, w):
    from src.simulation import _apply_factored_op
    S = singles(d)
    def fn(state, rng):
        qs = rng.choice(N, size=w, replace=False)
        s = state
        for q in qs:
            s = _apply_factored_op(s, S[rng.integers(len(S))], int(q), N, d)
        nrm = np.linalg.norm(s)
        return s / nrm if nrm > 1e-15 else s
    return fn


def save_code(npz, d, L, seed, loss, theta):
    from src.loss import save_varqec_result
    save_varqec_result(
        npz, np.asarray(theta), [loss], f'native_d{d}', 3, L,
        metadata={'K': K, 'n_qudit': N, 'd': d, 'seed': seed,
                  'entangler': 'ms', 'connectivity': 'all-to-all',
                  'backend': 'jax', 'weighted': True, 'eta': 0.9296,
                  'connections': np.asarray(CONNS, dtype=int)})


def code_states_from(npz):
    from src.decoders._common import encoder_forward
    return np.asarray(encoder_forward(npz, CONNS)[0])


def probes(cs, d, ws=(1, 2, 3)):
    from src.simulation import simulate_ler_with_correction_factored
    out = {}
    for w, sd, shots in PROBE:
        if w not in ws:
            continue
        rr = simulate_ler_with_correction_factored(
            cs, wk_noise(d, w), singles(d), N, d, n_shots=shots, seed=sd)
        k = int(round(float(rr['logical_error_rate']) * shots))
        out[f'ler_w{w}'] = {'k': k, 'n': shots, 'ler': k / shots,
                            'wilson95': wilson(k, shots)}
    return out


def phase_a(res):
    """L ascent + probes per (q, L, seed); minimal-L decision per q."""
    for d in QS:
        qkey = f'q{d}'
        if res['minimal_L'].get(qkey):
            continue
        found = None
        for L in LS:
            perfect = 0
            for seed in SEEDS:
                ukey = f'{qkey}_L{L}_s{seed}'
                if ukey not in res['units']:
                    set_status(state='running', unit=ukey)
                    t0 = time.time()
                    loss, theta = train_one(d, L, seed)
                    npz = os.path.join(CODEDIR, ukey + '.npz')
                    save_code(npz, d, L, seed, loss, theta)
                    cs = code_states_from(npz)
                    row = {'d': d, 'L': L, 'seed': seed, 'npz': npz,
                           'ms_budget': MSBUDGET(L, d),
                           'train_loss_weighted': loss,
                           'train_wall_s': round(time.time() - t0, 1)}
                    row.update(probes(cs, d))
                    row['wall_s'] = round(time.time() - t0, 1)
                    res['units'][ukey] = row
                    json.dump(res, open(OUT, 'w'), indent=2)
                    print(f"[t2b] {ukey}: wloss={loss:.3e} "
                          f"w2={row['ler_w2']['k']}/100k "
                          f"w3={row['ler_w3']['k']}/100k "
                          f"[{row['wall_s']}s]", flush=True)
                if res['units'][f'{qkey}_L{L}_s{seed}']['ler_w2']['k'] == 0:
                    perfect += 1
            if perfect == len(SEEDS):
                found = (L, False)
                break
        if found is None:
            # fallback: smallest L with >=1 perfect seed
            for L in LS:
                ks = [res['units'][f'{qkey}_L{L}_s{s}']['ler_w2']['k']
                      for s in SEEDS]
                if min(ks) == 0:
                    found = (L, True)
                    break
        if found is None:
            found = (LS[-1], True)
        Lmin, fallback = found
        rows = [(res['units'][f'{qkey}_L{Lmin}_s{s}']['ler_w2']['k'],
                 res['units'][f'{qkey}_L{Lmin}_s{s}']['ler_w3']['k'], s)
                for s in SEEDS]
        rows.sort()
        sel = rows[0][2]
        res['minimal_L'][qkey] = {
            'L_min': Lmin, 'probe_selection_mandatory': fallback,
            'selected_seed': sel,
            'selected_npz': res['units'][f'{qkey}_L{Lmin}_s{sel}']['npz'],
            'ms_budget': MSBUDGET(Lmin, d),
            'selection': '(k_w2, k_w3, seed) lexicographic'}
        json.dump(res, open(OUT, 'w'), indent=2)
        print(f"[t2b] {qkey}: L_min={Lmin} fallback={fallback} "
              f"selected s{sel}", flush=True)


def binom(n, p, w):
    return math.comb(n, w) * p ** w * (1 - p) ** (n - w)


def phase_b(res):
    """Selected code per q: w4/w5 probes + direct-MC p-grid + reconstruction."""
    from src.simulation import simulate_ler_with_correction_factored
    from src.errors import make_hardware_noise_fn
    for d in QS:
        qkey = f'q{d}'
        sel = res['minimal_L'][qkey]
        ukey = f'{qkey}_L{sel["L_min"]}_s{sel["selected_seed"]}'
        cs = code_states_from(sel['selected_npz'])
        pkey = f'panel_{qkey}'
        if pkey not in res['panel']:
            res['panel'][pkey] = {'probes': dict(
                (k, v) for k, v in res['units'][ukey].items()
                if k.startswith('ler_w')), 'cells': {}}
        pan = res['panel'][pkey]
        # extend probes to w=4,5
        for w in (4, 5):
            if f'ler_w{w}' not in pan['probes']:
                set_status(state='running', unit=f'{pkey}_w{w}')
                t0 = time.time()
                pan['probes'].update(probes(cs, d, ws=(w,)))
                json.dump(res, open(OUT, 'w'), indent=2)
                print(f"[t2b] {pkey} w{w}: "
                      f"k={pan['probes'][f'ler_w{w}']['k']}/100k "
                      f"[{round(time.time()-t0,1)}s]", flush=True)
        for p in P_GRID:
            ckey = f'p{p}'
            if ckey in pan['cells']:
                continue
            set_status(state='running', unit=f'{pkey}_{ckey}')
            t0 = time.time()
            shots = MC_START
            while True:
                rr = simulate_ler_with_correction_factored(
                    cs, make_hardware_noise_fn(d, N, p), singles(d), N, d,
                    n_shots=shots, seed=2000)
                k = int(round(float(rr['logical_error_rate']) * shots))
                if k >= MC_MINERR or shots >= MC_CAP:
                    break
                shots = min(MC_CAP, shots * 2)
            rec = sum(binom(N, p, w) * pan['probes'][f'ler_w{w}']['ler']
                      for w in (1, 2, 3, 4, 5))
            cell = {'p': p, 'k': k, 'shots': shots, 'resolved': bool(k >= MC_MINERR),
                    'ler_direct': k / shots, 'wilson95': wilson(k, shots),
                    'ler_reconstructed': rec,
                    'validation_factor': (k / shots) / rec
                                          if (k >= MC_MINERR and rec > 0) else None,
                    'wall_s': round(time.time() - t0, 1)}
            pan['cells'][ckey] = cell
            json.dump(res, open(OUT, 'w'), indent=2)
            print(f"[t2b] {pkey} p={p}: k={k}/{shots} rec={rec:.3e} "
                  f"[{cell['wall_s']}s]", flush=True)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {
        'protocol': __doc__, 'units': {}, 'minimal_L': {}, 'panel': {}}
    json.dump(res, open(OUT, 'w'), indent=2)
    phase_a(res)
    phase_b(res)
    set_status(state='done')
    print('[t2b] DONE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        set_status(state='FAILED', error=str(e))
        raise
