#!/usr/bin/env python3
"""Weighted benchmarks: re-measurement on the channel-weighted i.i.d. benchmark.

Protocol identical to the standing uniform instrument (lookup decoder over
single-qudit inverses, Haar states, tau=0.5, seed conventions unchanged);
ONLY the per-op draw changes, per src.errors.make_weighted_hardware_noise_fn
with weights from src.meth_weights.hardware_benchmark_weights (eta=0.9296):
  arm 'meth' = kind meth-spectator (the training prior's projections —
               mildly non-uniform [0.210, 0.307, 0.241, 0.242] at d=3);
  arm 'deph' = kind dephasing-exact (identity-shadow-free channel mass —
               strongly non-uniform [0.014, 0.656, 0.164, 0.166] at d=3).
Injection probability p is the SAME as the uniform benchmark at every
quoted p (calibration exact: all benchmark ops share Tr = d-2).

Units:
 (A) Residual ladder (Table IV roster, n=5) x {meth, deph}:
     weighted-draw conditioned probes LER|w for w=1,2,3 (100k shots,
     seeds 1001/1002/1003) + weighted iid LER @ p=0.05 (1e6 shots, seed
     2000) + reconstruction check sum_w P(w|p) LER|w (P(w|p) binomial —
     unchanged by reweighting). Central code vs [[5,1,3]] at 1e6 shots is
     rows 'headline' and 'code513'.
 (B) n-panel family (n=5..9 minimal-L matched codes) x {meth, deph} x
     p in {0.05, 0.08}: shots 3e5 escalated x2 until >=30 errors, cap 5M.

Writes t1c_results.json (incremental; resume anchor: completed units
skipped). Feeds the weighted benchmarks of the paper.
Launch: nohup nice -n 19 python3 t1c_queue.py >> t1c.log 2>&1 &
"""
import os, sys, json, math, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)
import src.jax_backend  # x64 first
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 't1c_results.json')
STATUS = os.path.join(HERE, 't1c_status.json')
BPR = os.path.join(REPO, 'results/campaigns')

D = 3
ETA = 0.9296
P_LADDER = 0.05
LADDER_IID_SHOTS = 1_000_000
PROBE_SHOTS = 100_000
PROBE_SEEDS = {1: 1001, 2: 1002, 3: 1003}
IID_SEED = 2000
PANEL_PS = [0.05, 0.08]
PANEL_START, PANEL_CAP, PANEL_MINERR = 300_000, 5_000_000, 30

LADDER = [
    ('headline', 'r1_campaign/t1_codes/A_L2_s100.npz', 5),
    ('hw_L16', 'loss_ler_diag/t1g/hw_s100.npz', 5),
    ('plateau_s202', 'r1_campaign/t1_codes/B_s202_plateau.npz', 5),
    ('conv_s202', 'r1_campaign/t1_codes/B_s202_conv.npz', 5),
    ('deep_L28', 'headline_p05_Lsweep/n5_L28.npz', 5),
    ('code513', None, 5),  # catalog
]
PANEL = [
    ('n5', 'r1_campaign/t1_codes/A_L2_s100.npz', 5),
    ('n6', 'r2_campaign/t4_codes/n6_L2_s102.npz', 6),
    ('n7', 'r2_campaign/t1a_codes/d3_L2_s100.npz', 7),
    ('n8', 'r2_campaign/t4_codes/n8_L2_s100.npz', 8),
    ('n9', 'r2_campaign/t4_codes/n9_L2_s100.npz', 9),
]
ARMS = ['meth', 'deph']
KIND = {'meth': 'meth-spectator', 'deph': 'dephasing-exact'}


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


_SING = None
def singles():
    global _SING
    if _SING is None:
        from src.errors import qudit_hardware_error_basis
        _SING = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]
    return _SING


_W = {}
def weights(arm):
    if arm not in _W:
        from src.meth_weights import hardware_benchmark_weights
        _W[arm] = hardware_benchmark_weights(D, eta=ETA, kind=KIND[arm])
    return _W[arm]


def weighted_wk_noise(n, w, arm):
    """Exactly-weight-w noise with the arm's per-op draw."""
    from src.simulation import _apply_factored_op
    S = singles()
    cumw = np.cumsum(weights(arm)); cumw[-1] = 1.0
    def fn(state, rng):
        qs = rng.choice(n, size=w, replace=False)
        s = state
        for q in qs:
            idx = int(np.searchsorted(cumw, rng.random(), side='right'))
            s = _apply_factored_op(s, S[idx], int(q), n, D)
        nrm = np.linalg.norm(s)
        return s / nrm if nrm > 1e-15 else s
    return fn


_CS = {}
def code_states(name, rel, n):
    if name not in _CS:
        if rel is None:
            from src.catalog import load_code
            _CS[name] = np.asarray(load_code('five_qudit_d3')['code_states'])
        else:
            from src.decoders._common import encoder_forward
            conns = [[i, j] for i in range(n) for j in range(i + 1, n)]
            _CS[name] = np.asarray(
                encoder_forward(os.path.join(BPR, rel), conns)[0])
    return _CS[name]


def run_cell(cs, noise_fn, n, shots, seed):
    from src.simulation import simulate_ler_with_correction_factored
    rr = simulate_ler_with_correction_factored(
        cs, noise_fn, singles(), n, D, n_shots=shots, seed=seed)
    ler = float(rr['logical_error_rate'])
    k = int(round(ler * shots))
    return k, shots


def binom_pw(n, p, w):
    return math.comb(n, w) * p ** w * (1 - p) ** (n - w)


def ladder_units(res):
    from src.errors import make_weighted_hardware_noise_fn
    for name, rel, n in LADDER:
        for arm in ARMS:
            key = f'ladder_{name}_{arm}'
            if key in res['ladder']:
                continue
            set_status(state='running', unit=key)
            t0 = time.time()
            cs = code_states(name, rel, n)
            row = {'code': name, 'arm': arm, 'n': n}
            for w in (1, 2, 3):
                k, sh = run_cell(cs, weighted_wk_noise(n, w, arm), n,
                                 PROBE_SHOTS, PROBE_SEEDS[w])
                row[f'ler_w{w}'] = {'k': k, 'n': sh, 'ler': k / sh,
                                    'wilson95': wilson(k, sh)}
            nf = make_weighted_hardware_noise_fn(D, n, P_LADDER,
                                                 weights(arm))
            k, sh = run_cell(cs, nf, n, LADDER_IID_SHOTS, IID_SEED)
            row['ler_iid_p05'] = {'k': k, 'n': sh, 'ler': k / sh,
                                  'wilson95': wilson(k, sh)}
            rec = sum(binom_pw(n, P_LADDER, w) * row[f'ler_w{w}']['ler']
                      for w in (1, 2, 3))
            row['reconstruction_w123'] = rec
            row['reconstruction_factor'] = (row['ler_iid_p05']['ler'] / rec
                                            if rec > 0 else None)
            row['wall_s'] = round(time.time() - t0, 1)
            res['ladder'][key] = row
            json.dump(res, open(OUT, 'w'), indent=2)
            print(f"[t1c] {key}: w2={row['ler_w2']['k']}/100k "
                  f"w3={row['ler_w3']['k']}/100k "
                  f"iid={row['ler_iid_p05']['ler']:.3e} "
                  f"[{row['wall_s']}s]", flush=True)


def panel_units(res):
    from src.errors import make_weighted_hardware_noise_fn
    for name, rel, n in PANEL:
        for arm in ARMS:
            for p in PANEL_PS:
                key = f'panel_{name}_{arm}_p{p}'
                if key in res['panel']:
                    continue
                set_status(state='running', unit=key)
                t0 = time.time()
                cs = code_states(f'panel_{name}', rel, n)
                shots = PANEL_START
                while True:
                    nf = make_weighted_hardware_noise_fn(D, n, p,
                                                         weights(arm))
                    k, sh = run_cell(cs, nf, n, shots, IID_SEED)
                    if k >= PANEL_MINERR or shots >= PANEL_CAP:
                        break
                    shots = min(PANEL_CAP, shots * 2)
                res['panel'][key] = {
                    'code': name, 'arm': arm, 'n': n, 'p': p,
                    'k': k, 'shots': sh, 'ler': k / sh,
                    'wilson95': wilson(k, sh),
                    'resolved': bool(k >= PANEL_MINERR),
                    'wall_s': round(time.time() - t0, 1)}
                json.dump(res, open(OUT, 'w'), indent=2)
                print(f"[t1c] {key}: k={k}/{sh} LER={k/sh:.3e} "
                      f"[{res['panel'][key]['wall_s']}s]", flush=True)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {
        'protocol': __doc__, 'eta': ETA,
        'draw_meth': list(map(float, weights('meth'))),
        'draw_deph': list(map(float, weights('deph'))),
        'ladder': {}, 'panel': {}}
    json.dump(res, open(OUT, 'w'), indent=2)
    ladder_units(res)
    panel_units(res)
    set_status(state='done')
    print('[t1c] DONE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        set_status(state='FAILED', error=str(e))
        raise
