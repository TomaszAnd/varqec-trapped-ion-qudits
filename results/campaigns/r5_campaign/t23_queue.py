#!/usr/bin/env python3
"""q-panel reconstruction measurements (Fig. 4) and distance-panel
verification.

Phases (sequential, incremental JSON t23_results.json, resume-safe):

P1  t3_residuals  — KL detection residuals of the ((7,3,d))_3 trio
    (t2c d2/d3/d4 selected codes, one shared ansatz) evaluated on the
    NESTED closures dist in {2,3,4} (hw basis, unweighted) plus the
    weighted dist=4 residual: does the d=4 arm carry a distinctly lower
    weight-3 detection residual than d2/d3?

P2  t3_detection_probes — conditioned exactly-w detection-and-postselect
    probes (declare-and-discard; batched instrument validated against
    src.simulation.simulate_ler_with_detection on a 2k-shot cell):
    w in {1,2,3,4} x {d2,d3,d4} at 100k shots. The discriminator the
    correction protocol cannot see: does d4 detect (or tolerate) w=3
    errors that d3 converts into undetected logical errors?

P3  t2_qpanel — conditioned LER|w for ((5,3,3))_4 and ((5,3,3))_5 at
    L=2 (t2b s100 codes): w1@4M, w2@2M, w3@2M, w4@2M(q4)/keep(q5 0/2M
    from t2b), w5@100k(q4)/keep(q5). Reconstruction LER(p) =
    sum_w P(w|p) LER|w (point + UB curves), measurable-cell computation
    at <=1e7 shots, and direct-MC anchors at the measurable p cells
    (q4: expected p=0.12, 0.15).

P4  t3_postselect_sweep — iid detection-and-postselect p-sweep for the
    trio: p in {0.05, 0.08, 0.12}, escalated x2 from 5e5 shots until
    >=30 undetected-logical errors or 5M cap; acceptance fraction +
    undetected-logical-error rate (per accepted shot) + Wilson CIs.

Writes t23_results.json. Feeds the q-panel reconstruction (Fig. 4).
Launch: nohup nice -n 19 python3 t23_queue.py >> t23.log 2>&1 &
"""
import os, sys, json, math, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)
import src.jax_backend  # x64 first
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 't23_results.json')
STATUS = os.path.join(HERE, 't23_status.json')

R3 = os.path.join(REPO, 'results/campaigns/r3_campaign')
TRIO = {'d2': f'{R3}/t2c_codes/d2_L2_s100.npz',
        'd3': f'{R3}/t2c_codes/d3_L2_s100.npz',
        'd4': f'{R3}/t2c_codes/d4_L2_s102.npz'}
QCODES = {'q4': f'{R3}/t2b_codes/q4_L2_s100.npz',
          'q5': f'{R3}/t2b_codes/q5_L2_s100.npz'}
N7, D7 = 7, 3
CONNS7 = [[i, j] for i in range(N7) for j in range(i + 1, N7)]
N5 = 5
CONNS5 = [[i, j] for i in range(N5) for j in range(i + 1, N5)]
PGRID = [0.02, 0.03, 0.05, 0.08, 0.12, 0.15]
DET_THRESH = 0.1


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


def singles(d):
    from src.errors import qudit_hardware_error_basis
    return [np.asarray(E, complex) for E in qudit_hardware_error_basis(d)]


def wk_noise(n, d, w, S):
    from src.simulation import _apply_factored_op
    def fn(state, rng):
        qs = rng.choice(n, size=w, replace=False)
        s = state
        for q in qs:
            s = _apply_factored_op(s, S[rng.integers(len(S))], int(q), n, d)
        nrm = np.linalg.norm(s)
        return s / nrm if nrm > 1e-15 else s
    return fn


def detection_postselect(cs, noise_fn, n_shots, seed, thresh=DET_THRESH,
                         batch=4096):
    """Batched declare-and-discard: project onto the code space via codeword
    coefficients (equivalent to the src detection_decoder projector for
    orthonormal codewords, without the dim^2 projector per shot).

    Returns dict(n_shots, detected, accepted, undet_errors,
    undetected_error_rate, acceptance, ps_fid_mean).
    """
    K, dim = cs.shape
    rng = np.random.default_rng(seed)
    csc = cs.conj()
    n_det = 0
    n_err = 0
    fid_sum = 0.0
    done = 0
    while done < n_shots:
        b = min(batch, n_shots - done)
        co = rng.standard_normal((b, K)) + 1j * rng.standard_normal((b, K))
        co /= np.linalg.norm(co, axis=1, keepdims=True)
        logical = co @ cs
        noisy = np.empty_like(logical)
        for i in range(b):
            noisy[i] = noise_fn(logical[i], rng)
        pc = noisy @ csc.T                      # (b, K) codeword coefficients
        pn2 = np.sum(np.abs(pc) ** 2, axis=1)   # projection norm^2
        det = pn2 < thresh
        n_det += int(det.sum())
        acc = ~det
        if acc.any():
            # fid = |<logical|projected_normalized>|^2
            ov = np.abs(np.sum(co.conj() * pc, axis=1)) ** 2
            fid = ov[acc] / pn2[acc]
            n_err += int((fid < 0.5).sum())
            fid_sum += float(fid.sum())
        done += b
    n_acc = n_shots - n_det
    return {'n_shots': n_shots, 'detected': n_det, 'accepted': n_acc,
            'undet_errors': n_err,
            'undetected_error_rate': (n_err / n_acc) if n_acc else 1.0,
            'acceptance': n_acc / n_shots,
            'ps_fid_mean': (fid_sum / n_acc) if n_acc else 0.0}


# ---------------------------------------------------------------- P1
def p1_residuals(res):
    if 't3_residuals' in res:
        return
    set_status(state='P1 residuals')
    from src.seed_race import build_race
    out = {}
    thetas = {lab: np.load(p)['params'] for lab, p in TRIO.items()}
    for dist in (2, 3, 4):
        t0 = time.time()
        _, full_u, gs, *_ = build_race(D7, N7, dist, 2, noise='hw',
                                       weighted=False, entangler='ms')
        n_ops = int(sum(gs))
        for lab, th in thetas.items():
            out[f'unweighted_dist{dist}_{lab}'] = float(full_u(th))
        out[f'n_ops_dist{dist}'] = n_ops
        print(f'[p1] unweighted dist={dist} ({n_ops} ops) '
              f'{time.time()-t0:.0f}s: ' +
              ' '.join(f"{l}={out[f'unweighted_dist{dist}_{l}']:.3e}"
                       for l in thetas), flush=True)
    t0 = time.time()
    _, full_w4, *_ = build_race(D7, N7, 4, 2, weighted=True, entangler='ms')
    for lab, th in thetas.items():
        out[f'weighted_dist4_{lab}'] = float(full_w4(th))
    print(f'[p1] weighted dist=4 {time.time()-t0:.0f}s: ' +
          ' '.join(f"{l}={out[f'weighted_dist4_{l}']:.3e}" for l in thetas),
          flush=True)
    res['t3_residuals'] = out
    json.dump(res, open(OUT, 'w'), indent=2)


# ---------------------------------------------------------------- P2
def p2_detection_probes(res):
    from src.decoders._common import encoder_forward
    if 't3_det_probes' not in res:
        res['t3_det_probes'] = {}
    probes = res['t3_det_probes']
    S = singles(D7)
    # validation cell vs the src instrument (2k shots, d2 w1)
    if 'validation' not in probes:
        set_status(state='P2 validation')
        from src.simulation import simulate_ler_with_detection
        cs = np.asarray(encoder_forward(TRIO['d2'], CONNS7)[0])
        ref = simulate_ler_with_detection(cs, wk_noise(N7, D7, 1, S),
                                          n_shots=2000, seed=501)
        got = detection_postselect(cs, wk_noise(N7, D7, 1, S), 2000, seed=501)
        # different rng interleaving -> statistical agreement only
        tol = 3 * math.sqrt(0.25 / 2000)
        ok = (abs(ref['detected_fraction'] - (1 - got['acceptance'])) < tol
              and abs(ref['undetected_error_rate']
                      - got['undetected_error_rate']) < tol)
        probes['validation'] = {'ref_det': ref['detected_fraction'],
                                'got_det': 1 - got['acceptance'],
                                'ref_undet': ref['undetected_error_rate'],
                                'got_undet': got['undetected_error_rate'],
                                'agree_3sigma': bool(ok)}
        json.dump(res, open(OUT, 'w'), indent=2)
        print(f"[p2] validation: ref_det={ref['detected_fraction']:.4f} "
              f"got_det={1-got['acceptance']:.4f} agree={ok}", flush=True)
        if not ok:
            raise RuntimeError('batched detection instrument disagrees')
    for lab, path in TRIO.items():
        cs = np.asarray(encoder_forward(path, CONNS7)[0])
        for w in (1, 2, 3, 4):
            key = f'{lab}_w{w}'
            if key in probes:
                continue
            set_status(state='P2 probes', unit=key)
            t0 = time.time()
            r = detection_postselect(cs, wk_noise(N7, D7, w, S),
                                     100_000, seed=600 + w)
            r['wilson95_undet'] = wilson(r['undet_errors'], r['accepted'])
            r['wall_s'] = round(time.time() - t0, 1)
            probes[key] = r
            json.dump(res, open(OUT, 'w'), indent=2)
            print(f"[p2] {key}: acc={r['acceptance']:.4f} "
                  f"undet={r['undet_errors']}/{r['accepted']} "
                  f"({r['undetected_error_rate']:.2e}) "
                  f"[{r['wall_s']}s]", flush=True)


# ---------------------------------------------------------------- P3
def p3_qpanel(res):
    from src.decoders._common import encoder_forward
    from src.simulation import simulate_ler_with_correction_factored
    from src.errors import make_hardware_noise_fn
    if 't2_qpanel' not in res:
        res['t2_qpanel'] = {}
    qp = res['t2_qpanel']
    # conditioned probes (escalation targets per docstring)
    SHOTS = {'q4': {1: 4_000_000, 2: 2_000_000, 3: 2_000_000,
                    4: 2_000_000, 5: 100_000},
             'q5': {1: 4_000_000, 2: 2_000_000, 3: 2_000_000,
                    4: 2_000_000, 5: 2_000_000}}
    for qk, path in QCODES.items():
        d = int(qk[1])
        S = singles(d)
        cs = np.asarray(encoder_forward(path, CONNS5)[0])
        for w in (1, 2, 3, 4, 5):
            key = f'{qk}_w{w}'
            if key in qp:
                continue
            set_status(state='P3 conditioned', unit=key)
            t0 = time.time()
            shots = SHOTS[qk][w]
            rr = simulate_ler_with_correction_factored(
                cs, wk_noise(N5, d, w, S), S, N5, d,
                n_shots=shots, seed=3000 + w)
            k = int(round(float(rr['logical_error_rate']) * shots))
            qp[key] = {'k': k, 'n': shots, 'ler': k / shots,
                       'wilson95': wilson(k, shots),
                       'wall_s': round(time.time() - t0, 1)}
            json.dump(res, open(OUT, 'w'), indent=2)
            print(f"[p3] {key}: {k}/{shots} [{qp[key]['wall_s']}s]",
                  flush=True)
    # reconstruction + measurable cells + anchors
    from math import comb
    for qk in QCODES:
        rk = f'{qk}_reconstruction'
        if rk in qp:
            continue
        set_status(state='P3 reconstruction', unit=qk)
        cells = {}
        for p in PGRID:
            pt = ub = 0.0
            for w in (1, 2, 3, 4, 5):
                pw = comb(N5, w) * p ** w * (1 - p) ** (N5 - w)
                pr = qp[f'{qk}_w{w}']
                pt += pw * pr['ler']
                ub += pw * (pr['ler'] if pr['k'] > 0 else pr['wilson95'][1])
            cells[str(p)] = {'point': pt, 'ub': ub,
                             'exp_errors_at_1e7': pt * 1e7,
                             'measurable_1e7': pt * 1e7 >= 30}
        qp[rk] = cells
        json.dump(res, open(OUT, 'w'), indent=2)
        print(f'[p3] {rk}: ' + ' '.join(
            f"p{p}:{cells[str(p)]['point']:.2e}"
            f"{'*' if cells[str(p)]['measurable_1e7'] else ''}"
            for p in PGRID), flush=True)
    # direct anchors at measurable cells (<=1e7 shots)
    for qk, path in QCODES.items():
        d = int(qk[1])
        S = singles(d)
        cs = None
        for p in PGRID:
            if not qp[f'{qk}_reconstruction'][str(p)]['measurable_1e7']:
                continue
            key = f'{qk}_anchor_p{p}'
            if key in qp:
                continue
            set_status(state='P3 anchor', unit=key)
            if cs is None:
                cs = np.asarray(encoder_forward(path, CONNS5)[0])
            t0 = time.time()
            shots = 10_000_000
            rr = simulate_ler_with_correction_factored(
                cs, make_hardware_noise_fn(d, N5, p), S, N5, d,
                n_shots=shots, seed=4000)
            k = int(round(float(rr['logical_error_rate']) * shots))
            rec = qp[f'{qk}_reconstruction'][str(p)]['point']
            qp[key] = {'p': p, 'k': k, 'n': shots, 'ler': k / shots,
                       'wilson95': wilson(k, shots),
                       'validation_factor': (k / shots) / rec if rec else None,
                       'wall_s': round(time.time() - t0, 1)}
            json.dump(res, open(OUT, 'w'), indent=2)
            print(f"[p3] {key}: {k}/{shots} vf={qp[key]['validation_factor']}"
                  f" [{qp[key]['wall_s']}s]", flush=True)


# ---------------------------------------------------------------- P4
def p4_postselect_sweep(res):
    from src.decoders._common import encoder_forward
    if 't3_postselect' not in res:
        res['t3_postselect'] = {}
    ps = res['t3_postselect']
    from src.errors import make_hardware_noise_fn
    S = singles(D7)
    for lab, path in TRIO.items():
        cs = np.asarray(encoder_forward(path, CONNS7)[0])
        for p in (0.05, 0.08, 0.12):
            key = f'{lab}_p{p}'
            if key in ps:
                continue
            set_status(state='P4 sweep', unit=key)
            t0 = time.time()
            shots = 500_000
            while True:
                r = detection_postselect(
                    cs, make_hardware_noise_fn(D7, N7, p), shots, seed=5000)
                if r['undet_errors'] >= 30 or shots >= 5_000_000:
                    break
                shots = min(2 * shots, 5_000_000)
            r['wilson95_undet'] = wilson(r['undet_errors'], r['accepted'])
            r['p'] = p
            r['wall_s'] = round(time.time() - t0, 1)
            ps[key] = r
            json.dump(res, open(OUT, 'w'), indent=2)
            print(f"[p4] {key}: acc={r['acceptance']:.4f} "
                  f"undet={r['undet_errors']}/{r['accepted']} "
                  f"({r['undetected_error_rate']:.2e}) [{r['wall_s']}s]",
                  flush=True)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {
        'protocol': {
            'trio': {k: os.path.relpath(v, REPO) for k, v in TRIO.items()},
            'qcodes': {k: os.path.relpath(v, REPO) for k, v in QCODES.items()},
            'detection': f'declare-and-discard, threshold {DET_THRESH} '
                         '(Appendix-G convention), batched instrument '
                         'validated vs src.simulation',
            'pgrid': PGRID}}
    p1_residuals(res)
    p2_detection_probes(res)
    p3_qpanel(res)
    p4_postselect_sweep(res)
    set_status(state='done')
    print('[t23] DONE', flush=True)


if __name__ == '__main__':
    main()
