#!/usr/bin/env python3
"""Gate-study p-sweep escalation: escalate the two unresolved cells of the
r4_campaign/a1_queue.py entangler p-sweep: ms and zz at p = 0.02, each
capped at 15 errors in 5e6 shots. Same instrument as
r4_campaign/a1_queue.py Job 1 (identical code npz, noise fn, decoder
path); fresh independent seeds (8000-series), counts combined with the a1
cells by summation (independent MC).

Chunks of 5e6 shots until combined k >= 30 (max 4 chunks per cell).
Writes esc_psweep.json (incremental; resume anchor: completed chunks
skipped). Feeds the gate-study p-sweep of the paper.
Launch: nohup nice -n 19 python3 esc_psweep.py >> esc_psweep.log 2>&1 &
"""
import os, sys, json, math, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
os.chdir(REPO)
import src.jax_backend  # x64 first
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'esc_psweep.json')
A1 = os.path.join(REPO, 'results/campaigns/r4_campaign/a1_results.json')

D, N = 3, 6
P = 0.02
CHUNK, MAX_CHUNKS, MIN_ERR = 5_000_000, 4, 30
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]
TARGETS = ['ms', 'zz']


def wilson(k, n, z=1.96):
    ph = k / n
    dn = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / dn
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / dn
    return (max(0.0, c - h), c + h)


def main():
    from src.decoders._common import encoder_forward
    from src.simulation import simulate_ler_with_correction_factored
    from src.errors import make_hardware_noise_fn, qudit_hardware_error_basis
    singles = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]
    a1 = json.load(open(A1))
    res = json.load(open(OUT)) if os.path.exists(OUT) else {'cells': {}}
    for gate in TARGETS:
        unit = a1['units'][f'psweep_{gate}']
        base = unit['cells'][f'p{P}']
        assert not base['resolved']
        cell = res['cells'].setdefault(gate, {
            'gate': gate, 'seed': unit['seed'], 'npz': unit['npz'],
            'p': P, 'base_k': base['k'], 'base_n': base['n'],
            'chunks': []})
        cs = np.asarray(encoder_forward(unit['npz'], CONNS)[0])
        while True:
            k_tot = cell['base_k'] + sum(c['k'] for c in cell['chunks'])
            n_tot = cell['base_n'] + sum(c['n'] for c in cell['chunks'])
            if k_tot >= MIN_ERR or len(cell['chunks']) >= MAX_CHUNKS:
                break
            sd = 8000 + len(cell['chunks'])
            t0 = time.time()
            rr = simulate_ler_with_correction_factored(
                cs, make_hardware_noise_fn(D, N, P), singles, N, D,
                n_shots=CHUNK, seed=sd)
            k = int(round(float(rr['logical_error_rate']) * CHUNK))
            cell['chunks'].append({'seed': sd, 'k': k, 'n': CHUNK,
                                   'wall_s': round(time.time() - t0, 1)})
            json.dump(res, open(OUT, 'w'), indent=2)
            print(f"[esc] {gate} p={P} chunk sd={sd}: {k}/{CHUNK} "
                  f"[{cell['chunks'][-1]['wall_s']}s]", flush=True)
        k_tot = cell['base_k'] + sum(c['k'] for c in cell['chunks'])
        n_tot = cell['base_n'] + sum(c['n'] for c in cell['chunks'])
        cell['combined'] = {
            'k': k_tot, 'n': n_tot, 'ler': k_tot / n_tot,
            'wilson95': list(wilson(k_tot, n_tot)),
            'resolved': bool(k_tot >= MIN_ERR)}
        json.dump(res, open(OUT, 'w'), indent=2)
        print(f"[esc] {gate} p={P} COMBINED: {k_tot}/{n_tot} = "
              f"{k_tot/n_tot:.3e} resolved={k_tot >= MIN_ERR}", flush=True)
    print("[esc] done", flush=True)


if __name__ == '__main__':
    main()
