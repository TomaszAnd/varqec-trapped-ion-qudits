#!/usr/bin/env python3
"""Retrain n=5 on the HARDWARE closure basis (the basis the i.i.d. LER
benchmark draws from), mirroring the L-sweep protocol exactly: best-of-8,
3000 steps, all-to-all, MS, full-batch uniform, target 1e-9 (never certifies;
keeps the loss-best uncertified theta, same as the L-sweep).

Then measure i.i.d. LER @ p=0.05 at 1M shots and report both losses.
noise="hw" != "full" -> ErrorModel(basis=None) = hardware basis+closure
(verified by reading build_race; NOT dephasing-only).
Writes t1c_results.json and n5_hw_L{16,24}.npz (Table IV rows).
"""
import os, sys, json, math, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import src.jax_backend  # x64 first
import numpy as np
import jax.numpy as jnp
from src.seed_race import run_race, build_race
from src.errors import qudit_hardware_error_basis, make_hardware_noise_fn
from src.simulation import simulate_ler_with_correction_factored
from src.decoders._common import encoder_forward

D, N, P = 3, 5, 0.05
SCRATCH = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(REPO, 'results', 'campaigns', 'loss_ler_diag')
OUT = os.path.join(SCRATCH, 't1c_results.json')
SINGLES = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]


def wilson(k, n, z=1.96):
    ph = k / n
    dn = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / dn
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / dn
    return (max(0.0, c - h), c + h)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    conns = [[i, j] for i in range(N) for j in range(i + 1, N)]
    log = {'config': 'hw-basis retrain, best-of-8, 3000 steps, a2a, MS, '
                     'full batch uniform, target 1e-9', 'per_L': []}

    # full-basis loss evaluator for cross-reporting
    _, full_loss_16, *_ = build_race(D, N, 3, 16, noise='full')

    for L in (16, 24):
        out = os.path.join(OUTDIR, f'n5_hw_L{L}.npz')
        t = time.time()
        res = run_race(d=D, n=N, distance=3, layers=L, num_seeds=8, steps=3000,
                       target_loss=1e-9, sample_frac=1.0, sampler='uniform',
                       trigger_margin=0.25, trigger_patience=5, out=out,
                       first_seed=0, verbose=False, connectivity='all-to-all',
                       entangler='ms', noise='hw')
        hw_loss = float(res['result']['full_loss'])
        theta = jnp.asarray(res['result']['theta'])
        floss = float(full_loss_16(theta))
        cs, *_ = encoder_forward(out, conns)
        r = simulate_ler_with_correction_factored(
            cs, make_hardware_noise_fn(D, N, P), SINGLES, N, D,
            n_shots=1_000_000, seed=42)
        ler = float(r['logical_error_rate'])
        k = int(round(ler * 1_000_000))
        row = {'L': L, 'hw_basis_loss': hw_loss, 'full_basis_loss': floss,
               'seed': int(res['result']['seed']),
               'ler_1M_p05': ler, 'k': k, 'wilson95': wilson(k, 1_000_000),
               'wall_s': round(time.time() - t, 1), 'npz': out}
        log['per_L'].append(row)
        json.dump(log, open(OUT, 'w'), indent=2)
        print(f"L={L}: hw_loss={hw_loss:.3e} full_loss={floss:.3e} "
              f"LER={ler:.4e} ({k}/1M) [{row['wall_s']}s]", flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
