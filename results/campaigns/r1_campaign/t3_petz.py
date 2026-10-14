#!/usr/bin/env python3
"""Decoder-independence of the loss<->LER decoupling (measurement-only).

For every Table-II code (+ extras spanning the residual axis + the central
code once available): channel fidelity under Petz (transpose) recovery
for the weight-<=2 i.i.d. hardware channel at p=0.05, n=5 (dim 243, dense
tractable), computed EXACTLY (deterministic Kraus enumeration, no MC), plus
a 5k-shot MC cross-check on the full (untruncated) i.i.d. channel.

Channel: identity + 20 singles + 160 cross-qudit pairs over the hw closure
grouped basis (`uniform_depolarizing_prior`, n_single=4), renormalized to
sum 1 (conditioning on weight<=2; dropped w>=3 mass recorded). Petz built
per code against that channel (src/decoders/petz.py machinery).

Exact fidelity: with A_i = sqrt(w_i) E_i, R_j = P_C A_j^dag N(P_C)^{-1/2},
  F_e  = (1/K^2) sum_{i,j} |sum_k <K_psi[j,k]| N^{-1/2} |K_psi[i,k]>|^2
  F_avg = (K F_e + 1)/(K + 1)   (Haar average over logical states)
Petz-recovered infidelity := 1 - F_avg.

Writes t3_results.json (resumable per code); the figure is rendered later
from this JSON by scripts/make_figures.py.
"""
import os, sys, json, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import src.jax_backend  # x64 first (must precede any jnp array creation)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 't3_results.json')
D, N, P = 3, 5, 0.05
A2A = [[i, j] for i in range(N) for j in range(i + 1, N)]
BPR = 'results/campaigns'

# (name, npz path or catalog key, documented 1M-shot lookup iid LER @p=0.05
#  from earlier 1M-shot measurements / loss_ler_diag t1g / c2 300k], source note)
CODES = [
    ('published_ring_L10', 'catalog:qutrit_d3', 4.28e-4, 'S3.6 1M'),
    ('stageH_stalled_L16', f'{BPR}/stage_h/race_fullbatch_d3_n5_dist3.npz',
     5.30e-4, 'S3.6 1M'),
    ('deep_L16', f'{BPR}/headline_p05_Lsweep/n5_L16.npz', 1.18e-3, 'S3.6 1M'),
    ('deep_L20', f'{BPR}/headline_p05_Lsweep/n5_L20.npz', 1.14e-3, 'S3.6 1M'),
    ('deep_L24', f'{BPR}/headline_p05_Lsweep/n5_L24.npz', 1.12e-3, 'S3.6 1M'),
    ('deep_L28', f'{BPR}/headline_p05_Lsweep/n5_L28.npz', 1.19e-3, 'S3.6 1M'),
    ('hw_L16', f'{BPR}/loss_ler_diag/n5_hw_L16.npz', 4.97e-4, 'T1c 1M'),
    ('hw_L24', f'{BPR}/loss_ler_diag/n5_hw_L24.npz', 5.19e-4, 'T1c 1M'),
    ('full_s101_stalled', f'{BPR}/loss_ler_diag/t1g/full_s101.npz',
     2.46e-4, 'T1g 1M'),
    ('weighted_c2_s100', f'{BPR}/sampling_budget_v2/c2/wimp02_s100.npz',
     None, 'c2 300k (read from c2_results.json)'),
    ('benchmark_513', 'catalog:five_qudit_d3', 8.45e-4, 'S3.6 1M'),
]


def get_states(spec):
    from src.catalog import load_code
    from src.decoders._common import encoder_forward
    if spec.startswith('catalog:'):
        return np.asarray(load_code(spec.split(':', 1)[1])['code_states'])
    data = np.load(spec, allow_pickle=True)
    if 'connections' in data.files:
        conns = [list(map(int, c)) for c in data['connections']]
    else:
        conns = A2A
    return np.asarray(encoder_forward(spec, conns)[0])


def exact_petz_fidelity(petz):
    """F_e and F_avg of the Petz-recovered truncated channel, exact."""
    K_psi = petz['K_psi']                      # (n_kept, K, dim)
    n_kept, K, dim = K_psi.shape
    # Q[i,k,:] = sqrt(s_inv) * (Uh @ K_psi[i,k])
    Q = np.einsum('r,rd,ikd->ikr', np.sqrt(petz['s_inv']), petz['Uh'],
                  K_psi, optimize=True)
    # T[j,i] = (1/K) sum_k <Q[j,k]|Q[i,k]>
    T = np.einsum('jkr,ikr->ji', Q.conj(), Q, optimize=True) / K
    F_e = float(np.sum(np.abs(T) ** 2))
    F_avg = (K * F_e + 1.0) / (K + 1.0)
    return F_e, F_avg


def mc_crosscheck(cs, petz, shots=5000, seed=777):
    """Mean fidelity + LER under Petz on the FULL (untruncated) iid channel."""
    from src.errors import make_hardware_noise_fn, qudit_hardware_error_basis
    from src.decoders.petz import fidelity_under_petz
    noise_fn = make_hardware_noise_fn(D, N, P)
    rng = np.random.default_rng(seed)
    K, dim = cs.shape
    fids = np.empty(shots)
    for t in range(shots):
        a = rng.normal(size=K) + 1j * rng.normal(size=K)
        a /= np.linalg.norm(a)
        psi = a @ cs
        noisy = noise_fn(psi, rng)
        fids[t] = fidelity_under_petz(petz, cs, a, noisy)
    return {'mean_fidelity': float(fids.mean()),
            'ler_thresh0.5': float((fids < 0.5).mean()), 'shots': shots}


def main():
    from src.errors import ErrorModel
    from src.decoders.priors import uniform_depolarizing_prior
    from src.decoders.petz import build_petz_recovery
    from src.jax_backend import create_jax_loss_vmap_weighted_v2
    import jax.numpy as jnp

    res = json.load(open(OUT)) if os.path.exists(OUT) else {'codes': {}}

    # channel weights on the hw closure grouped basis
    hw_grouped = ErrorModel(d=D, n_qudit=N, distance=3, closed=True,
                            basis=None).build_grouped(verbose=False)
    w_raw = uniform_depolarizing_prior(hw_grouped, N, D, P, n_single=4)
    covered = float(w_raw.sum())
    w = w_raw / covered
    res['channel'] = {
        'desc': 'iid hardware channel p=0.05 truncated to weight<=2, '
                'renormalized (conditioned on w<=2)',
        'covered_mass': covered, 'dropped_wge3_mass': 1.0 - covered,
        'n_nonzero_ops': int((w > 0).sum())}

    # full-basis residual evaluator (state-level, closure full basis)
    full_grouped = ErrorModel(d=D, n_qudit=N, distance=3, closed=True,
                              basis='full').build_grouped(verbose=False)
    core = create_jax_loss_vmap_weighted_v2(full_grouped, D, D, N, 3)
    ones = tuple(jnp.ones((g['matrices'].shape[0],), dtype=jnp.float64)
                 for g in full_grouped)

    # optional: the central code, once t1_queue.py has selected it
    codes = list(CODES)
    t1p = os.path.join(HERE, 't1_results.json')
    if os.path.exists(t1p):
        t1 = json.load(open(t1p))
        if 'headline' in t1:
            codes.append(('t1_headline', t1['headline']['selected_npz'],
                          t1['headline']['measurement']['t1_headline']['ler'],
                          'T1 part C 2e6'))

    c2p = f'{BPR}/sampling_budget_v2/c2_results.json'
    c2ler = None
    if os.path.exists(c2p):
        c2 = json.load(open(c2p))
        c2ler = c2['units'].get('wimp02_s100', {}).get(
            'ler_iid_p05', {}).get('ler')

    for name, spec, doc_ler, src in codes:
        if name in res['codes']:
            continue
        t0 = time.time()
        cs = get_states(spec)
        residual = float(core(jnp.asarray(cs), ones))
        petz = build_petz_recovery(cs, hw_grouped, w, D, N)
        F_e, F_avg = exact_petz_fidelity(petz)
        mc = mc_crosscheck(cs, petz)
        row = {'spec': spec, 'full_basis_residual': residual,
               'petz_F_e': F_e, 'petz_F_avg': F_avg,
               'petz_infidelity_1mFavg': 1.0 - F_avg,
               'petz_infidelity_1mFe': 1.0 - F_e,
               'rank': petz['rank'], 'n_kept': petz['n_kept'],
               'mc_full_channel': mc,
               'lookup_iid_ler_doc': (c2ler if name == 'weighted_c2_s100'
                                      else doc_ler),
               'lookup_ler_source': src,
               'wall_s': round(time.time() - t0, 1)}
        res['codes'][name] = row
        json.dump(res, open(OUT, 'w'), indent=2)
        print(f"[t3] {name}: residual={residual:.3e} "
              f"1-F_avg={row['petz_infidelity_1mFavg']:.3e} "
              f"mcF={mc['mean_fidelity']:.5f} [{row['wall_s']}s]", flush=True)

    print('[t3] DONE', flush=True)


if __name__ == '__main__':
    main()
