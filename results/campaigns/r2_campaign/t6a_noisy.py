#!/usr/bin/env python3
"""LER rows for the explicit [[5,1,3]]_Z3 encoder (t6a_build.py), including
the noisy-encoding variant (Sec. V E).

Rows (all p=0.05 convention, 1M shots, lookup decoder, Wilson 95%):
  A  compiled-direct   : codewords prepared by the compiled MS circuit
                         (rephased to the catalog phases), standard one-round
                         iid instrument, seed 2000. MUST reproduce the
                         codeword-direct row exactly (845/1M) - circuit gate.
  B  noisy-encoder 513 : Haar logical input |0000psi>, the 36-MS compiled
                         circuit with a discrete iid hardware error applied
                         to every qudit with prob p_step after each MS gate,
                         p_step = 1-(1-p)^(1/36) (marginal per-qudit error
                         budget across the encoder = p = 0.05); mid-circuit
                         errors PROPAGATE through the remaining encoder
                         Cliffords. Decode + score vs the ideal logical state.
  B' noisy-encoder head: same protocol for the central code A_L2_s100
                         (40 MS gates, p_step = 1-(1-p)^(1/40)); its gate
                         program is extracted from the npz and validated
                         against encoder_forward to <1e-10 first.
Codeword-direct reference rows already on record: central code 333/1M,
[[5,1,3]] 845/1M (t6bc_results.json / r1_campaign).
Writes t6a_noisy_results.json. Feeds Sec. V E of the paper.
"""
import os, sys, json, math, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)
import src.jax_backend  # x64 first
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 't6a_results.json')
D, N, P = 3, 5, 0.05
DIM = D ** N
SHOTS, SEED = 1_000_000, 2000
BATCH = 65536
HEADLINE_NPZ = os.path.join(REPO, 'results/campaigns/r1_campaign',
                            't1_codes', 'A_L2_s100.npz')
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    dn = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / dn
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / dn
    return (max(0.0, c - h), c + h)


def load_513_program():
    """t6a_circuit.json -> list of ('s', q, U3) / ('ms', q1, q2, U9)."""
    from src.gates import MS_gate
    circ = json.load(open(os.path.join(HERE, 't6a_circuit.json')))
    prog = []
    for g in circ['gates']:
        if g['type'] == 's':
            prog.append(('s', g['q'],
                         np.array(g['u_re']) + 1j * np.array(g['u_im'])))
        else:
            prog.append(('ms', g['q1'], g['q2'],
                         np.asarray(MS_gate(g['phi'], g['theta'],
                                            g['j'], g['k'], D))))
    return prog


def load_headline_program():
    """Extract the L=2 MS-ansatz gate program from the headline npz,
    mirroring jax_backend._apply_layer exactly."""
    from src.jax_backend import _xy_gate_jax, _z_gate_jax, _ms_gate_jax
    from src.gates import _build_ms_masks
    import jax.numpy as jnp
    dat = np.load(HEADLINE_NPZ, allow_pickle=True)
    params = np.asarray(dat['params'])          # (L, ppl)
    L = params.shape[0]
    masks = {t: tuple(jnp.array(m, dtype=jnp.complex128)
                      for m in _build_ms_masks(t, t + 1, D))
             for t in range(D - 1)}
    prog = []
    for l in range(L):
        lp = params[l]
        pi = 0
        for q in range(N):
            for _rep in range(2):
                for t in range(D - 1):
                    U = np.asarray(_xy_gate_jax(lp[pi], lp[pi + 1],
                                                t, t + 1, D))
                    prog.append(('s', q, U))
                    pi += 2
        for q1, q2 in CONNS:
            for t in range(D - 1):
                U = np.asarray(_ms_gate_jax(lp[pi], lp[pi + 1],
                                            t, t + 1, masks[t]))
                prog.append(('ms', q1, q2, U.reshape(D * D, D * D)))
                pi += 2
        for q in range(N):
            for t in range(D - 1):
                U = np.asarray(_z_gate_jax(lp[pi], t, t + 1, D))
                prog.append(('s', q, U))
                pi += 1
        assert pi == params.shape[1], (pi, params.shape)
    return prog


def apply_prog_batch(states, prog, rng=None, p_step=0.0, singles_bank=None):
    """states: (B, DIM). Applies the program; if p_step > 0, after each MS
    gate every qudit independently errs w.p. p_step (uniform over the 4
    hardware ops)."""
    B = states.shape[0]
    s = states.reshape((B,) + (D,) * N)
    def one(q, U):
        nonlocal s
        s = np.moveaxis(np.tensordot(U, s, axes=([1], [q + 1])), 0, q + 1)
    for g in prog:
        if g[0] == 's':
            one(g[1], g[2])
        else:
            _, q1, q2, U9 = g
            u = U9.reshape(D, D, D, D)
            s = np.moveaxis(np.tensordot(u, s, axes=([2, 3], [q1 + 1, q2 + 1])),
                            [0, 1], [q1 + 1, q2 + 1])
            if p_step > 0:
                for q in range(N):
                    idx = rng.random(B) < p_step
                    if not idx.any():
                        continue
                    ops = rng.integers(len(singles_bank), size=int(idx.sum()))
                    sub = s[idx]
                    for oi in range(len(singles_bank)):
                        m = ops == oi
                        if not m.any():
                            continue
                        which = np.where(idx)[0][m]
                        blk = s[which]
                        blk = np.moveaxis(
                            np.tensordot(singles_bank[oi], blk,
                                         axes=([1], [q + 1])), 0, q + 1)
                        s[which] = blk
    return s.reshape(B, DIM)


def decode_count(LOGICAL, FINAL, cs):
    """Lookup decode FINAL against code basis cs; count fc<0.5 vs LOGICAL."""
    from src.simulation import _apply_factored_op
    from src.errors import qudit_hardware_error_basis
    singles = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]
    ces = [np.asarray(cs)]
    for q in range(N):
        for E in singles:
            ces.append(np.stack([_apply_factored_op(cs[k], E, q, N, D)
                                 for k in range(cs.shape[0])]))
    CES = np.stack(ces)
    n_corr = CES.shape[0]
    M = np.conj(CES.reshape(n_corr * cs.shape[0], DIM))
    inner = (M @ FINAL.T).reshape(n_corr, cs.shape[0], FINAL.shape[0])
    best = np.argmax(np.sum(np.abs(inner) ** 2, axis=1), axis=0)
    coeffs = inner[best, :, np.arange(FINAL.shape[0])]
    projected = coeffs @ cs
    nrm = np.linalg.norm(projected, axis=1)
    ok = nrm > 1e-10
    projected[ok] /= nrm[ok, None]
    fc = np.abs(np.einsum('jd,jd->j', np.conj(LOGICAL), projected)) ** 2
    return int(np.sum(fc < 0.5)), float(np.sum(fc))


def noisy_row(prog, input_qudit, cs_ideal, n_ms, seed):
    from src.errors import qudit_hardware_error_basis
    singles = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]
    p_step = 1.0 - (1.0 - P) ** (1.0 / n_ms)
    rng = np.random.default_rng(seed)
    K = cs_ideal.shape[0]
    k_err, fsum = 0, 0.0
    stride = D ** (N - 1 - input_qudit)
    for start in range(0, SHOTS, BATCH):
        b = min(BATCH, SHOTS - start)
        alpha = rng.standard_normal((b, K)) + 1j * rng.standard_normal((b, K))
        alpha /= np.linalg.norm(alpha, axis=1, keepdims=True)
        inp = np.zeros((b, DIM), complex)
        for k in range(K):
            inp[:, k * stride] = alpha[:, k]
        FINAL = apply_prog_batch(inp, prog, rng=rng, p_step=p_step,
                                 singles_bank=singles)
        LOGICAL = alpha @ cs_ideal
        dk, fs = decode_count(LOGICAL, FINAL, cs_ideal)
        k_err += dk
        fsum += fs
    return {'k': k_err, 'n': SHOTS, 'ler': k_err / SHOTS,
            'wilson95': wilson(k_err, SHOTS), 'p_step': p_step,
            'n_ms': n_ms, 'mean_fidelity': fsum / SHOTS}


def main():
    from src.catalog import load_code
    from src.decoders._common import encoder_forward
    from src.simulation import simulate_ler_with_correction_factored
    from src.errors import qudit_hardware_error_basis, make_hardware_noise_fn
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    singles = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]
    cs = np.asarray(load_code('five_qudit_d3')['code_states'])

    prog513 = load_513_program()
    # compiled codewords, rephased to catalog phases
    inp = np.zeros((D, DIM), complex)
    for a in range(D):
        inp[a, a] = 1.0   # |0000a>
    comp = apply_prog_batch(inp, prog513)
    for a in range(D):
        ov = np.vdot(cs[a], comp[a])
        comp[a] *= (ov / abs(ov)).conjugate()
    dev = float(np.max(np.linalg.norm(comp - cs, axis=1)))

    if 'rowA_compiled_direct' not in res:
        t0 = time.time()
        rr = simulate_ler_with_correction_factored(
            comp, make_hardware_noise_fn(D, N, P), singles, N, D,
            n_shots=SHOTS, seed=SEED)
        k = int(round(rr['logical_error_rate'] * SHOTS))
        res['rowA_compiled_direct'] = {
            'k': k, 'n': SHOTS, 'ler': rr['logical_error_rate'],
            'wilson95': wilson(k, SHOTS), 'rephased_state_dev': dev,
            'expected_k': 845, 'exact_match': k == 845,
            'wall_s': round(time.time() - t0, 1)}
        json.dump(res, open(OUT, 'w'), indent=2)
        print('[t6a] rowA compiled-direct: k=%d (expect 845) dev=%.1e'
              % (k, dev), flush=True)

    if 'rowB_noisy_513' not in res:
        t0 = time.time()
        row = noisy_row(prog513, input_qudit=4, cs_ideal=cs, n_ms=36,
                        seed=SEED)
        row['wall_s'] = round(time.time() - t0, 1)
        res['rowB_noisy_513'] = row
        json.dump(res, open(OUT, 'w'), indent=2)
        print('[t6a] rowB noisy 513: k=%d ler=%.3e [%.0fs]'
              % (row['k'], row['ler'], row['wall_s']), flush=True)

    if 'rowBp_noisy_headline' not in res:
        progH = load_headline_program()
        csH = np.asarray(encoder_forward(HEADLINE_NPZ, CONNS)[0])
        # validate the numpy replay against encoder_forward
        inpH = np.zeros((D, DIM), complex)
        for k in range(D):
            inpH[k, k * D ** (N - 1)] = 1.0
        outH = apply_prog_batch(inpH, progH)
        devH = float(np.max(np.linalg.norm(outH - np.asarray(csH), axis=1)))
        assert devH < 1e-10, devH
        n_ms_H = sum(1 for g in progH if g[0] == 'ms')
        t0 = time.time()
        row = noisy_row(progH, input_qudit=0, cs_ideal=csH, n_ms=n_ms_H,
                        seed=SEED)
        row['wall_s'] = round(time.time() - t0, 1)
        row['replay_dev_vs_encoder_forward'] = devH
        res['rowBp_noisy_headline'] = row
        json.dump(res, open(OUT, 'w'), indent=2)
        print('[t6a] rowBp noisy headline: k=%d ler=%.3e (replay dev %.1e)'
              % (row['k'], row['ler'], devH), flush=True)

    print('[t6a] DONE', flush=True)


if __name__ == '__main__':
    main()
