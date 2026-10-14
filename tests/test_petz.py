"""sanity tests for Petz transpose recovery.

1. CP-trace-preservation (low-rank form): the codespace lies in supp(N(P_C)),
   so Σ_a R_a† R_a = Π_supp acts as identity on each codeword. (The dense
   ||Σ_a R_a† R_a − P_C||_F check is infeasible at d^n; this is the equivalent
   feasible condition.)
2. Knill-Laflamme limit: [[5,1,3]]_3 under a single weight-1 Pauli channel →
   Petz LER = 0 (perfect recovery of a detectable single error).
3. No-noise: identity channel → Petz LER = 0.
4. Self-consistency: Petz built against the twirled Meth Pauli weights,
   evaluated under the same flat-685 Meth Pauli channel on the n = 9 reference code (seed 1) →
   LER = 0 (single Pauli errors are recovered; single-error null).
5. Recorded observation (no pass/fail): twirl-built Petz evaluated under the
   real Meth Kraus channel at calibrated η=0.9296 on the n = 9 reference code (seed 1) — record LER.
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from src.decoders._common import apply_local_np, flat_group_offsets
from src.decoders.petz import (
    build_petz_recovery, simulate_ler_with_petz, support_projector_apply,
)
from src.decoders.priors import uniform_depolarizing_prior


# ─────────── small n=5 fixture for tests 1-3 ───────────

@pytest.fixture(scope="module")
def n5():
    from src.errors import ErrorModel, qudit_hardware_error_basis
    from src.catalog import load_code
    d, n_qudit, K = 3, 5, 3
    code_states = np.asarray(load_code('five_qudit_d3')['code_states'])
    E_full = ErrorModel(d=d, n_qudit=n_qudit, distance=3,
                        closed=True).build_grouped(verbose=False)
    offsets = flat_group_offsets(E_full)
    n_single = len(qudit_hardware_error_basis(d))
    return {'d': d, 'n_qudit': n_qudit, 'K': K, 'code_states': code_states,
            'E_full': E_full, 'offsets': offsets, 'n_single': n_single}


def test_cp_trace_preservation(n5):
    """Codespace ⊆ supp(N(P_C)): Π_supp acts as identity on each codeword."""
    w = uniform_depolarizing_prior(n5['E_full'], n5['n_qudit'], n5['d'],
                                    p=0.10, n_single=n5['n_single'])
    petz = build_petz_recovery(n5['code_states'], n5['E_full'], w,
                               n5['d'], n5['n_qudit'])
    max_resid = 0.0
    for k in range(n5['K']):
        psi = n5['code_states'][k]
        proj = support_projector_apply(petz, psi)
        max_resid = max(max_resid, float(np.linalg.norm(proj - psi)))
    print(f"\n  rank={petz['rank']}, n_kept={petz['n_kept']}, "
          f"max ||Π_supp ψ_k − ψ_k|| = {max_resid:.2e}")
    assert max_resid < 1e-6, (
        f"codespace not in supp(N(P_C)); residual {max_resid:.2e}")


def test_knill_laflamme_limit(n5):
    """Single weight-1 Pauli channel (Z_1 on qudit 2) → Petz LER = 0."""
    # Locate the (2,) group, e=0 (Z_1 is the first hardware op).
    gidx = next(i for i, g in enumerate(n5['E_full'])
                if tuple(g['wires']) == (2,))
    g = n5['E_full'][gidx]
    M = np.asarray(g['matrices'][0])
    wires = tuple(int(x) for x in g['wires'])
    inv = tuple(int(x) for x in g['inverse_perm'])
    flat_idx = int(n5['offsets'][gidx]) + 0

    n_total = int(n5['offsets'][-1])
    w = np.zeros(n_total)
    w[flat_idx] = 1.0
    petz = build_petz_recovery(n5['code_states'], n5['E_full'], w,
                               n5['d'], n5['n_qudit'])

    def noise_fn(state, rng):
        return apply_local_np(state, M, wires, inv, n5['d'], n5['n_qudit'])

    out = simulate_ler_with_petz(n5['code_states'], petz, noise_fn,
                                 n5['n_qudit'], n5['d'], n_shots=2000, seed=1)
    n_err = int(out.sum())
    print(f"\n  single weight-1 Pauli channel → {n_err}/2000 logical errors")
    assert n_err == 0, f"KL-limit Petz should perfectly recover: {n_err}/2000"


def test_no_noise(n5):
    """Identity channel (weight on identity only) → Petz LER = 0."""
    n_total = int(n5['offsets'][-1])
    w = np.zeros(n_total)
    w[0] = 1.0  # identity sentinel at flat 0
    petz = build_petz_recovery(n5['code_states'], n5['E_full'], w,
                               n5['d'], n5['n_qudit'])

    def noise_fn(state, rng):
        return state

    out = simulate_ler_with_petz(n5['code_states'], petz, noise_fn,
                                 n5['n_qudit'], n5['d'], n_shots=1000, seed=2)
    assert int(out.sum()) == 0, "no-noise Petz must give LER=0"


# ─────────── n=9 fixture for tests 4-5 ───────────

@pytest.fixture(scope="module")
def n9():
    import jax
    import jax.numpy as jnp
    from src.errors import ErrorModel
    from src.jax_backend import create_jax_encoder
    from src.decoders.priors import meth_pauli_prior

    d, n_qudit, K = 3, 9, 3
    connections = [[i, j] for i in range(n_qudit)
                    for j in range(i + 1, n_qudit)]
    ckpt = os.path.join(
        REPO, "results/campaigns/n9_reference/"
        "d3_n9_dist3_4L_meth_physical_seed1.npz")
    params = jnp.asarray(np.load(ckpt, allow_pickle=True)['params'])
    enc, _, _ = create_jax_encoder(n_qudit, d, connections=connections,
                                    use_scan=True)
    enc_vm = jax.jit(jax.vmap(enc, in_axes=(None, 0)))
    code_states = np.asarray(enc_vm(params, jnp.arange(K)))
    E_full = ErrorModel(d=d, n_qudit=n_qudit, distance=3,
                        closed=True).build_grouped(verbose=False)
    weights = meth_pauli_prior()
    petz = build_petz_recovery(code_states, E_full, weights, d, n_qudit)
    return {'d': d, 'n_qudit': n_qudit, 'K': K, 'code_states': code_states,
            'E_full': E_full, 'weights': weights, 'petz': petz,
            'connections': connections,
            'offsets': flat_group_offsets(E_full)}


def test_self_consistency_pauli(n9):
    """Twirl-built Petz, evaluated on the same flat-685 Meth Pauli channel,
    recovers single Pauli errors → LER = 0 (single-error null)."""
    from benchmark_ler_meth_pauli import make_flat_noise_fn
    noise_fn = make_flat_noise_fn(n9['weights'], n9['E_full'], n9['offsets'],
                                   n9['d'], n9['n_qudit'])
    out = simulate_ler_with_petz(n9['code_states'], n9['petz'], noise_fn,
                                 n9['n_qudit'], n9['d'], n_shots=500, seed=0)
    n_err = int(out.sum())
    print(f"\n  Petz on flat-685 Meth Pauli channel → {n_err}/500 logical errors")
    assert n_err == 0, (
        f"twirl-Petz on its own Pauli channel should recover: {n_err}/500")


def test_kraus_observation(n9):
    """Recorded observation (no pass/fail): twirl-built Petz under the real
    Meth Kraus channel at calibrated η=0.9296."""
    from src.simulation import make_correlated_dephasing_noise_fn
    from benchmark_ler_meth_kraus import build_gate_pairs
    gate_pairs = build_gate_pairs(n9['n_qudit'], n9['connections'])
    noise_fn = make_correlated_dephasing_noise_fn(
        n_qudits=n9['n_qudit'], d=n9['d'], gate_pairs=gate_pairs,
        eta=0.9296, n_max=5, noise_model='physical')
    out = simulate_ler_with_petz(n9['code_states'], n9['petz'], noise_fn,
                                 n9['n_qudit'], n9['d'], n_shots=500, seed=0)
    ler = float(out.mean())
    print(f"\n  [observation] twirl-Petz on real Meth Kraus η=0.9296: "
          f"LER = {ler:.4e} (n=500)")
    # No assertion — recorded for the §3 writeup.
