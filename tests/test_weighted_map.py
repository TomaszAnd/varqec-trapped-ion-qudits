"""sanity tests for the weighted MAP decoder.

1. Uniform-prior reduction: MAP with uniform prior over the lookup-equivalent
   correction set ({I, 9×4 single-qudit originals}) reproduces the unweighted
   lookup decoder LER within bootstrap CI.
2. Identity-prior reduction: MAP with all weight on the identity correction
   equals the projection-only decoder LER within bootstrap CI.
3. Weight-2 coverage: MAP with the full weight-≤2 set recovers a deterministic
   weight-2 error (LER=0), and the correction set contains weight-2 entries.
   (Documents that the unweighted lookup ALSO recovers single weight-2 errors
   via projection — the distance-3 projection-recoverability property — so
   this is a mechanical-correctness test, not a "MAP beats lookup" test.)
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from src.decoders._common import apply_local_np
from benchmark_ler_meth_pauli import (
    simulate_per_shot_factored, bootstrap_ler_ci,
)
from src.decoders.weighted_map import (
    build_weighted_correction_set, simulate_ler_with_weighted_map,
)
from src.decoders.priors import uniform_depolarizing_prior  # noqa: F401


@pytest.fixture(scope="module")
def setup():
    import jax
    import jax.numpy as jnp
    from src.errors import ErrorModel, qudit_hardware_error_basis
    from src.jax_backend import create_jax_encoder

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

    model = ErrorModel(d=d, n_qudit=n_qudit, distance=3, closed=True)
    E_full = model.build_grouped(verbose=False)
    single_errors = [np.asarray(E, dtype=complex)
                     for E in qudit_hardware_error_basis(d)]
    return {'d': d, 'n_qudit': n_qudit, 'K': K, 'code_states': code_states,
            'E_full': E_full, 'single_errors': single_errors,
            'n_single': len(single_errors)}


def _build_lookup_equivalent_set(E_full, n_qudit, d, n_single):
    """Correction set matching simulate_per_shot_factored: {I} ∪ {9×4 originals}.
    Single-qudit groups' first n_single entries are the originals
    (= qudit_hardware_error_basis order), uniform prior."""
    corrections = []
    for g in E_full:
        wires = tuple(int(w) for w in g['wires'])
        inv = tuple(int(p) for p in g['inverse_perm'])
        mats = np.asarray(g['matrices'])
        if len(wires) == 0:
            corrections.append((wires, mats[0].conj().T, inv, 1.0))
        elif len(wires) == 1:
            for e in range(min(n_single, mats.shape[0])):
                corrections.append((wires, mats[e].conj().T, inv, 1.0))
        # skip weight-2 to match lookup
    return corrections


def test_uniform_prior_reduction(setup):
    """MAP-uniform over the lookup-equivalent set ≈ unweighted lookup."""
    from src.simulation import make_pauli_depolarizing_noise_fn
    p = 0.30  # high enough for measurable LER
    noise_fn = make_pauli_depolarizing_noise_fn(
        single_qudit_errors=setup['single_errors'],
        n_qudits=setup['n_qudit'], dim_qudit=setup['d'], p=p)

    corr = _build_lookup_equivalent_set(
        setup['E_full'], setup['n_qudit'], setup['d'], setup['n_single'])
    out_map = simulate_ler_with_weighted_map(
        setup['code_states'], noise_fn, corr,
        setup['n_qudit'], setup['d'], n_shots=2000, seed=7)
    ler_map = float(out_map.mean())

    # Unweighted lookup on the SAME noise realizations (same seed)
    noise_fn2 = make_pauli_depolarizing_noise_fn(
        single_qudit_errors=setup['single_errors'],
        n_qudits=setup['n_qudit'], dim_qudit=setup['d'], p=p)
    out_look = simulate_per_shot_factored(
        setup['code_states'], noise_fn2, setup['single_errors'],
        setup['n_qudit'], setup['d'], n_shots=2000, seed=7)
    ler_look = float(out_look.mean())

    s_map = np.sqrt(max(ler_map * (1 - ler_map), 1e-12) / 2000)
    s_look = np.sqrt(max(ler_look * (1 - ler_look), 1e-12) / 2000)
    comb_sigma = np.sqrt(s_map ** 2 + s_look ** 2)
    diff = abs(ler_map - ler_look)
    print(f"\n  MAP-uniform LER = {ler_map:.4e}, lookup LER = {ler_look:.4e}, "
          f"diff = {diff:.4e}, 3σ = {3*comb_sigma:.4e}")
    assert diff <= 3 * comb_sigma + 5e-3, (
        f"MAP-uniform ({ler_map}) should reduce to lookup ({ler_look})")


def test_identity_prior_reduction(setup):
    """MAP with all weight on identity == projection-only decoder."""
    from src.simulation import make_pauli_depolarizing_noise_fn
    p = 0.30
    noise_fn = make_pauli_depolarizing_noise_fn(
        single_qudit_errors=setup['single_errors'],
        n_qudits=setup['n_qudit'], dim_qudit=setup['d'], p=p)

    # Identity-only correction set (weight 1 on identity, nothing else)
    id_group = setup['E_full'][0]
    assert len(id_group['wires']) == 0, "expected identity sentinel first"
    corr = [((), np.asarray(id_group['matrices'][0]).conj().T,
             tuple(int(x) for x in id_group['inverse_perm']), 1.0)]
    out_map = simulate_ler_with_weighted_map(
        setup['code_states'], noise_fn, corr,
        setup['n_qudit'], setup['d'], n_shots=2000, seed=11)
    ler_map = float(out_map.mean())

    # Projection-only reference: lookup with empty single_errors → only the
    # identity init + projection.
    noise_fn2 = make_pauli_depolarizing_noise_fn(
        single_qudit_errors=setup['single_errors'],
        n_qudits=setup['n_qudit'], dim_qudit=setup['d'], p=p)
    out_proj = simulate_per_shot_factored(
        setup['code_states'], noise_fn2, [],
        setup['n_qudit'], setup['d'], n_shots=2000, seed=11)
    ler_proj = float(out_proj.mean())

    s1 = np.sqrt(max(ler_map * (1 - ler_map), 1e-12) / 2000)
    s2 = np.sqrt(max(ler_proj * (1 - ler_proj), 1e-12) / 2000)
    comb_sigma = np.sqrt(s1 ** 2 + s2 ** 2)
    diff = abs(ler_map - ler_proj)
    print(f"\n  MAP-identity LER = {ler_map:.4e}, projection-only LER = "
          f"{ler_proj:.4e}, diff = {diff:.4e}, 3σ = {3*comb_sigma:.4e}")
    assert diff <= 3 * comb_sigma + 5e-3, (
        f"MAP-identity ({ler_map}) should equal projection-only ({ler_proj})")


def test_weight2_coverage(setup):
    """MAP with the full weight-≤2 set recovers a deterministic weight-2 error.

    NB: a single weight-2 error on a distance-3 code is also projection-
    recoverable (KL: P_C·E|ψ_L⟩ ∝ |ψ_L⟩), so the unweighted lookup recovers
    it too. This test verifies (i) the correction set contains weight-2 ops
    and (ii) MAP achieves LER=0 — mechanical correctness of the extension."""
    E_full = setup['E_full']
    d, n_qudit = setup['d'], setup['n_qudit']

    # Find the first weight-2 group and use its e=0 op as deterministic noise.
    g2 = next(g for g in E_full if len(g['wires']) == 2)
    wires = tuple(int(w) for w in g2['wires'])
    inv = tuple(int(p) for p in g2['inverse_perm'])
    M2 = np.asarray(g2['matrices'][0])

    def fixed_w2_noise(state, rng):
        return apply_local_np(state, M2, wires, inv, d, n_qudit)

    # Full weight-≤2 correction set with a prior favoring the true op.
    # Use a uniform-ish prior (all ones) so the weight-2 op is a candidate.
    n_total = sum(int(g['matrices'].shape[0]) for g in E_full)
    prior = np.ones(n_total)
    corr = build_weighted_correction_set(E_full, prior, n_qudit, d, max_weight=2)
    n_w2 = sum(1 for (w, _, _, _) in corr if len(w) == 2)
    assert n_w2 > 0, "weight-2 corrections missing from the set"

    out = simulate_ler_with_weighted_map(
        setup['code_states'], fixed_w2_noise, corr,
        n_qudit, d, n_shots=500, seed=3)
    n_err = int(out.sum())
    print(f"\n  weight-2 corrections in set: {n_w2}; "
          f"deterministic weight-2 noise → {n_err}/500 logical errors")
    assert n_err == 0, (
        f"MAP failed to recover a deterministic weight-2 error: "
        f"{n_err}/500 logical errors")
