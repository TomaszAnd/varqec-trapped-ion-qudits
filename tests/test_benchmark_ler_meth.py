"""sanity tests for the Meth-channel LER benchmarks.

1. No-noise limit, pipeline (b): η=1.0 → LER=0.
2. No-noise limit, pipeline (a): all weight on identity → LER=0.
3. Depolarizing equivalence: flat 685-vector reproducing uniform
   per-qudit Pauli depolarizing at p=0.05 must agree with
   `src.simulation.make_pauli_depolarizing_noise_fn` on the same code
   within 3σ bootstrap. Catches sampler bugs in pipeline (a).
4. Twirl-Kraus consistency observation (not a hard pass/fail): records
   the ratio of pipeline (a) at α=1 to pipeline (b) at η=0.9296 on
   the n = 9 reference code (seed 1), flags if > factor of 3 in either direction.
"""
from __future__ import annotations
import os
import sys
import warnings

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

# Module imports from benchmark scripts
from benchmark_ler_meth_pauli import (
    make_flat_noise_fn, simulate_per_shot_factored,
    flat_group_offsets,
)
from benchmark_ler_meth_kraus import build_gate_pairs


@pytest.fixture(scope="module")
def setup():
    """Build code_states and structural basis once for all tests."""
    import jax
    import jax.numpy as jnp
    from src.errors import ErrorModel, qudit_hardware_error_basis
    from src.jax_backend import create_jax_encoder

    d, n_qudit, K = 3, 9, 3
    connections = [[i, j] for i in range(n_qudit)
                    for j in range(i + 1, n_qudit)]
    ckpt = os.path.join(
        REPO,
        "results/campaigns/n9_reference/"
        "d3_n9_dist3_4L_meth_physical_seed1.npz")
    data = np.load(ckpt, allow_pickle=True)
    params = jnp.asarray(data['params'])

    enc, _, _ = create_jax_encoder(n_qudit, d, connections=connections,
                                    use_scan=True)
    enc_vm = jax.jit(jax.vmap(enc, in_axes=(None, 0)))
    code_states = np.asarray(enc_vm(params, jnp.arange(K)))

    model = ErrorModel(d=d, n_qudit=n_qudit, distance=3, closed=True)
    E_full = model.build_grouped(verbose=False)
    group_starts = flat_group_offsets(E_full)
    single_errors = [np.asarray(E, dtype=complex)
                     for E in qudit_hardware_error_basis(d)]
    gate_pairs = build_gate_pairs(n_qudit, connections)

    return {
        'd': d, 'n_qudit': n_qudit, 'K': K,
        'connections': connections,
        'code_states': code_states,
        'E_full': E_full, 'group_starts': group_starts,
        'single_errors': single_errors,
        'gate_pairs': gate_pairs,
    }


# ─────────── Test 1: no-noise Kraus ───────────

def test_no_noise_kraus(setup):
    """η=1.0 makes every Gaussian-phase Kraus diagonal degenerate to identity
    (n=0 is uniform 1s, n≥1 vanishes), so the channel is the identity. LER=0."""
    from src.simulation import make_correlated_dephasing_noise_fn

    noise_fn = make_correlated_dephasing_noise_fn(
        n_qudits=setup['n_qudit'], d=setup['d'],
        gate_pairs=setup['gate_pairs'],
        eta=1.0, n_max=5, noise_model='physical')
    outcomes = simulate_per_shot_factored(
        setup['code_states'], noise_fn, setup['single_errors'],
        setup['n_qudit'], setup['d'], n_shots=1000, seed=0)
    n_err = int(outcomes.sum())
    assert n_err == 0, f"Expected 0 logical errors at η=1.0, got {n_err}/1000"


# ─────────── Test 2: no-noise Pauli (flat-vector all-identity) ───────────

def test_no_noise_pauli(setup):
    """Flat 685-vector with weight 1 on identity (flat=0) and 0 elsewhere.
    Every shot draws identity; noise_fn does nothing. LER=0."""
    n_total = sum(int(g['matrices'].shape[0]) for g in setup['E_full'])
    w = np.zeros(n_total)
    w[0] = 1.0  # identity is at flat index 0 per the storage convention

    noise_fn = make_flat_noise_fn(
        w, setup['E_full'], setup['group_starts'],
        setup['d'], setup['n_qudit'])
    outcomes = simulate_per_shot_factored(
        setup['code_states'], noise_fn, setup['single_errors'],
        setup['n_qudit'], setup['d'], n_shots=1000, seed=0)
    n_err = int(outcomes.sum())
    assert n_err == 0, f"Expected 0 logical errors at all-identity, got {n_err}/1000"


# ─────────── Test 3: depolarizing equivalence (load-bearing for pipeline (a)) ───────────

def _build_uniform_depolarizing_flat_weights(E_full: list, n_qudit: int,
                                              n_single: int, p: float
                                              ) -> np.ndarray:
    """Construct a flat 685-vector that exactly reproduces uniform per-qudit
    Pauli depolarizing at strength `p` on the weight-≤2 sector.

    `make_pauli_depolarizing_noise_fn` semantics: each qudit independently,
    with prob p apply one of n_single single-qudit Paulis uniformly, with
    prob (1-p) do nothing. The induced multi-qudit distribution on the
    closure basis:
      - Identity: (1-p)^n_qudit.
      - Single-qudit (q, e_idx ∈ [0..n_single-1]): (1-p)^(n_qudit-1)·p·(1/n_single).
      - Same-qudit closure (q, e_idx ∈ [n_single..]): 0 — not produced by the
        per-qudit single-Pauli channel.
      - Two-qudit (q1,q2) products of single Paulis: (1-p)^(n_qudit-2)·(p/n_single)².
    Mass on weight-≥3 ops not in the closure basis is missing; the
    truncation error is ~ sum_{k≥3} C(n,k)p^k(1-p)^(n-k) and rescales the
    flat-vector LER by a small factor after renormalization.
    """
    n_total = sum(int(g['matrices'].shape[0]) for g in E_full)
    w = np.zeros(n_total)
    flat = 0
    for g in E_full:
        wires = g['wires']
        n_g = int(g['matrices'].shape[0])
        if len(wires) == 0:
            w[flat] = (1 - p) ** n_qudit
        elif len(wires) == 1:
            for e in range(min(n_single, n_g)):
                w[flat + e] = (1 - p) ** (n_qudit - 1) * p * (1.0 / n_single)
        else:
            # All entries in (q1, q2) groups are E1 ⊗ E2 with E1, E2 ∈ single_errors
            for e in range(n_g):
                w[flat + e] = (1 - p) ** (n_qudit - 2) * (p / n_single) ** 2
        flat += n_g
    return w


def test_depolarizing_equivalence(setup):
    """Pipeline (a) on a flat-vector emulating uniform Pauli depolarizing at p=0.05
    should match `make_pauli_depolarizing_noise_fn(p=0.05)` on the same code within
    3σ + truncation on n_shots=5000.

    Truncation: the flat-vector channel covers weight-≤2 events only, missing
    `Σ_{k≥3} C(n,k)·p^k·(1-p)^(n-k)` of the reference channel's mass. Those
    missing events would mostly be uncorrectable (weight ≥3 lies outside the
    weight-≤1 correction set), so pipeline (a) systematically UNDER-counts
    LER by ~truncation. The tolerance below accounts for this.
    """
    from math import comb
    from src.simulation import make_pauli_depolarizing_noise_fn

    p = 0.05
    n_single = len(setup['single_errors'])
    n_qudit = setup['n_qudit']
    w = _build_uniform_depolarizing_flat_weights(
        setup['E_full'], n_qudit, n_single, p)

    # Truncation: weight-3+ mass missing from the flat-vector channel.
    actual_truncation = 1.0 - w.sum()
    expected_truncation = sum(
        comb(n_qudit, k) * (p ** k) * ((1 - p) ** (n_qudit - k))
        for k in range(3, n_qudit + 1))
    assert abs(actual_truncation - expected_truncation) < 1e-12, (
        f"Flat-vector weight construction inconsistent with binomial: "
        f"actual={actual_truncation:.6e} vs expected={expected_truncation:.6e}")
    print(f"\n  truncation (weight≥3 mass): {actual_truncation:.4e}")

    # Pipeline (a) on the flat-vector
    noise_fn_pauli = make_flat_noise_fn(
        w, setup['E_full'], setup['group_starts'],
        setup['d'], n_qudit)
    out_a = simulate_per_shot_factored(
        setup['code_states'], noise_fn_pauli, setup['single_errors'],
        n_qudit, setup['d'], n_shots=5000, seed=42)
    ler_a = float(out_a.mean())

    # Reference channel via src.simulation
    noise_fn_ref = make_pauli_depolarizing_noise_fn(
        single_qudit_errors=setup['single_errors'],
        n_qudits=n_qudit, dim_qudit=setup['d'], p=p)
    out_ref = simulate_per_shot_factored(
        setup['code_states'], noise_fn_ref, setup['single_errors'],
        n_qudit, setup['d'], n_shots=5000, seed=42)
    ler_ref = float(out_ref.mean())

    sigma_a = np.sqrt(max(ler_a * (1 - ler_a), 1e-12) / 5000)
    sigma_ref = np.sqrt(max(ler_ref * (1 - ler_ref), 1e-12) / 5000)
    combined_sigma = np.sqrt(sigma_a ** 2 + sigma_ref ** 2)
    diff = abs(ler_a - ler_ref)

    print(f"  pipeline(a) flat-685 LER = {ler_a:.4e}")
    print(f"  reference  Pauli   LER  = {ler_ref:.4e}")
    print(f"  diff = {diff:.4e}, combined 1σ = {combined_sigma:.4e}")

    # Tolerance: 3σ from random MC + truncation systematic upper bound
    tol = 3 * combined_sigma + actual_truncation
    print(f"  tolerance (3σ + truncation) = {tol:.4e}")
    assert diff <= tol, (
        f"Depolarizing equivalence failed: LER(a)={ler_a:.4e} vs "
        f"LER(ref)={ler_ref:.4e}, diff={diff:.4e} > tol={tol:.4e}")


# ─────────── Test 4: twirl-Kraus ratio observation ───────────

def test_twirl_kraus_ratio(setup):
    """Recorded observation: ratio of pipeline (a) α=1 to pipeline (b) η=0.9296
    on the n = 9 reference code (seed 1). Not a hard pass/fail; warn if ratio > 3."""
    from src.simulation import make_correlated_dephasing_noise_fn

    # Pipeline (a) at α=1: actual Meth weights, no rescaling
    W = np.load(os.path.join(
        REPO, "results/campaigns/n9_reference/meth_pauli_weights.npz"),
        allow_pickle=True)
    weights = np.asarray(W['weights'])
    noise_fn_a = make_flat_noise_fn(
        weights, setup['E_full'], setup['group_starts'],
        setup['d'], setup['n_qudit'])
    out_a = simulate_per_shot_factored(
        setup['code_states'], noise_fn_a, setup['single_errors'],
        setup['n_qudit'], setup['d'], n_shots=1000, seed=0)
    ler_a = float(out_a.mean())

    # Pipeline (b) at calibrated η
    noise_fn_b = make_correlated_dephasing_noise_fn(
        n_qudits=setup['n_qudit'], d=setup['d'],
        gate_pairs=setup['gate_pairs'],
        eta=0.9296, n_max=5, noise_model='physical')
    out_b = simulate_per_shot_factored(
        setup['code_states'], noise_fn_b, setup['single_errors'],
        setup['n_qudit'], setup['d'], n_shots=1000, seed=0)
    ler_b = float(out_b.mean())

    if ler_a > 0 and ler_b > 0:
        ratio = max(ler_a, ler_b) / min(ler_a, ler_b)
    elif ler_a == 0 and ler_b == 0:
        ratio = 1.0
    else:
        ratio = float('inf')

    print(f"\n  pipeline(a) α=1.0      LER = {ler_a:.4e}")
    print(f"  pipeline(b) η=0.9296   LER = {ler_b:.4e}")
    print(f"  ratio (max/min)            = {ratio:.2f}")

    if ratio > 3.0:
        warnings.warn(
            f"Twirl-Kraus ratio = {ratio:.2f} (> 3) on the n = 9 reference code (seed 1). "
            f"The twirled channel discards per-gate composition; "
            f"pipeline (b) hits {ler_b:.2e} ≈ random-state floor 2/3 at "
            f"calibrated η. Document in writeup; do NOT treat as test failure.",
            RuntimeWarning)
    # No assert — this is an observation test.
