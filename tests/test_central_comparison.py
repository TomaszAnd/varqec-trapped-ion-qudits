"""Tests for src/central_comparison.py and the catalog entry of the central code.

Pinned values are the paper's (Table I, Table II, Sec. VI A, App. F) and the
committed campaign records they were printed from.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.jax_backend  # noqa: F401  (enable x64 before any JAX use)
from src import central_comparison as cc


def test_constants_and_strength_grid():
    assert math.isclose(cc.ETA_CAL, 0.9296, abs_tol=5e-5)
    assert math.isclose(cc.eta_of_s(1.0), cc.ETA_CAL)
    assert math.isclose(cc.sigma_a2_of_s(0.5), 0.0235)
    assert cc.S_GRID == (0.125, 0.25, 0.5, 1.0, 1.5)


def test_coherence_multipliers_are_half_sigma_squared():
    """e^{-sigma_p^2 Delta_f^2 / 2}: 1.0 / 0.9642 / 0.8642 at eta = 0.9296."""
    m = cc.coherence_multipliers(cc.SIGMA_P2, [0, 1, 2])
    assert np.allclose(m, [1.0, 0.9642, 0.8642], atol=5e-5)
    # and the exact-channel masks use the same factor
    steps = cc.noise_round_steps(5, 3, 1.0)
    assert len(steps) == 10
    spec = steps[0]['deph'][4]           # qudit 4 is a spectator in step (0,1)
    assert np.isclose(spec[0, 1], m[1]) and np.isclose(spec[0, 2], m[2])


def test_catalog_central_code_matches_jax_encoder():
    from src.catalog import load_code, code_path
    from src.decoders._common import encoder_forward
    code = load_code('central_553_3')
    cs = np.asarray(code['code_states'])
    assert cs.shape == (3, 243)
    assert np.allclose(cs @ cs.conj().T, np.eye(3), atol=1e-10)
    assert code['metadata']['n_layers'] == 2
    assert code['metadata']['connections'] == cc.all_to_all_edges(5)
    cs_jax, d, n, K = encoder_forward(code_path('central_553_3'), cc.all_to_all_edges(5))
    assert (d, n, K) == (3, 5, 3)
    assert np.linalg.norm(cs - cs_jax) < 1e-10


def test_resource_counts_table_iii():
    r = cc.ansatz_resource_counts(5, 3)
    assert (r['P_layer'], r['params'], r['XY_per_layer'], r['MS_per_layer'],
            r['Z_per_layer'], r['MS_budget']) == (90, 180, 20, 20, 10, 40)
    r9 = cc.ansatz_resource_counts(9, 3)
    assert (r9['E'], r9['P_layer'], r9['MS_budget']) == (36, 234, 144)
    r5 = cc.ansatz_resource_counts(5, 5)
    assert (r5['P_layer'], r5['MS_budget']) == (180, 80)


def test_table_i_calibrated_cells():
    """Table I at s = 1 (units of 1e-2): proj 39.62 / 37.16, Petz 16.22 / 10.43,
    ratio 1.56; sector structure 81 x 3-dim (central) and 74/7/7 (stabilizer)."""
    steps = cc.noise_round_steps(5, 3, 1.0)
    cen = cc.channel_cell(cc.central_code_states(), 1.0, steps=steps)
    stab = cc.channel_cell(cc.stabilizer_code_states(), 1.0, steps=steps)
    assert round(100 * cen['projective_rate'], 2) == 39.62
    assert round(100 * cen['petz_rate'], 2) == 16.22
    assert round(100 * stab['projective_rate'], 2) == 37.16
    assert round(100 * stab['petz_rate'], 2) == 10.43
    assert round(cen['petz_rate'] / stab['petz_rate'], 2) == 1.56
    assert cen['n_sectors'] == 81 and cen['sector_dim_counts'] == {3: 81}
    assert cen['rest_dim'] == 0
    assert stab['sector_dim_counts'] == {1: 7, 2: 7, 3: 74}


def test_table_i_without_angle_fluctuation():
    """Caption of Table I: sigma_a^2 = 0 at s = 1 gives Petz 11.8 / 5.9."""
    steps = cc.noise_round_steps(5, 3, 1.0, with_angle_fluctuation=False)
    cen = cc.channel_cell(cc.central_code_states(), 1.0, steps=steps,
                          with_angle_fluctuation=False)
    stab = cc.channel_cell(cc.stabilizer_code_states(), 1.0, steps=steps,
                           with_angle_fluctuation=False)
    assert round(100 * cen['petz_rate'], 1) == 11.8
    assert round(100 * stab['petz_rate'], 1) == 5.9


def test_certified_optimal_recovery_s_half():
    """App. F: certified optimum at s = 0.5; Petz-over-optimal 1.46 (stabilizer)
    and 1.34 (central), from the committed r12 record."""
    steps = cc.noise_round_steps(5, 3, 0.5)
    stab = cc.channel_cell(cc.stabilizer_code_states(), 0.5, steps=steps, optimal=True)
    assert stab['F_e_opt_ub'] - stab['F_e_opt_lb'] < 1e-5
    assert stab['certified_opt_ge_petz'] and stab['certified_opt_ge_projective']
    assert abs(stab['optimal_rate_lb'] - 0.02116) < 2e-5
    assert round(stab['petz_rate'] / stab['optimal_rate_lb'], 2) == 1.46


def test_iid_and_probe_helpers_small():
    cs = cc.stabilizer_code_states()
    r = cc.iid_logical_error_rate(cs, p=0.05, n_shots=2000, seed=1)
    assert 0 <= r['ler'] <= 1 and r['wilson95'][0] <= r['ler'] <= r['wilson95'][1]
    w1 = cc.weight_conditioned_probe(cs, 1, n_shots=500, seed=2)
    assert w1['k'] == 0                      # weight-1 errors are always corrected
    assert w1['mean_fidelity'] > 1 - 1e-10
    lo, hi = cc.wilson_interval(0, 100000)
    assert lo == 0.0 and hi < 4e-5


def test_layer_gate_sequence_matches_ansatz_counts():
    from src.encoder import layer_gate_sequence, create_native_encoder
    conns = cc.all_to_all_edges(5)
    seq = layer_gate_sequence(5, 3, conns)
    kinds = [g['gate'] for g in seq]
    assert kinds.count('XY') == 20 and kinds.count('MS') == 20 and kinds.count('Z') == 10
    assert kinds == ['XY'] * 20 + ['MS'] * 20 + ['Z'] * 10
    slots = [s for g in seq for s in g['param_slots']]
    _, _, ppl = create_native_encoder(5, 3, connections=conns, force_manual=True)
    assert slots == list(range(ppl)) and ppl == 90
