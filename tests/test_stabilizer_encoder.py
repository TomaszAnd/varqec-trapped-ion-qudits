"""Tests for src/stabilizer_encoder.py (paper App. C: the 36-MS encoder)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stabilizer_encoder import (
    cz_gate, cz_gatelist, csub_gatelist, count_ms, count_single,
    encoder_circuit_codewords, five_qudit_encoder_gatelist, fourier_gate,
    frobenius_distance_up_to_phase, pair_gatelist_unitary, subspace_hadamard,
    subspace_z, zz_from_ms, verify_encoder,
)
from src.gates import CSUB_gate, CSUM_gate
from src.catalog import five_qudit_code_states


def test_fourier_unitary_and_cz_phases():
    F = fourier_gate(3)
    assert np.allclose(F @ F.conj().T, np.eye(3), atol=1e-14)
    CZ = cz_gate(3)
    w = np.exp(2j * np.pi / 3)
    for a in range(3):
        for b in range(3):
            assert np.isclose(CZ[a * 3 + b, a * 3 + b], w ** (a * b))


def test_subspace_hadamard_maps_x_to_z():
    H = subspace_hadamard(0, 1, 3)
    X01 = np.zeros((3, 3), complex)
    X01[0, 1] = X01[1, 0] = 1.0
    assert np.allclose(H @ X01 @ H.conj().T, subspace_z(0, 1, 3), atol=1e-14)


def test_single_zz_block_is_one_ms():
    theta = -4 * np.pi / 3
    for ab in ((0, 1), (1, 2)):
        for cd in ((0, 1), (1, 2)):
            gl = zz_from_ms(theta, ab, cd, 3)
            assert count_ms(gl) == 1
            U = pair_gatelist_unitary(gl, 3)
            target = np.diag(np.exp(-1j * theta / 2 * np.kron(
                np.diag(subspace_z(*ab, 3)), np.diag(subspace_z(*cd, 3)))))
            assert frobenius_distance_up_to_phase(U, target) < 1e-12


def test_cz_and_csub_are_four_ms_each():
    CZ = cz_gate(3)
    assert count_ms(cz_gatelist(1)) == 4
    assert frobenius_distance_up_to_phase(pair_gatelist_unitary(cz_gatelist(1)), CZ) < 1e-12
    assert frobenius_distance_up_to_phase(pair_gatelist_unitary(cz_gatelist(2)), CZ.conj().T) < 1e-12
    assert count_ms(csub_gatelist()) == 4
    assert frobenius_distance_up_to_phase(pair_gatelist_unitary(csub_gatelist()),
                                          np.asarray(CSUB_gate(3))) < 1e-12
    assert np.allclose(np.asarray(CSUB_gate(3)), np.asarray(CSUM_gate(3)).conj().T)


def test_full_encoder_budget_and_codewords():
    gates = five_qudit_encoder_gatelist()
    assert count_ms(gates) == 36                      # paper App. C / Sec. V E
    assert count_single(gates) > 0
    cs = five_qudit_code_states(3)
    circ = encoder_circuit_codewords()
    for a in range(3):
        ov = np.vdot(cs[a], circ[a])
        assert abs(ov) > 1 - 1e-12
        assert np.linalg.norm(circ[a] - ov / abs(ov) * cs[a]) < 1e-13
    checks = verify_encoder()
    assert checks['ms_budget'] == 36
    assert checks['two_qudit_cliffords'] == 9
    assert max(checks['codeword_devs_up_to_phase']) < 1e-13


def test_matches_committed_circuit_record():
    """The committed campaign record lists the same gate count and budget."""
    import json
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(here, "results", "campaigns", "r2_campaign", "t6a_circuit.json")
    if not os.path.exists(p):
        return
    rec = json.load(open(p))
    gates = five_qudit_encoder_gatelist()
    assert rec['checks']['ms_budget'] == count_ms(gates)
    assert len(rec['gates']) == len(gates)
    # same MS parameters in the same order
    ms_rec = [(g['q1'], g['q2'], g['phi'], g['theta'], g['j'], g['k'])
              for g in rec['gates'] if g['type'] == 'ms']
    ms_new = [(g[1], g[2], g[3], g[4], g[5], g[6]) for g in gates if g[0] == 'ms']
    assert len(ms_rec) == len(ms_new) == 36
    for a, b in zip(ms_rec, ms_new):
        assert a[0] == b[0] and a[1] == b[1] and a[4] == b[4] and a[5] == b[5]
        assert np.isclose(a[2], b[2]) and np.isclose(a[3], b[3])
