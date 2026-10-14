#!/usr/bin/env python3
"""Per-gate-infidelity axis verification: per-qudit-per-gate average infidelity.

(1) iid side: 1 - F_avg = c_q * p with c_q = (4q-4)/(q^2+q) (cross-q
    appendix claim) re-verified directly from qudit_hardware_error_basis via
    the Nielsen average-fidelity formula, q = 3, 4, 5.
(2) channel side: per-role average infidelity I_role(eta, q) of the per-gate
    Gaussian dephasing Kraus (control / target / spectator-ion tiers);
    at q = 3 all roles share the coupling-gap multiset {1,1,2} -> one number.
(3) equivalence table eta <-> I <-> equivalent iid p at q=3.
(4) timing: n=5 MAP per-shot cost (2k-shot probe on the central code).

Writes r7_t3_results.json. Feeds the per-gate-infidelity axis of the paper.
"""
import os, sys, json, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
os.chdir(REPO)
import src.jax_backend  # x64 first
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
res = {}

# (1) iid mapping
from src.errors import qudit_hardware_error_basis
for q in (3, 4, 5):
    ops = [np.asarray(E, complex) for E in qudit_hardware_error_basis(q)]
    infids = [1 - (abs(np.trace(U)) ** 2 + q) / (q ** 2 + q) for U in ops]
    cq_num = float(np.mean(infids))
    cq_form = (4 * q - 4) / (q ** 2 + q)
    res[f'c_q{q}'] = {'numeric': cq_num, 'formula': cq_form,
                      'n_ops': len(ops), 'rel_dev': abs(cq_num - cq_form) / cq_form}

# (2)+(3) channel mapping per role
from src.correlated_noise import (control_qudit_kraus, target_qudit_kraus,
                                  spectator_qudit_kraus)


def role_infid(kraus_diags, q):
    s = sum(abs(np.sum(dg)) ** 2 for dg in kraus_diags)  # sum |tr E_k|^2
    return float(1 - (s + q) / (q ** 2 + q))


ETAS = [0.8642, 0.88, 0.90, 0.92, 0.9296, 0.95, 0.97, 0.99]
tab = {}
for eta in ETAS:
    row = {}
    for q in (3, 4, 5):
        row[f'q{q}'] = {
            'control': role_infid(control_qudit_kraus(q, 0, eta, 12), q),
            'target': role_infid(target_qudit_kraus(q, 0, 1, eta, 12), q),
            'spectator_ion': role_infid(spectator_qudit_kraus(q, eta, 12), q)}
    I3 = row['q3']['target']
    row['equiv_iid_p_q3'] = I3 / res['c_q3']['formula']
    tab[eta] = row
res['eta_to_infid'] = tab
res['calibrated_point'] = {
    'eta': 0.9296, 'I_pergate_perqudit_q3': tab[0.9296]['q3']['target'],
    'equiv_iid_p_q3': tab[0.9296]['equiv_iid_p_q3'],
    'note': 'p=0.05 benchmark convention vs equivalent p of one calibrated gate'}

# (4) n=5 MAP timing probe
from weighted_race_test import map_channel_ler
npz5 = os.path.join(REPO, 'results/campaigns/r1_campaign/t1_codes/A_L2_s100.npz')
t0 = time.time()
ler, ci = map_channel_ler(npz5, 3, 5, eta=0.9296, n_shots=2000, seed=99)
w = time.time() - t0
res['n5_map_probe'] = {'ler_2k': ler, 'ci': list(ci), 'wall_s': round(w, 1),
                       'ms_per_shot_incl_setup': w / 2000 * 1e3}

json.dump(res, open(os.path.join(HERE, 'r7_t3_results.json'), 'w'), indent=2)
print(json.dumps(res, indent=2))
