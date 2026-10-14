#!/usr/bin/env python3
"""Profiling: where does an n=7 v2 exact cell spend its time?"""
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

import numpy as np

import src.jax_backend  # noqa: F401
from src.decoders.exact_channel import (build_dose_steps, apply_dose,
                                        apply_step, _rotate_pair)

D, N = 3, 7
GP = [(i, j, 0, 1) for i in range(N) for j in range(i + 1, N)]
ETA, SA2 = float(np.exp(-0.073)), 0.047

rng = np.random.default_rng(0)
dim = D ** N
A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
rho = A @ A.conj().T
rho /= np.trace(rho).real

steps = build_dose_steps(N, D, GP, ETA, sigma_a2=SA2)

# full dose timing
t0 = time.time()
out = apply_dose(rho, steps, D, N)
t_full = time.time() - t0
print(f"full v2 dose ({len(GP)} steps): {t_full:.2f}s", flush=True)

# one step breakdown: dephasing-only vs jitter-only
rho_t = rho.reshape((D,) * (2 * N)).astype(complex, copy=True)
step = steps[0]

t0 = time.time()
for q in range(N):
    m = step['deph'][q]
    shape = [1] * (2 * N)
    shape[q] = D
    shape[N + q] = D
    rho_t *= m.reshape(shape)
t_deph = time.time() - t0
print(f"one step, deph part (7 schur multiplies): {t_deph:.3f}s", flush=True)

V1 = step['V1']
qa, qb = step['qa'], step['qb']
t0 = time.time()
rho_t2 = _rotate_pair(rho_t, V1.conj().T, qa, qb, N)
t_rot1 = time.time() - t0
t0 = time.time()
jm = step['jmask']
axes_sorted = np.argsort([qa, qb, N + qa, N + qb])
jmT = np.transpose(jm, axes_sorted)
shape = [1] * (2 * N)
for ax in (qa, qb, N + qa, N + qb):
    shape[ax] = D
rho_t2 *= jmT.reshape(shape)
t_mask = time.time() - t0
t0 = time.time()
rho_t2 = _rotate_pair(rho_t2, V1, qa, qb, N)
t_rot2 = time.time() - t0
print(f"one step, jitter: rot_in={t_rot1:.3f}s mask={t_mask:.3f}s "
      f"rot_out={t_rot2:.3f}s", flush=True)

# instrument + petz component costs
from src.decoders.projective_decode import (build_projective_instrument,
                                            build_petz_exact_dose)
from src.decoders._common import encoder_forward
BR = os.path.join(REPO, "results", "campaigns")
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]
cs, *_ = encoder_forward(os.path.join(BR, "r3_campaign", "t2c_codes",
                                      "d3_L2_s100.npz"), CONNS)
t0 = time.time()
inst = build_projective_instrument(cs, D, N, ETA, sigma_a2=SA2)
print(f"instrument build (n=7, v2 prior): {time.time()-t0:.1f}s "
      f"({inst['report']['n_sectors']} sectors, "
      f"R={inst['report']['claimed_dim_R']})", flush=True)
t0 = time.time()
petz = build_petz_exact_dose(cs, steps, D, N)
print(f"petz build (1 dose + eigh + invsqrt): {time.time()-t0:.1f}s",
      flush=True)
