#!/usr/bin/env python3
"""Shor-Laflamme enumerators (weights j <= 2, hardware basis, wire-grouped
path) for the r10_sampler certified-stop sampler-study code
importance@0.05 s100 -- the Table II sampler row (replaces the retired
L=16 sampler-study row). Writes w5_enum_imp005.json."""
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import src.jax_backend  # noqa: F401  (x64 first)
import numpy as np
from reproduce_tables import enumerators
from src.decoders._common import encoder_forward

N, D = 9, 3
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]
NPZ = os.path.join(REPO, "results", "campaigns", "r10_sampler",
                   "t1_codes", "importance_0.05_s100.npz")

cs, d_, n_, K_ = encoder_forward(NPZ, CONNS)
assert (d_, n_, K_) == (D, N, 3), (d_, n_, K_)
t0 = time.time()
A, B = enumerators(np.stack(cs), N, D)
out = {"code": "importance_0.05_s100 (((9,3,3))_3 channel-weighted L=2, "
               "certified stop)",
       "npz": os.path.relpath(NPZ, REPO),
       "A1": float(A[1]), "A2": float(A[2]),
       "B1": float(B[1]), "B2": float(B[2]),
       "B1_minus_A1": float(B[1] - A[1]),
       "B2_minus_A2": float(B[2] - A[2]),
       "wall_s": round(time.time() - t0, 1)}
with open(os.path.join(os.path.dirname(__file__),
                       "w5_enum_imp005.json"), "w") as f:
    json.dump(out, f, indent=2, default=float)
print(out)
