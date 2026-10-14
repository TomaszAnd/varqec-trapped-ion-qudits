#!/usr/bin/env python3
"""Training-basis study retrained at L=2 (replaces the L=16 arms).

The training-basis study (loss_ler_diag/t1g_queue.py) trained hw-closure
vs full-closure unweighted objectives at n=5 L=16, 8 seeds each. Here
BOTH arms are retrained at L=2, same seeds (100..107), same protocol
otherwise (3000 steps, full-batch uniform, target 1e-9 never certifies,
loss-best theta kept), so that no L=16 code remains load-bearing.

Per code, the study's comparison measurements PLUS:
  - train/hw-basis/full-basis losses (cross-evaluated, as t1g);
  - conditioned probes LER|w for w in {1,2,3} @ 1e5 shots
    (seeds 1001/1002/1003; w3 added for standard-selection parity);
  - iid LER @ p=0.05, escalated 3e5 doubling to >= 30 errors, cap 4.8e6,
    Wilson 95% (selector ground truth (a));
  - exact channel-v2 cells over the FULL s grid {0.125,...,1.5}:
    projective-MAP + full-dose Petz rates (selector ground truth (b) at
    s in {0.5, 1.0}; the rest de-arbitrates the s choice).

The npz checkpoints keep run_race's per-step 'losses' history (full-batch
here, so the trace IS the full loss) -- the plateau/loss selectors of
t6_selectors.py are computed from these traces in the analysis step.

Shards: --arm hw | full (one detached worker each).
Writes t1b_results_{arm}.json (resumable per unit). Feeds the selector
comparison of the paper.
Launch: cd results/campaigns/r11_gapfill
        nohup nice -n 19 python3 t1b_queue.py --arm hw  >> t1b_hw.log 2>&1 &
        nohup nice -n 19 python3 t1b_queue.py --arm full >> t1b_full.log 2>&1 &
"""
import argparse
import json
import math
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
os.chdir(REPO)

import src.jax_backend  # noqa: F401  x64 first (footgun)
import numpy as np
import jax.numpy as jnp

from src.seed_race import run_race, build_race
from src.errors import qudit_hardware_error_basis, make_hardware_noise_fn
from src.simulation import (simulate_ler_with_correction_factored,
                            _apply_factored_op)
from src.decoders._common import encoder_forward
from src.decoders.exact_channel import build_dose_steps, evolve_code_blocks
from src.decoders.projective_decode import (
    build_petz_exact_dose, build_projective_instrument,
    entanglement_fidelity_petz, entanglement_fidelity_projective,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CODEDIR = os.path.join(HERE, "t1b_codes")

D, N, L, STEPS, P = 3, 5, 2, 3000, 0.05
SEEDS = list(range(100, 108))
SIGMA_P2, SIGMA_A2_CAL = 0.073, 0.047
S_GRID = (0.125, 0.25, 0.5, 1.0, 1.5)
IID_START, IID_CAP, MIN_ERRORS = 300_000, 4_800_000, 30

SINGLES = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]
GP = [(i, j, 0, 1) for i in range(N) for j in range(i + 1, N)]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    dn = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / dn
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / dn
    return (max(0.0, c - h), c + h)


def wk_noise(w):
    def fn(state, rng):
        qs = rng.choice(N, size=w, replace=False)
        s = state
        for q in qs:
            s = _apply_factored_op(s, SINGLES[rng.integers(len(SINGLES))],
                                   int(q), N, D)
        nrm = np.linalg.norm(s)
        return s / nrm if nrm > 1e-15 else s
    return fn


def iid_cell(cs, p, seed):
    n_shots = IID_START
    while True:
        r = simulate_ler_with_correction_factored(
            cs, make_hardware_noise_fn(D, N, p), SINGLES, N, D,
            n_shots=n_shots, seed=seed)
        ler = float(r["logical_error_rate"])
        k = int(round(ler * n_shots))
        if k >= MIN_ERRORS or n_shots >= IID_CAP:
            lo, hi = wilson(k, n_shots)
            return {"p": p, "ler": ler, "k": k, "n_shots": n_shots,
                    "wilson95": [float(lo), float(hi)],
                    "resolved": bool(k >= MIN_ERRORS)}
        n_shots = min(2 * n_shots, IID_CAP)


def exact_cells(cs):
    out = {}
    for s in S_GRID:
        eta, sa2 = float(np.exp(-s * SIGMA_P2)), s * SIGMA_A2_CAL
        steps = build_dose_steps(N, D, GP, eta, sigma_a2=sa2)
        inst = build_projective_instrument(cs, D, N, eta, sigma_a2=sa2)
        blocks = evolve_code_blocks(cs, steps, D, N)
        proj = entanglement_fidelity_projective(inst, blocks)
        petz = build_petz_exact_dose(cs, steps, D, N)
        pz = entanglement_fidelity_petz(petz, cs, blocks)
        out[f"s{s:.3f}"] = {
            "s": s, "eta": eta, "sigma_a2": sa2,
            "projective_rate": proj["rate_1mFavg"],
            "petz_rate": pz["rate_1mFavg"],
            "petz_leak_max": float(max(pz["recovery_leak_per_codeword"]))}
    return out


def set_status(path, **kw):
    kw["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(kw, open(path, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("hw", "full"), required=True)
    arm = ap.parse_args().arm
    os.makedirs(CODEDIR, exist_ok=True)
    OUT = os.path.join(HERE, f"t1b_results_{arm}.json")
    STATUS = os.path.join(HERE, f"t1b_status_{arm}.json")

    res = json.load(open(OUT)) if os.path.exists(OUT) else {
        "protocol": "Training-basis study protocol at L=2: n=5 L=2 a2a MS, 3000 steps, "
                    "full-batch uniform, target 1e-9 (never certifies; "
                    "loss-best theta kept); probes w1/w2/w3 @1e5 "
                    "(seeds 1001/1002/1003); iid p=0.05 escalated "
                    "3e5->4.8e6 to >=30 errors (seed 2000); exact v2 "
                    "proj+Petz cells on s grid {0.125..1.5}; "
                    "metric rate = 1 - F_avg (channel infidelity)",
        "units": {}}

    print(f"[t1b:{arm}] building loss evaluators ...", flush=True)
    _, hw_loss_fn, *_ = build_race(D, N, 3, L, noise="hw")
    _, full_loss_fn, *_ = build_race(D, N, 3, L, noise="full")

    for i, seed in enumerate(SEEDS):
        key = f"{arm}_L2_s{seed}"
        if key in res["units"]:
            print(f"[t1b:{arm}] {key} done -- skip", flush=True)
            continue
        set_status(STATUS, state="running", unit=key,
                   done=len(res["units"]), total=len(SEEDS))
        t0 = time.time()
        npz = os.path.join(CODEDIR, key + ".npz")
        r = run_race(d=D, n=N, distance=3, layers=L, num_seeds=1,
                     steps=STEPS, target_loss=1e-9, sample_frac=1.0,
                     sampler="uniform", trigger_margin=0.25,
                     trigger_patience=5, out=npz, first_seed=seed,
                     verbose=False, connectivity="all-to-all",
                     entangler="ms", noise=arm)
        theta = jnp.asarray(r["result"]["theta"])
        row = {"basis": arm, "seed": seed, "L": L, "npz": npz,
               "train_loss": float(r["result"]["full_loss"]),
               "hw_basis_loss": float(hw_loss_fn(theta)),
               "full_basis_loss": float(full_loss_fn(theta)),
               "train_wall_s": round(time.time() - t0, 1)}

        cs = np.asarray(encoder_forward(npz, CONNS)[0])
        for w, sd in ((1, 1001), (2, 1002), (3, 1003)):
            rr = simulate_ler_with_correction_factored(
                cs, wk_noise(w), SINGLES, N, D, n_shots=100_000, seed=sd)
            ler = float(rr["logical_error_rate"])
            k = int(round(ler * 100_000))
            row[f"ler_w{w}"] = {"ler": ler, "k": k, "n": 100_000,
                                "wilson95": wilson(k, 100_000)}
        row["ler_iid_p05"] = iid_cell(cs, P, seed=2000)
        row["exact_v2"] = exact_cells(cs)
        row["wall_s"] = round(time.time() - t0, 1)
        res["units"][key] = row
        json.dump(res, open(OUT, "w"), indent=2)
        print(f"[t1b:{arm}] UNIT DONE {key}: train={row['train_loss']:.3e} "
              f"full={row['full_basis_loss']:.3e} "
              f"w2={row['ler_w2']['k']}/1e5 w3={row['ler_w3']['ler']:.3f} "
              f"iid={row['ler_iid_p05']['ler']:.3e} "
              f"petz@s0.5={row['exact_v2']['s0.500']['petz_rate']:.3e} "
              f"[{row['wall_s']}s]", flush=True)

    tally = {}
    rows = list(res["units"].values())
    tally[arm] = {
        "n": len(rows),
        "perfect_w1": sum(1 for u in rows if u["ler_w1"]["k"] == 0),
        "perfect_w2": sum(1 for u in rows if u["ler_w2"]["k"] == 0),
        "w2_lers": sorted(round(u["ler_w2"]["ler"], 6) for u in rows),
        "w3_lers": sorted(round(u["ler_w3"]["ler"], 6) for u in rows),
        "iid_p05_lers": sorted(round(u["ler_iid_p05"]["ler"], 7)
                               for u in rows)}
    res["tally"] = tally
    json.dump(res, open(OUT, "w"), indent=2)
    set_status(STATUS, state="done", done=len(res["units"]),
               total=len(SEEDS))
    print(f"[t1b:{arm}] QUEUE DONE tally={json.dumps(tally)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        arm = "unknown"
        for a in sys.argv:
            if a in ("hw", "full"):
                arm = a
        set_status(os.path.join(HERE, f"t1b_status_{arm}.json"),
                   state="FAILED", error=str(e))
        raise
