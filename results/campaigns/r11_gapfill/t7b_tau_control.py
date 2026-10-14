#!/usr/bin/env python3
"""Tighter-tau control at n=5 (certified budgets, Sec. IX C).

At tau = 1e-2 (3.0x the measured floor) the certified stop lands at
steps 6-39 and 5 of 12 codes carry small weight-2 leaks (k_w2 in
{1, 6, 7} per 1e5 conditioned shots). Control: tau = 5e-3 (1.5x floor,
still attainable) for the recommended sampler (importance@0.05) and
full batch, same seeds -- does modest tightening restore per-seed strict
w2-perfection, and at what budget?

Run: nohup nice -n 19 python3 t7b_tau_control.py >> t7b.log 2>&1 &
Writes t7_tau5e3_results.json (resumable per unit).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import t7_budget as t7  # noqa: E402  (reuses env caps, x64 import, machinery)

OUT = os.path.join(HERE, "t7_tau5e3_results.json")
TAU = 5e-3
CELLS = [("importance", 0.05), ("full_batch", 1.0)]
SEEDS = [100, 101, 102, 103]


def main():
    res = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            res = json.load(f)
    res.setdefault("meta", {
        "tau": TAU,
        "purpose": "tighter-tau control (1.5x floor 3.36e-3); does "
                   "per-seed strict w2-perfection return, at what budget"})
    res.setdefault("units", {})

    from src.errors import qudit_hardware_error_basis, make_hardware_noise_fn
    from src.simulation import (simulate_ler_with_correction_factored,
                                _apply_factored_op)
    from src.decoders._common import encoder_forward
    D, N = t7.D, t7.N
    SINGLES = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]

    def w2_noise(state, rng):
        qs = rng.choice(N, size=2, replace=False)
        s = state
        for q in qs:
            s = _apply_factored_op(s, SINGLES[rng.integers(len(SINGLES))],
                                   int(q), N, D)
        nrm = np.linalg.norm(s)
        return s / nrm if nrm > 1e-15 else s

    for sampler, frac in CELLS:
        for seed in SEEDS:
            base = f"{sampler}@f{frac}" if sampler != "full_batch" \
                else "full_batch"
            key = f"{base}_tau5e3_s{seed}"
            if key in res["units"]:
                continue
            print(f"[t7b] training {key} ...", flush=True)
            r = t7.train_certified(sampler, frac, seed, TAU)
            theta = r.pop("theta")
            npz = os.path.join(t7.CODEDIR, key + ".npz")
            np.savez_compressed(npz, params=theta, d=D, n=N, n_qudit=N,
                                K=3, layers=t7.LAYERS, entangler="ms")
            row = dict(r)
            row.update({"sampler": sampler, "frac": frac, "seed": seed,
                        "tau": TAU, "npz": npz})
            cs = np.asarray(encoder_forward(npz, t7.CONNS)[0])
            rr = simulate_ler_with_correction_factored(
                cs, w2_noise, SINGLES, N, D, n_shots=100_000, seed=1002)
            lw2 = float(rr["logical_error_rate"])
            row["ler_w2"] = {"ler": lw2, "k": int(round(lw2 * 100_000)),
                             "n": 100_000}
            n_shots = t7.IID_START
            while True:
                rr = simulate_ler_with_correction_factored(
                    cs, make_hardware_noise_fn(D, N, t7.IID_P), SINGLES,
                    N, D, n_shots=n_shots, seed=2000)
                ler = float(rr["logical_error_rate"])
                k = int(round(ler * n_shots))
                if k >= t7.MIN_ERRORS or n_shots >= t7.IID_CAP:
                    row["ler_iid_p05"] = {
                        "ler": ler, "k": k, "n_shots": n_shots,
                        "wilson95": list(t7.wilson(k, n_shots)),
                        "resolved": bool(k >= t7.MIN_ERRORS)}
                    break
                n_shots = min(2 * n_shots, t7.IID_CAP)
            res["units"][key] = row
            json.dump(res, open(OUT, "w"), indent=2)
            print(f"[t7b] {key}: cert={row['certified']} "
                  f"stop={row['stop_step']} budget={row['budget_op_evs']} "
                  f"k_w2={row['ler_w2']['k']} "
                  f"iid={row['ler_iid_p05']['ler']:.3e}", flush=True)

    summ = {}
    for sampler, frac in CELLS:
        base = f"{sampler}@f{frac}" if sampler != "full_batch" \
            else "full_batch"
        rows = [res["units"][f"{base}_tau5e3_s{seed}"] for seed in SEEDS]
        budgets = sorted(r["budget_op_evs"] for r in rows)
        summ[base] = {
            "certified": sum(1 for r in rows if r["certified"]),
            "budget_median": float(np.median(budgets)),
            "budget_range": [budgets[0], budgets[-1]],
            "stop_steps": [r["stop_step"] for r in rows],
            "w2_perfect": sum(1 for r in rows if r["ler_w2"]["k"] == 0),
            "k_w2": [r["ler_w2"]["k"] for r in rows],
            "iid_lers": [r["ler_iid_p05"]["ler"] for r in rows]}
    res["summary"] = summ
    json.dump(res, open(OUT, "w"), indent=2)
    print("summary:", json.dumps(summ, indent=1), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
