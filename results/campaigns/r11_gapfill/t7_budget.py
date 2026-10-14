#!/usr/bin/env python3
"""Certified budgets (Sec. IX C): certified-stop budget measurement at n=5
(the per-seed budget quote, measured instead of preset).

The earlier per-seed quote of ~1.1e5 training op-EVs for the n=5
protocol was a PRESET-1500-step importance@0.2 number. This measures the
certified-stop budget at n=5 under the adopted rule (sampled loss < tau
for 5 consecutive steps, >= 50 steps between full-batch confirmations,
confirmed full-batch < tau stops; backstop 3000 steps), for full batch,
importance@0.2 (published protocol) and importance@0.05 (the sampler
frontier's recommendation), 4 seeds each.

TAU CHOICE (documented, per the convention "attainable, just above the
measured capacity floor"): the n=5 L=2 channel-weighted floor is
measured FIRST from the four committed production checkpoints
A_L2_s{100..103} (full weighted loss of the trained theta); tau = the
smallest decade point >= 2x the max of those floors. This mirrors the
Table VI / r10_sampler convention (floor 4.35e-5 -> tau 1e-4).

Per trained code: w2 probe @1e5 conditioned shots + iid LER @ p=0.05
(3e5 doubling to >= 30 errors, cap 2.4e6) so the budget quote is tied to
production-quality codes. Train + confirmation op-EVs reported per seed.

Writes t7_budget_results.json (resumable per unit). Feeds Sec. IX C.
Launch: cd results/campaigns/r11_gapfill
        nohup nice -n 19 python3 t7_budget.py >> t7_budget.log 2>&1 &
"""
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
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.chdir(REPO)

import src.jax_backend  # noqa: F401  x64 first (footgun)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "t7_budget_results.json")
CODEDIR = os.path.join(HERE, "t7_codes")
BR = os.path.join(REPO, "results", "campaigns")

D, N, LAYERS = 3, 5, 2
MAX_STEPS = 3000
LR = 0.05
PATIENCE, CONFIRM_GAP, MONITOR_EVERY = 5, 50, 25
SEEDS = [100, 101, 102, 103]
CELLS = [("full_batch", 1.0), ("importance", 0.2), ("importance", 0.05)]
IID_P, IID_START, IID_CAP, MIN_ERRORS = 0.05, 300_000, 2_400_000, 30
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    dn = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / dn
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / dn
    return (max(0.0, c - h), c + h)


def measure_floor():
    """Full weighted loss of the four committed production checkpoints."""
    import jax.numpy as jnp
    from src.seed_race import build_race
    _, full_loss_fn, *_ = build_race(D, N, 3, LAYERS, weighted=True,
                                     connectivity="all-to-all")
    floors = {}
    for seed in SEEDS:
        npz = os.path.join(BR, "r1_campaign", "t1_codes",
                           f"A_L2_s{seed}.npz")
        data = np.load(npz, allow_pickle=True)
        theta = jnp.asarray(data["params"])
        floors[f"A_L2_s{seed}"] = float(full_loss_fn(theta))
    fmax = max(floors.values())
    tau = 10.0 ** math.ceil(math.log10(2.0 * fmax))
    return floors, fmax, tau


def train_certified(sampler, frac, seed, tau):
    import jax
    import optax
    from src.seed_race import build_race
    from src.sampling.stratified_importance import (
        make_stratified_importance_weights)

    (val_grad_fn, full_loss_fn, group_sizes, ppl, _,
     full_weights, sgw) = build_race(D, N, 3, LAYERS, weighted=True,
                                     connectivity="all-to-all")
    total_ops = int(sum(group_sizes))
    budget = max(1, int(round(frac * total_ops)))

    theta = jax.random.uniform(jax.random.PRNGKey(seed), (LAYERS, ppl),
                               minval=0.0, maxval=2 * np.pi)
    rng = np.random.default_rng(seed)
    opt = optax.inject_hyperparams(optax.adam)(learning_rate=LR)
    st = opt.init(theta)
    stage = 0

    train_evs = confirm_evs = monitor_evs = 0
    n_confirms = n_premature = 0
    consec, last_confirm = 0, -10 ** 9
    sampled_losses, cum_train = [], []
    mon_steps, mon_losses = [], []
    mon_best, mon_best_theta = np.inf, None
    certified, stop_step, final_full = False, None, None
    theta_out = None
    t0 = time.time()

    for step in range(MAX_STEPS):
        if sampler == "full_batch":
            w = full_weights
            step_evs = total_ops
        elif sampler == "importance":
            w = make_stratified_importance_weights(sgw, budget, rng)
            step_evs = int(sum(int(np.count_nonzero(np.asarray(x)))
                               for x in w))
        else:
            raise ValueError(sampler)
        train_evs += step_evs
        lv, g = val_grad_fn(theta, w)
        lvf = float(lv)
        sampled_losses.append(lvf)
        cum_train.append(train_evs)

        if step % MONITOR_EVERY == 0 or step == MAX_STEPS - 1:
            fv = float(full_loss_fn(theta))
            monitor_evs += total_ops
            mon_steps.append(step)
            mon_losses.append(fv)
            if fv < mon_best:
                mon_best, mon_best_theta = fv, np.array(theta)

        consec = consec + 1 if lvf < tau else 0
        if consec >= PATIENCE and (step - last_confirm) >= CONFIRM_GAP:
            if sampler == "full_batch":
                certified, final_full, stop_step = True, lvf, step
                theta_out = np.array(theta)
                break
            fv = float(full_loss_fn(theta))
            confirm_evs += total_ops
            n_confirms += 1
            last_confirm = step
            if fv < tau:
                certified, final_full, stop_step = True, fv, step
                theta_out = np.array(theta)
                break
            n_premature += 1
            consec = 0

        upd, st = opt.update(g, st, theta)
        theta = optax.apply_updates(theta, upd)
        if stage == 0 and lvf < 0.5:
            st.hyperparams['learning_rate'] = LR / 5
            stage = 1
        if stage <= 1 and lvf < 1e-3:
            st.hyperparams['learning_rate'] = LR / 50
            stage = 2

    if not certified:
        fv_final = float(full_loss_fn(theta))
        monitor_evs += total_ops
        if fv_final < mon_best:
            mon_best, mon_best_theta = fv_final, np.array(theta)
        theta_out, final_full = mon_best_theta, mon_best
        stop_step = MAX_STEPS

    return {"theta": theta_out, "certified": bool(certified),
            "stop_step": int(stop_step), "full_loss": float(final_full),
            "train_op_evs": int(train_evs),
            "confirm_op_evs": int(confirm_evs),
            "monitor_op_evs": int(monitor_evs),
            "budget_op_evs": int(train_evs + confirm_evs),
            "n_confirms": int(n_confirms),
            "n_premature": int(n_premature),
            "total_ops": total_ops,
            "train_wall_s": round(time.time() - t0, 1)}


def main():
    res = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            res = json.load(f)
    os.makedirs(CODEDIR, exist_ok=True)

    if "floor" not in res:
        floors, fmax, tau = measure_floor()
        res["floor"] = {
            "per_checkpoint_full_weighted_loss": floors,
            "floor_max": fmax,
            "tau": tau,
            "tau_rule": "smallest decade point >= 2x max checkpoint floor "
                        "(Sec IX A / Table VI convention)"}
        json.dump(res, open(OUT, "w"), indent=2)
        print(f"floors={floors} -> tau={tau}", flush=True)
    tau = res["floor"]["tau"]

    from src.errors import qudit_hardware_error_basis, make_hardware_noise_fn
    from src.simulation import (simulate_ler_with_correction_factored,
                                _apply_factored_op)
    from src.decoders._common import encoder_forward
    SINGLES = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]

    def w2_noise(state, rng):
        qs = rng.choice(N, size=2, replace=False)
        s = state
        for q in qs:
            s = _apply_factored_op(s, SINGLES[rng.integers(len(SINGLES))],
                                   int(q), N, D)
        nrm = np.linalg.norm(s)
        return s / nrm if nrm > 1e-15 else s

    res.setdefault("units", {})
    for sampler, frac in CELLS:
        for seed in SEEDS:
            key = f"{sampler}@f{frac}" if sampler != "full_batch" \
                else "full_batch"
            key = f"{key}_s{seed}"
            if key in res["units"]:
                continue
            print(f"[t7] training {key} (tau={tau}) ...", flush=True)
            r = train_certified(sampler, frac, seed, tau)
            theta = r.pop("theta")
            npz = os.path.join(CODEDIR, key + ".npz")
            np.savez_compressed(npz, params=theta, d=D, n=N, n_qudit=N,
                                K=3, layers=LAYERS, entangler="ms")
            row = dict(r)
            row.update({"sampler": sampler, "frac": frac, "seed": seed,
                        "npz": npz})
            # quality scoring
            cs = np.asarray(encoder_forward(npz, CONNS)[0])
            rr = simulate_ler_with_correction_factored(
                cs, w2_noise, SINGLES, N, D, n_shots=100_000, seed=1002)
            lw2 = float(rr["logical_error_rate"])
            row["ler_w2"] = {"ler": lw2,
                             "k": int(round(lw2 * 100_000)), "n": 100_000}
            n_shots = IID_START
            while True:
                rr = simulate_ler_with_correction_factored(
                    cs, make_hardware_noise_fn(D, N, IID_P), SINGLES, N, D,
                    n_shots=n_shots, seed=2000)
                ler = float(rr["logical_error_rate"])
                k = int(round(ler * n_shots))
                if k >= MIN_ERRORS or n_shots >= IID_CAP:
                    row["ler_iid_p05"] = {
                        "ler": ler, "k": k, "n_shots": n_shots,
                        "wilson95": list(wilson(k, n_shots)),
                        "resolved": bool(k >= MIN_ERRORS)}
                    break
                n_shots = min(2 * n_shots, IID_CAP)
            res["units"][key] = row
            json.dump(res, open(OUT, "w"), indent=2)
            print(f"[t7] {key}: cert={row['certified']} "
                  f"stop={row['stop_step']} "
                  f"budget={row['budget_op_evs']} "
                  f"({row['n_confirms']} confirms, "
                  f"{row['n_premature']} premature) "
                  f"w2={row['ler_w2']['k']}/1e5 "
                  f"iid={row['ler_iid_p05']['ler']:.3e}", flush=True)

    # summary
    summ = {}
    for sampler, frac in CELLS:
        base = f"{sampler}@f{frac}" if sampler != "full_batch" \
            else "full_batch"
        rows = [res["units"][f"{base}_s{seed}"] for seed in SEEDS]
        budgets = sorted(r["budget_op_evs"] for r in rows)
        summ[base] = {
            "certified": sum(1 for r in rows if r["certified"]),
            "n": len(rows),
            "budget_median": float(np.median(budgets)),
            "budget_range": [budgets[0], budgets[-1]],
            "train_op_evs": [r["train_op_evs"] for r in rows],
            "confirm_op_evs": [r["confirm_op_evs"] for r in rows],
            "stop_steps": [r["stop_step"] for r in rows],
            "n_confirms": [r["n_confirms"] for r in rows],
            "n_premature": [r["n_premature"] for r in rows],
            "w2_perfect": sum(1 for r in rows if r["ler_w2"]["k"] == 0),
            "iid_lers": [r["ler_iid_p05"]["ler"] for r in rows]}
    res["summary"] = summ
    json.dump(res, open(OUT, "w"), indent=2)
    print("summary:", json.dumps(summ, indent=1), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
