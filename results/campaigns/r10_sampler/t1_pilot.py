#!/usr/bin/env python3
"""Sampler frontier with certified stop (Figs. 10-11): sampler-frontier
variance pilot.

System: ((9,3,3))_3, all-to-all MS, L=2, channel-weighted objective
(build_race weighted=True: v1 Meth prior eta=0.9296, 685-op closure).
Stopping (all cells incl. full batch): certified rule — sampled
loss < TAU for 5 CONSECUTIVE steps, >= 50 steps between full-batch
confirmations, stop on confirmed full-batch < TAU; backstop 3000 steps
(uncertified -> best monitored theta, stated). TAU = 1e-4 (the paper's
convention; measured L=2 weighted floor 4.2-5.3e-5).
Accounting: realized op-EVs (nonzero weight entries) = train; 685 per
confirmation = confirm; every-25-step monitor evals logged for traces,
EXCLUDED from budgets (paper convention).

Scoring per code:
  (a) w1@2e5 / w2@1e5 conditioned probes + iid LER p=0.05 (3e5 shots
      doubling to >= 30 errors, cap 2.4e6, Wilson 95%);
  (b) channel-v2 projective sampled rate at s=0.5 (eta=e^{-0.0365},
      sigma_a2=0.0235), MODE A, 1000 traj/ensemble, SE stated.

Shards:
  0: score the five committed L=2 strategy codes (r3_campaign/t3_sampler_L2)
  1: even pilot units   2: odd pilot units
Pilot units = {full_batch, importance@0.05, importance@0.4,
full_basis_is@0.05, full_basis_is@0.4} x seeds {100..103}.

Writes t1_results_shard{K}.json (resumable per unit).
Launch (one per shard):
  cd results/campaigns/r10_sampler
  nohup nice -n 19 python3 t1_pilot.py --shard K >> t1_shard{K}.log 2>&1 &
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
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.chdir(REPO)

import src.jax_backend  # noqa: F401  x64 first (footgun)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
L2_DIR = os.path.join(REPO, "results", "campaigns",
                      "r3_campaign", "t3_sampler_L2")
CODEDIR = os.path.join(HERE, "t1_codes")
os.makedirs(CODEDIR, exist_ok=True)

D, N, LAYERS = 3, 9, 2
TAU = 1e-4
MAX_STEPS = 3000
LR = 0.05
PATIENCE, CONFIRM_GAP, MONITOR_EVERY = 5, 50, 25
SIGMA_P2, SIGMA_A2_CAL = 0.073, 0.047
S_B = 0.5
ETA_B = math.exp(-S_B * SIGMA_P2)          # 0.9641580938963240
SA2_B = S_B * SIGMA_A2_CAL                 # 0.0235
CONNS = [[i, j] for i in range(N) for j in range(i + 1, N)]
GATE_PAIRS = [(i, j, 0, 1) for i in range(N) for j in range(i + 1, N)]

IID_P, IID_START, IID_CAP, IID_MIN_ERRORS = 0.05, 300_000, 2_400_000, 30
W1_SHOTS, W2_SHOTS = 200_000, 100_000
NTRAJ_B = 1000

PILOT_CELLS = [("full_batch", 1.0), ("importance", 0.05),
               ("importance", 0.4), ("full_basis_is", 0.05),
               ("full_basis_is", 0.4)]
SEEDS = [100, 101, 102, 103]
EXISTING = ["full_batch", "stratified_0.2", "importance_0.2",
            "importance_0.35", "full_basis_is_0.1"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    dn = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / dn
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / dn
    return (max(0.0, c - h), c + h)


# ---------------------------------------------------------------- training
def train_certified(sampler, frac, seed, max_steps=MAX_STEPS, tau=TAU):
    """Adopted certified stopping rule on the weighted ((9,3,3))_3 L=2 race
    machinery; returns result dict + saves nothing (caller saves)."""
    import jax
    import optax
    from src.seed_race import build_race
    from src.jax_backend import stratified_weights
    from src.sampling.stratified_importance import (
        make_stratified_importance_weights,
        make_stratified_importance_weights_full_basis)

    (val_grad_fn, full_loss_fn, group_sizes, ppl, _,
     full_weights, sgw) = build_race(D, N, 3, LAYERS, weighted=True,
                                     connectivity="all-to-all")
    total_ops = int(sum(group_sizes))
    budget = max(1, int(round(frac * total_ops)))
    if sampler == "full_basis_is":
        from loss_split_helpers import build_w1_w2_masks
        from src.errors import ErrorModel
        grouped = ErrorModel(d=D, n_qudit=N, distance=3,
                             closed=True).build_grouped(verbose=False)
        w1_mask, w2_mask = build_w1_w2_masks(grouped, D)
        w1_budget = max(1, int(round(0.75 * frac * total_ops)))
        w2_budget = max(1, int(round(frac * total_ops)) - w1_budget)

    theta = jax.random.uniform(jax.random.PRNGKey(seed), (LAYERS, ppl),
                               minval=0.0, maxval=2 * np.pi)
    rng = np.random.default_rng(seed)
    opt = optax.inject_hyperparams(optax.adam)(learning_rate=LR)
    st = opt.init(theta)
    stage = 0

    train_evs = confirm_evs = monitor_evs = 0
    n_confirms = n_premature = 0
    consec, last_confirm = 0, -10**9
    sampled_losses, cum_train = [], []
    mon_steps, mon_losses = [], []
    mon_best, mon_best_theta = np.inf, None
    certified, stop_step, final_full = False, None, None
    theta_out = None
    t0 = time.time()

    for step in range(max_steps):
        if sampler == "full_batch":
            w = full_weights
            step_evs = total_ops
        elif sampler == "stratified":
            w = stratified_weights(group_sizes, frac, rng)
            step_evs = int(sum(int(np.count_nonzero(np.asarray(x))) for x in w))
        elif sampler == "full_basis_is":
            w = make_stratified_importance_weights_full_basis(
                sgw, w1_mask, w2_mask, w1_budget, w2_budget, rng)
            step_evs = int(sum(int(np.count_nonzero(np.asarray(x))) for x in w))
        elif sampler == "importance":
            w = make_stratified_importance_weights(sgw, budget, rng)
            step_evs = int(sum(int(np.count_nonzero(np.asarray(x))) for x in w))
        else:
            raise ValueError(sampler)
        train_evs += step_evs
        lv, g = val_grad_fn(theta, w)
        lvf = float(lv)
        sampled_losses.append(lvf)
        cum_train.append(train_evs)

        if step % MONITOR_EVERY == 0 or step == max_steps - 1:
            fv = float(full_loss_fn(theta))
            monitor_evs += total_ops
            mon_steps.append(step)
            mon_losses.append(fv)
            if fv < mon_best:
                mon_best, mon_best_theta = fv, np.array(theta)

        consec = consec + 1 if lvf < tau else 0

        if consec >= PATIENCE and (step - last_confirm) >= CONFIRM_GAP:
            if sampler == "full_batch":
                # every step IS full batch: certification is free
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
        # backstop hit: select the best monitored full-batch theta
        fv_final = float(full_loss_fn(theta))
        monitor_evs += total_ops
        if fv_final < mon_best:
            mon_best, mon_best_theta = fv_final, np.array(theta)
        theta_out, final_full = mon_best_theta, mon_best
        stop_step = max_steps

    return {"theta": theta_out, "certified": bool(certified),
            "stop_step": int(stop_step), "full_loss": float(final_full),
            "train_op_evs": int(train_evs), "confirm_op_evs": int(confirm_evs),
            "monitor_op_evs": int(monitor_evs),
            "budget_op_evs": int(train_evs + confirm_evs),
            "n_confirms": int(n_confirms), "n_premature": int(n_premature),
            "total_ops": total_ops, "train_wall_s": round(time.time() - t0, 1),
            "sampled_losses": sampled_losses, "cum_train_evs": cum_train,
            "monitor_steps": mon_steps, "monitor_losses": mon_losses}


# ---------------------------------------------------------------- scoring
_SING = None
def singles():
    global _SING
    if _SING is None:
        from src.errors import qudit_hardware_error_basis
        _SING = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]
    return _SING


def wk_noise(w):
    from src.simulation import _apply_factored_op
    S = singles()
    def fn(state, rng):
        qs = rng.choice(N, size=w, replace=False)
        s = state
        for q in qs:
            s = _apply_factored_op(s, S[rng.integers(len(S))], int(q), N, D)
        nrm = np.linalg.norm(s)
        return s / nrm if nrm > 1e-15 else s
    return fn


def score_code(cs, iid_cap=IID_CAP, w1_shots=W1_SHOTS, w2_shots=W2_SHOTS,
               ntraj_b=NTRAJ_B):
    from src.simulation import (simulate_ler_with_correction_factored,
                                make_meth_mc_noise_fn)
    from src.errors import make_hardware_noise_fn
    from src.decoders.projective_decode import (build_projective_instrument,
                                                sampled_entanglement_fidelity)
    out = {}
    # certification probes
    for w, sd, shots in ((1, 1001, w1_shots), (2, 1002, w2_shots)):
        t0 = time.time()
        rr = simulate_ler_with_correction_factored(
            cs, wk_noise(w), singles(), N, D, n_shots=shots, seed=sd)
        k = int(round(float(rr['logical_error_rate']) * shots))
        out[f'ler_w{w}'] = {'k': k, 'n': shots, 'ler': k / shots,
                            'wilson95': wilson(k, shots),
                            'wall_s': round(time.time() - t0, 1)}
    # (a) iid LER at p=0.05, doubling escalation. A w2-perfect code cannot
    # resolve 30 errors below the cap at the LERs seen at n=9 (~1e-5), so
    # the ladder's sub-cap runs are pure waste for it: jump straight to the
    # cap. seed=2000 is fixed and RNG draws are shot-ordered, so the cap run
    # is bit-identical to the one the ladder would have recorded.
    shots = iid_cap if out['ler_w2']['k'] == 0 else IID_START
    while True:
        t0 = time.time()
        rr = simulate_ler_with_correction_factored(
            cs, make_hardware_noise_fn(D, N, IID_P), singles(), N, D,
            n_shots=shots, seed=2000)
        k = int(round(float(rr['logical_error_rate']) * shots))
        wall = round(time.time() - t0, 1)
        print(f"    iid p={IID_P}: k={k}/{shots} ({wall}s)", flush=True)
        if k >= IID_MIN_ERRORS or shots >= iid_cap:
            break
        shots = min(2 * shots, iid_cap)
    out['ler_iid_p05'] = {'k': k, 'n': shots, 'ler': k / shots,
                          'wilson95': wilson(k, shots),
                          'resolved': bool(k >= IID_MIN_ERRORS),
                          'wall_s': wall}
    # (b) channel-v2 projective sampled rate at s=0.5
    t0 = time.time()
    inst = build_projective_instrument(cs, D, N, ETA_B, sigma_a2=SA2_B)
    nf = make_meth_mc_noise_fn(N, D, GATE_PAIRS, ETA_B, sigma_a2=SA2_B)
    est = sampled_entanglement_fidelity(inst, nf, ntraj_b, seed=7)
    out['corr_v2_s05_proj'] = {
        's': S_B, 'eta': ETA_B, 'sigma_a2': SA2_B,
        'rate_1mFavg': float(est['rate_1mFavg']),
        'F_avg_se': float(est['F_avg_se']),
        'n_traj_per_ensemble': ntraj_b,
        'n_sectors': inst['report']['n_sectors'],
        'wall_s': round(time.time() - t0, 1)}
    return out


# ---------------------------------------------------------------- shards
def unit_list(shard):
    if shard == 0:
        return [("existing", name, None) for name in EXISTING]
    units = [("pilot", cell, seed) for cell in PILOT_CELLS for seed in SEEDS]
    return units[0::2] if shard == 1 else units[1::2]


def run_unit(kind, spec, seed, res, smoke=False):
    from src.decoders._common import encoder_forward
    if kind == "existing":
        key = f"existing_{spec}"
        if key in res["units"]:
            return
        npz = os.path.join(L2_DIR, f"{spec}.npz")
        z = np.load(npz, allow_pickle=True)
        cs = np.asarray(encoder_forward(npz, CONNS)[0])
        row = {"kind": "existing", "name": spec, "npz": npz,
               "min_loss_trace": float(np.min(z["loss"])),
               "final_op_evs_trace": int(np.asarray(z["op_evs"])[-1]),
               "protocol": "preset 1500 steps (earlier protocol), NOT the certified rule"}
        t0 = time.time()
        row.update(score_code(cs))
        row["wall_s"] = round(time.time() - t0, 1)
    else:
        sampler, frac = spec
        key = f"{sampler}_{frac}_s{seed}" if sampler != "full_batch" \
            else f"full_batch_s{seed}"
        if key in res["units"]:
            return
        t0 = time.time()
        tr = train_certified(sampler, frac, seed,
                             max_steps=(40 if smoke else MAX_STEPS))
        npz = os.path.join(CODEDIR, key + ".npz")
        np.savez(npz, params=tr["theta"], n=N, d=D, K=D, n_qudit=N,
                 n_layers=LAYERS, sampler=sampler, frac=frac, seed=seed,
                 certified=tr["certified"], stop_step=tr["stop_step"],
                 full_loss=tr["full_loss"], tau=TAU,
                 train_op_evs=tr["train_op_evs"],
                 confirm_op_evs=tr["confirm_op_evs"],
                 monitor_op_evs=tr["monitor_op_evs"],
                 sampled_losses=np.asarray(tr["sampled_losses"]),
                 cum_train_evs=np.asarray(tr["cum_train_evs"]),
                 monitor_steps=np.asarray(tr["monitor_steps"], dtype=int),
                 monitor_losses=np.asarray(tr["monitor_losses"]),
                 connections=np.asarray(CONNS, dtype=int))
        row = {"kind": "pilot", "sampler": sampler, "frac": frac,
               "seed": seed, "npz": npz}
        row.update({k: tr[k] for k in
                    ("certified", "stop_step", "full_loss", "train_op_evs",
                     "confirm_op_evs", "monitor_op_evs", "budget_op_evs",
                     "n_confirms", "n_premature", "train_wall_s")})
        cs = np.asarray(encoder_forward(npz, CONNS)[0])
        if smoke:
            row.update(score_code(cs, iid_cap=10_000, w1_shots=2_000,
                                  w2_shots=2_000, ntraj_b=50))
        else:
            row.update(score_code(cs))
        row["wall_s"] = round(time.time() - t0, 1)
    res["units"][key] = row
    return key, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="one tiny unit end-to-end, then exit")
    args = ap.parse_args()

    out = os.path.join(HERE, f"t1_results_shard{args.shard}.json")
    status_f = os.path.join(HERE, f"t1_status_shard{args.shard}.json")
    open(os.path.join(HERE, f"t1_shard{args.shard}.pid"), "w").write(
        str(os.getpid()))
    res = json.load(open(out)) if os.path.exists(out) else {
        "protocol": __doc__, "tau": TAU, "units": {}}

    def set_status(**kw):
        kw["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        json.dump(kw, open(status_f, "w"), indent=2)

    if args.smoke:
        set_status(state="smoke")
        t0 = time.time()
        r = run_unit("pilot", ("importance", 0.4), 999, res, smoke=True)
        print(f"[smoke] OK in {round(time.time()-t0,1)}s:",
              json.dumps(r[1], default=str)[:600], flush=True)
        set_status(state="smoke_done")
        return

    units = unit_list(args.shard)
    for i, (kind, spec, seed) in enumerate(units):
        set_status(state="running", unit=str(spec) + (f"_s{seed}" if seed else ""),
                   progress=f"{i}/{len(units)}")
        t0 = time.time()
        r = run_unit(kind, spec, seed, res)
        if r is None:
            continue
        key, row = r
        json.dump(res, open(out, "w"), indent=2)
        extra = ""
        if row["kind"] == "pilot":
            extra = (f" cert={row['certified']} stop={row['stop_step']} "
                     f"budget={row['budget_op_evs']}")
        print(f"[t1.{args.shard}] {key}:{extra} "
              f"iid={row['ler_iid_p05']['ler']:.3e} "
              f"(k={row['ler_iid_p05']['k']}) "
              f"w2={row['ler_w2']['k']}/{row['ler_w2']['n']} "
              f"corr={row['corr_v2_s05_proj']['rate_1mFavg']:.4e} "
              f"[{round(time.time()-t0,1)}s]", flush=True)
    set_status(state="done", progress=f"{len(units)}/{len(units)}")
    print(f"[t1.{args.shard}] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        json.dump({"state": "FAILED", "error": str(e),
                   "updated": time.strftime("%Y-%m-%d %H:%M:%S")},
                  open(os.path.join(HERE, "t1_status_FAILED.json"), "w"))
        raise
