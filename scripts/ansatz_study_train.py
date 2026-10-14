#!/usr/bin/env python3
"""Task B — ansatz/entangler study trainer (docs/ANSATZ_STUDY_STATUS.md).

One single-seed weighted run per entangler {ms, zz, ls, csum, cex} at the
Task-A-chosen (n, dist=3, q=3, L=16, a2a) setting. The training objective is
FIXED across gates: the weighted corrected-Meth KL with 1 seed + weighted
importance sampling @0.2 (Sec IX best-practice line) — the same instrument as
fig8_weighted_train.train_one (per-step sampled loss; full-batch weighted
loss logged every 25 steps against cumulative op-EVs), with the entangler
passed through src.seed_race.build_race.

Per gate writes <outdir>/<gate>.npz: steps/loss/op_evs (full-batch log),
sampled_loss (per step), params + decoder-compatible metadata (entangler,
connections — encoder_forward dispatches on the entangler field). Resumable:
existing <gate>.npz with matching steps are skipped. Status in
<outdir>/train_status.json.

Usage: ansatz_study_train.py --n 6 --steps 2000 [--gates ms,zz,...]
"""
from __future__ import annotations
import os, sys, time, json, argparse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import src.jax_backend  # enable 64-bit JAX before building encoders
import numpy as np

GATES = ["ms", "zz", "ls", "csum", "cex"]
FRAC, LOG_EVERY = 0.2, 25


def train_one(gate, args):
    import jax, jax.numpy as jnp, optax
    from src.seed_race import build_race
    from src.sampling.stratified_importance import (
        make_stratified_importance_weights)

    (val_grad_fn, full_loss_fn, group_sizes, ppl, connections,
     full_weights, sgw) = build_race(args.d, args.n, 3, args.layers,
                                     weighted=True, entangler=gate,
                                     connectivity="all-to-all")
    total_ops = int(sum(group_sizes))
    budget = max(1, int(round(FRAC * total_ops)))
    theta = jax.random.uniform(jax.random.PRNGKey(args.seed),
                               (args.layers, ppl), minval=0.0,
                               maxval=2 * np.pi)
    rng = np.random.default_rng(args.seed)
    opt = optax.inject_hyperparams(optax.adam)(learning_rate=args.lr)
    st = opt.init(theta); stage = 0
    steps_l, loss_l, opev_l, sampled_l = [], [], [], []
    cum_ev = 0
    for step in range(args.steps):
        w = make_stratified_importance_weights(sgw, budget, rng)
        cum_ev += int(sum(int(np.count_nonzero(np.asarray(x))) for x in w))
        lv, g = val_grad_fn(theta, w)
        sampled_l.append(float(lv))
        if step % LOG_EVERY == 0 or step == args.steps - 1:
            fl = float(full_loss_fn(theta)); cum_ev += total_ops
            steps_l.append(step); loss_l.append(fl); opev_l.append(cum_ev)
        upd, st = opt.update(g, st, theta)
        theta = optax.apply_updates(theta, upd)
        lvf = float(lv)
        if stage == 0 and lvf < 0.5:
            st.hyperparams['learning_rate'] = args.lr / 5; stage = 1
        if stage <= 1 and lvf < 1e-3:
            st.hyperparams['learning_rate'] = args.lr / 50; stage = 2
    return (steps_l, loss_l, opev_l, sampled_l, total_ops, ppl, connections,
            np.asarray(theta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--gates", default=",".join(GATES))
    ap.add_argument("--outdir", default=os.path.join(
        REPO, "results", "campaigns", "ansatz_study"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    spath = os.path.join(args.outdir, "train_status.json")
    status = json.load(open(spath)) if os.path.exists(spath) else {}
    status["config"] = vars(args)
    for gate in args.gates.split(","):
        out = os.path.join(args.outdir, f"{gate}.npz")
        if os.path.exists(out):
            prev = np.load(out, allow_pickle=True)
            if int(prev["train_steps"]) == args.steps:
                print(f"[study] {gate} already trained -- skip", flush=True)
                continue
        t = time.time()
        print(f"[study] training {gate} (n={args.n}, {args.steps} steps, "
              f"importance@{FRAC}) ...", flush=True)
        status[gate] = {"status": "RUNNING"}
        json.dump(status, open(spath, "w"), indent=2)
        try:
            (steps_l, loss_l, opev_l, sampled_l, tops, ppl, conns,
             theta) = train_one(gate, args)
            np.savez(out, steps=np.array(steps_l), loss=np.array(loss_l),
                     op_evs=np.array(opev_l),
                     sampled_loss=np.array(sampled_l),
                     total_ops=tops, train_steps=args.steps,
                     params=theta, d=args.d, n_qudit=args.n, K=args.d,
                     n_layers=args.layers, entangler=gate,
                     sampler="importance", sample_frac=FRAC,
                     seed=args.seed, ppl=ppl,
                     connections=np.asarray(conns, dtype=int))
            status[gate] = {"status": "DONE",
                            "wall_s": round(time.time() - t, 1),
                            "min_full_loss": float(min(loss_l)),
                            "final_full_loss": float(loss_l[-1]),
                            "final_op_evs": int(opev_l[-1])}
            print(f"[study] {gate} DONE min_loss={min(loss_l):.3e} "
                  f"final_opEV={opev_l[-1]} "
                  f"({status[gate]['wall_s']}s)", flush=True)
        except Exception as e:
            status[gate] = {"status": f"FAILED: {e}",
                            "wall_s": round(time.time() - t, 1)}
            print(f"[study] {gate} FAILED: {e}", flush=True)
        json.dump(status, open(spath, "w"), indent=2)
    print("[study] TRAIN DONE", flush=True)


if __name__ == "__main__":
    main()
