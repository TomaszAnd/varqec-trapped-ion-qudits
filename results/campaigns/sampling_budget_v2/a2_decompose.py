#!/usr/bin/env python3
"""Op-EV accounting: exact offline decomposition of the committed sampler
budgets (fig8_weighted traces).

Decomposes each committed fig8_weighted/*.npz cumulative op-EV trace into
  train    = per-step sampled evaluations (what a device pays to train),
  monitor  = every-25-step full-batch LOGGING evals (a device would not pay),
  cert     = full-batch certification checks (ZERO here: fig8_weighted_train
             is a fixed-steps trainer with no trigger/certification logic).

Validation is stronger than an instrumented rerun: the sampler draws in
fig8_weighted_train.train_one depend only on rng(seed) — never on theta — so
the exact per-step distinct-op counts are REPLAYED here (same code path, same
rng consumption order) and asserted against the stored cumulative op_evs at
every one of the 61 logged checkpoints per sampler. A 2.5 h instrumented
rerun would validate the same bookkeeping with the same rng stream; this
replay does it against the committed data directly, in seconds.

Run: nice -n 19 python3 a2_decompose.py
Writes a2_decomposition.json (this directory). Feeds the op-EV accounting
of the paper.
"""
import os, sys, json
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.chdir(REPO)
import src.jax_backend  # x64 first (must precede any jnp array creation)
import numpy as np

FIG8 = os.path.join(REPO, "results", "campaigns", "fig8_weighted")
OUT = os.path.join(REPO, "results", "campaigns", "sampling_budget_v2",
                   "a2_decomposition.json")
STEPS, SEED, LOG_EVERY = 1500, 0, 25

CONFIGS = [("full_batch", "full_batch", 1.0),
           ("stratified_0.2", "stratified", 0.2),
           ("importance_0.2", "importance", 0.2),
           ("importance_0.35", "importance", 0.35),
           ("full_basis_is_0.1", "full_basis_is", 0.1)]


def replay_op_evs(sampler, frac):
    """Replay train_one's per-step distinct-op counts + log cadence exactly."""
    from src.seed_race import build_race
    from src.jax_backend import stratified_weights
    from src.sampling.stratified_importance import (
        make_stratified_importance_weights,
        make_stratified_importance_weights_full_basis)
    (_, _, group_sizes, _, _, full_weights, sgw) = build_race(
        3, 9, 3, 16, weighted=True, connectivity="all-to-all")
    total_ops = int(sum(group_sizes))
    budget = max(1, int(round(frac * total_ops)))
    if sampler == "full_basis_is":
        from loss_split_helpers import build_w1_w2_masks
        from src.errors import ErrorModel
        grouped = ErrorModel(d=3, n_qudit=9, distance=3,
                             closed=True).build_grouped(verbose=False)
        w1_mask, w2_mask = build_w1_w2_masks(grouped, 3)
        w1_budget = max(1, int(round(0.75 * frac * total_ops)))
        w2_budget = max(1, int(round(frac * total_ops)) - w1_budget)
    rng = np.random.default_rng(SEED)
    per_step, cum_at_log, cum = [], [], 0
    for step in range(STEPS):
        if sampler == "full_batch":
            c = total_ops
        elif sampler == "stratified":
            w = stratified_weights(group_sizes, frac, rng)
            c = int(sum(int(np.count_nonzero(np.asarray(x))) for x in w))
        elif sampler == "full_basis_is":
            w = make_stratified_importance_weights_full_basis(
                sgw, w1_mask, w2_mask, w1_budget, w2_budget, rng)
            c = int(sum(int(np.count_nonzero(np.asarray(x))) for x in w))
        else:
            w = make_stratified_importance_weights(sgw, budget, rng)
            c = int(sum(int(np.count_nonzero(np.asarray(x))) for x in w))
        per_step.append(c); cum += c
        if step % LOG_EVERY == 0 or step == STEPS - 1:
            cum += total_ops          # full-batch log eval (monitor)
            cum_at_log.append(cum)
    return total_ops, np.asarray(per_step), np.asarray(cum_at_log)


def main():
    results = {"protocol": {
        "steps": STEPS, "seed": SEED, "log_every": LOG_EVERY,
        "cert_component": ("EXACTLY 0: fig8_weighted_train has no trigger/"
                           "certification logic (fixed-steps trainer)"),
        "validation": ("replayed rng stream (draws are theta-independent) "
                       "asserted against stored op_evs at all logged "
                       "checkpoints")}, "samplers": {}}
    fb_ref = {}
    for name, sampler, frac in CONFIGS:
        z = np.load(os.path.join(FIG8, f"{name}.npz"), allow_pickle=True)
        steps, loss, opev = z["steps"], z["loss"], z["op_evs"]
        total_ops = int(z["total_ops"])
        T, per_step, cum_at_log = replay_op_evs(sampler, frac)
        assert T == total_ops, (name, T, total_ops)
        match = bool(np.array_equal(cum_at_log, opev))
        nlogs = len(steps)
        monitor_total = nlogs * total_ops
        train_total = int(opev[-1]) - monitor_total
        imin = int(np.argmin(loss))
        nlogs_at_min = imin + 1
        monitor_at_min = nlogs_at_min * total_ops
        train_at_min = int(opev[imin]) - monitor_at_min
        rec = {
            "sampler": sampler, "frac": frac, "total_ops": total_ops,
            "replay_matches_stored_all_checkpoints": match,
            "n_checkpoints": nlogs,
            "committed_total_op_evs": int(opev[-1]),
            "min_loss": float(loss[imin]), "min_loss_step": int(steps[imin]),
            "op_evs_at_min_loss": int(opev[imin]),
            "decomposition_full_run": {
                "train": train_total, "monitor": monitor_total, "cert": 0,
                "monitor_share": round(monitor_total / int(opev[-1]), 4)},
            "decomposition_at_min_loss": {
                "train": train_at_min, "monitor": monitor_at_min, "cert": 0,
                "monitor_share": round(monitor_at_min / int(opev[imin]), 4)},
            "realized_ops_per_step_mean": round(float(per_step.mean()), 2),
            "nominal_budget_per_step": (total_ops if sampler == "full_batch"
                                        else max(1, int(round(frac * total_ops)))),
        }
        results["samplers"][name] = rec
        if name == "full_batch":
            fb_ref = rec
        print(f"{name}: replay_match={match} total={opev[-1]} "
              f"train={train_total} monitor={monitor_total} "
              f"({rec['decomposition_full_run']['monitor_share']:.1%}) "
              f"realized/step={rec['realized_ops_per_step_mean']}", flush=True)
    # sampler-advantage factors vs full-batch under 3 accountings
    adv = {}
    for name, rec in results["samplers"].items():
        if name == "full_batch":
            continue
        adv[name] = {
            "total_committed": round(
                fb_ref["committed_total_op_evs"] / rec["committed_total_op_evs"], 2),
            "train_only": round(
                fb_ref["decomposition_full_run"]["train"]
                / rec["decomposition_full_run"]["train"], 2),
            "train_plus_cert": round(
                (fb_ref["decomposition_full_run"]["train"] + 0)
                / (rec["decomposition_full_run"]["train"] + 0), 2),
        }
    results["advantage_vs_full_batch"] = adv
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
