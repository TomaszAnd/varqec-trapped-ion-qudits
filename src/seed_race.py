#!/usr/bin/env python3
"""Multi-start optimization ("seed race") with certified stopping.

What it implements
------------------
The multi-start protocol of paper Sec. IV D: several independent runs of the
same ansatz, each from a fresh uniform [0, 2pi) initialisation of all
parameters, optimised with Adam on the detection loss of Eq. (12) (uniform
or channel-weighted, Sec. IV D), and the certified stopping rule of Sec. IX B
in which a sampled loss below the target only *triggers* a full-batch
evaluation and a run counts as converged only when the FULL-BATCH loss is
below the target. Runs are "seeds" here: run index = PRNG seed.

Built on the differentiable backend: `ErrorModel.build_grouped` (src/errors.py)
for the wire-grouped detection set (App. D), `create_jax_encoder` /
`create_jax_encoder_variant` for the ansatz, `build_varqec_loss_weighted`
(src/jax_backend.py) for the per-operator-weighted loss, and two per-step
samplers for the detection set: `src.jax_backend.stratified_weights`
(uniform per-wire-group subsampling) and
`src.sampling.stratified_importance` (Neyman + channel-weight importance
sampling, Sec. IX A). This module only orchestrates those primitives; the
thin CLI wrapper scripts/seed_race.py re-exports its functions.

The seed-racing structure (race many initialisations, keep the one that
reaches the target fastest, with a "mercy rule" that stops a run once it
exceeds the current record) was adapted from U. Holzer's qutrit prototype,
with these differences:

- A run is certified only by its full-batch loss. When sample_frac < 1 the
  sampled loss is a cheap trigger; every trigger is re-evaluated at full
  batch before acceptance, and both values are logged. Pinned by
  tests/test_full_vs_sampled_convergence.py and tests/test_seed_race.py.
- The mercy rule activates only after the first CERTIFIED record exists
  (mercy_baseline may seed it; its provenance is stored in the metadata).
  Until then every run gets its full step budget.
- The K/4 variance coefficient of Eq. (12) is the loss's own
  (src.jax_backend.create_jax_loss_vmap_weighted).
- No identity padding of the error set: ErrorModel builds E_det without the
  identity, so no evaluations are spent on it.

Consumers
---------
scripts/seed_race.py (CLI), scripts/headline_lsweep.py,
scripts/ansatz_study_train.py and scripts/fig6_panel_train.py (build_race),
src/decoders/_common.py reads the npz this module writes, the campaign
drivers under results/campaigns/ (run_race), and tests/test_seed_race.py,
tests/test_ms1_variant.py.
"""
import os
import sys

import numpy as np

# Default per-step sampling fraction and sampler for the CLI
# (scripts/seed_race.py reads both; nothing else hard-codes a fraction).
#
# The fraction is a pure search-cost knob: certification is always full-batch,
# so any fraction that certifies yields a full-batch-quality code; it changes
# only the op-EV budget and the per-run certification yield.
#
# Measured on the UNWEIGHTED objective at n=5 ((5,3,3))_3 (--noise full): the
# subsampled loss diverges for f <= 0.3, is stochastic (~75% yield) at
# 0.32-0.35, and certifies 4/4 runs at f = 0.40 (5.4x fewer op-EVs than full
# batch). 0.50 is that knee plus one grid step of margin. On the
# channel-weighted objective the paper's protocol uses f = 0.05
# (Sec. IX A, IX C); the divergence below f ~ 0.4 is a property of the
# unweighted objective only (Sec. IX A).
CAMPAIGN_SAMPLE_FRAC = 0.5
CAMPAIGN_SAMPLER = "importance"


def evaluate_candidate(full_loss_fn, theta, target_loss):
    """Full-batch certification gate (paper Sec. IV D, Sec. IX B).

    A candidate theta -- no matter how its sampled loss looks -- is certified
    as converged if and only if its FULL-batch loss is below target_loss.

    Args:
        full_loss_fn: theta -> scalar full-batch loss (every operator of
            the detection set, unweighted or channel-weighted).
        theta: parameter array of shape (layers, params_per_layer).
        target_loss: certification threshold (paper: 1e-6 for the
            Knill-Laflamme certificate; 1.5x the measured capacity floor
            for the certified-stop budgets of Sec. IX).

    Returns (certified: bool, full_loss: float).
    """
    full_val = float(full_loss_fn(theta))
    return bool(full_val < target_loss), full_val


def build_race(d, n, distance, layers, noise="full", entangler="ms",
               connectivity="all-to-all", weighted=False, eta=0.9296, n_max=2,
               sigma_a2=0.0, jitter_phi=0.0):
    """Assemble the encoder, the detection set and the loss for one race.

    Args:
        d: qudit dimension q (also the number of codewords K = d: one
            logical qudit).
        n: number of physical qudits.
        distance: target code distance; the detection set is completed
            with weight-2 products iff distance >= 3 (paper Sec. IV C).
        layers: ansatz depth L. use_scan (lax.scan + checkpoint) is used
            for L >= 3; shallower encoders are unrolled.
        noise: "full" selects the extended error basis (basis="full" in
            ErrorModel) for the unweighted objective; anything else the
            channel error basis (basis=None). Ignored when weighted=True.
        entangler: "ms" (paper ansatz, Sec. IV A) or one of the variant
            entanglers of `create_jax_encoder_variant` (Sec. VIII B).
        connectivity: "all-to-all" (paper) or "ring".
        weighted: False -> uniform operator weights (unweighted objective);
            True -> channel-weighted objective of Sec. IV D: each operator's
            term multiplied by its channel probability (src.meth_weights).
        eta, n_max, sigma_a2, jitter_phi: calibrated-channel parameters
            forwarded to `compute_meth_group_weights_v2` when weighted=True
            (sigma_a2 > 0 adds the two-body sigma_phi x sigma_phi term;
            sigma_a2 = 0 reproduces the dephasing-only weights exactly).

    Returns (val_grad_fn, full_loss_fn, group_sizes, params_per_layer,
    connections, full_weights, sampler_group_weights):
    - val_grad_fn(theta, weights) -> (loss, grad); jit-compiled, gradient
      w.r.t. theta only. `weights` is the per-wire-group tuple of
      per-operator weights (full_weights for a full-batch step, a sampler
      draw otherwise).
    - full_loss_fn(theta) is the FULL-BATCH loss: uniform when
      weighted=False, channel-weighted when weighted=True.
    - group_sizes: list of n_g, operators per wire group (App. D).
    - full_weights: the per-group weight tuple of the full batch (ones, or
      the channel weights), jnp float64 arrays of shape (n_g,).
    - sampler_group_weights: the per-operator weights the importance sampler
      draws by (ones per group when unweighted -> Neyman-by-sqrt(group size);
      the channel weights when weighted -> draws concentrate on
      high-probability errors). numpy arrays of shape (n_g,).

    When weighted=True the model uses the channel error basis (basis=None),
    which is the basis the channel weights are defined on, NOT the extended
    "full" basis.
    """
    import jax
    import jax.numpy as jnp
    from src.errors import ErrorModel
    from src.jax_backend import (create_jax_encoder,
                                 build_varqec_loss_weighted)

    if connectivity == "all-to-all":
        connections = [[i, j] for i in range(n) for j in range(i + 1, n)]
    else:
        connections = [[i, (i + 1) % n] for i in range(n)]

    if entangler == "ms":
        enc, _, ppl = create_jax_encoder(n, d, connections=connections,
                                         use_scan=layers >= 3)
    else:
        # Variant entanglers (ms1 / zz / ls / csum / cex) for the
        # entangling-gate study (paper Sec. VIII B).
        from src.jax_backend import create_jax_encoder_variant
        enc, _, ppl = create_jax_encoder_variant(
            n, d, connections=connections, entangler=entangler,
            use_scan=layers >= 3)

    closed = distance >= 3
    # The channel-weighted objective uses the channel error basis the weights
    # are defined on; the unweighted objective keeps the extended "full"
    # basis when noise == "full".
    basis = None if weighted else ("full" if noise == "full" else None)
    model = ErrorModel(d=d, n_qudit=n, distance=distance, closed=closed,
                       basis=basis)
    grouped = model.build_grouped(verbose=False)
    group_sizes = [g['matrices'].shape[0] for g in grouped]

    # The K/4 variance coefficient of Eq. (12) lives inside
    # create_jax_loss_vmap_weighted; the error set carries no identity
    # because ErrorModel's bases exclude it.
    loss_w = build_varqec_loss_weighted(
        encoder_fn=enc, K=d, d=d, n_qudit=n, distance=distance,
        E_det_grouped=grouped)

    if weighted:
        # Per-operator channel probabilities (paper Sec. IV C, App. B).
        # sigma_a2 > 0 adds the two-body sigma_phi x sigma_phi term of the
        # angle-fluctuation channel; sigma_a2 = 0 gives the dephasing-only
        # weights bit-identically.
        from src.meth_weights import compute_meth_group_weights_v2
        sampler_group_weights, _ = compute_meth_group_weights_v2(
            d, n, distance=distance, eta=eta, sigma_a2=sigma_a2,
            phi=jitter_phi, n_max=n_max)
        if sum(len(w) for w in sampler_group_weights) != sum(group_sizes):
            raise ValueError("Meth weight / basis misalignment")
        full_weights = tuple(jnp.asarray(w, dtype=jnp.float64)
                             for w in sampler_group_weights)
    else:
        sampler_group_weights = [np.ones(m, dtype=float) for m in group_sizes]
        full_weights = tuple(jnp.ones((m,), dtype=jnp.float64) for m in group_sizes)

    # loss_w is internally @jit'd, but value_and_grad over it still re-traces
    # the grad transform every call (measured 38.7 -> 0.59 ms/step on the
    # n=3/d=3/dist=2 smoke config); the outer jit caches it.
    val_grad_fn = jax.jit(jax.value_and_grad(loss_w))  # grads w.r.t. theta only

    def full_loss_fn(theta):
        return loss_w(theta, full_weights)

    return (val_grad_fn, full_loss_fn, group_sizes, ppl, connections,
            full_weights, sampler_group_weights)


def run_race(d, n, distance, layers, num_seeds, steps, target_loss=1e-6,
             sample_frac=1.0, noise="full", entangler="ms",
             connectivity="all-to-all", mercy_baseline=None, out=None,
             lr=0.05, first_seed=0, verbose=True,
             sampler="uniform", trigger_margin=0.0, trigger_patience=1,
             weighted=False, eta=0.9296, n_max=2,
             sigma_a2=0.0, jitter_phi=0.0,
             monitor_every=0, anneal_steps=0, final_lr=None):
    """Race num_seeds initialisations; return a result dict and save an npz.

    Multi-start protocol (paper Sec. IV D): run index `seed` in
    [first_seed, first_seed + num_seeds) seeds both the uniform [0, 2pi)
    initialisation of theta (jax PRNG) and the sampler's numpy RNG. Each run
    is optimised with Adam for at most `steps` steps. The winner is the run
    reaching a CERTIFIED (full-batch) loss below target_loss in the fewest
    steps; once a certified record exists, later runs are stopped at the
    record step ("mercy rule"). If no run certifies, the best uncertified
    theta (lowest full-batch loss among the candidates described under
    monitor_every) is saved instead, with certified=False in the metadata.

    Per-step sampling (paper Sec. IX A). When sample_frac < 1 the per-step
    loss is a subsampled estimate that only TRIGGERS a full-batch
    certification; it never grants victory. Two sampler modes:
      - "uniform":     per-wire-group uniform subsampling at fraction
                       sample_frac (jax_backend.stratified_weights).
      - "importance":  Neyman + within-group importance sampling
                       (src.sampling.stratified_importance) with a total
                       budget of round(sample_frac * |E_det|) draws per step
                       over ALL wire groups (weight-1 and weight-2 operators
                       alike), drawing by the per-operator weights of the
                       loss: uniform weights for the unweighted objective
                       (-> Neyman by sqrt of group size), the channel weights
                       for the channel-weighted objective.

    Certified stopping (paper Sec. IX B). A full-batch confirmation is
    attempted once the sampled loss has been below target_loss *
    (1 + trigger_margin) for trigger_patience CONSECUTIVE steps; a rejected
    confirmation resets the consecutive count and training continues. (The
    paper's rule uses five consecutive steps and at least 50 steps between
    confirmations; here the patience is a parameter, default 1, and no
    minimum gap is enforced beyond the patience itself.)

    Learning rate: Adam at `lr`, lowered to lr/5 once the (sampled) loss
    drops below 0.5 and to lr/50 once it drops below 1e-3, with the Adam
    moments carried across the switches. Optional anneal_steps adds a final
    step-keyed cosine anneal (below).

    Op-EV accounting (paper Sec. V D): race_op_evs counts, over all runs,
    the distinct operators evaluated per sampled step (non-zero weights)
    plus |E_det| for every full-batch certification, end-of-run evaluation
    and monitor evaluation; op_evs_at_first_certified is the running total
    at the moment the winning run first certified. This lets full-batch and
    sampled races be compared at equal certified loss.

    De-noising options (default OFF; prior behaviour unchanged):
      - monitor_every=k > 0: every k steps the FULL-batch loss of the
        current theta is evaluated and recorded (counted in race_op_evs).
        The npz gains 'monitor_steps'/'monitor_losses' arrays (the
        de-noised trace), and an uncertified run's saved theta is chosen by
        the lowest MONITORED full-batch loss if that beats the full-batch
        value of the best-SAMPLED theta -- the sampled argmin is
        winner's-curse-biased when the subsampled estimator has high
        variance (measured up to ~50x max/min at fixed theta for csum at
        ((6,3,3))_3, L=2, importance sampling at f=0.2).
      - anneal_steps=m > 0: over the last m steps the LR is cosine-annealed
        from the current stage LR down to final_lr (default lr/500).
        Step-keyed, so every run sees the same final schedule regardless of
        where its plateau sits relative to the 1e-3 stage threshold (whose
        first-passage entry is otherwise a noisy-estimator event).

    Args (beyond build_race's): num_seeds, steps, target_loss, sample_frac,
    mercy_baseline (externally known record in steps, or None), out (npz
    path or None), lr, first_seed, verbose, sampler, trigger_margin,
    trigger_patience, monitor_every, anneal_steps, final_lr.

    Returns a dict with keys certified, result (winner or best-uncertified
    dict: seed, steps, theta, losses, full_loss, monitor_steps,
    monitor_losses, selected_by), record_steps, sampler, total_ops,
    race_op_evs, op_evs_at_first_certified, n_trigger_attempts,
    n_false_triggers. The npz (src.loss.save_varqec_result) stores theta,
    the sampled loss trace and the metadata listed at the end of this
    function; src/decoders/_common.py rebuilds encoders from it.
    """
    import jax
    import jax.numpy as jnp
    import optax
    from src.jax_backend import stratified_weights
    from src.loss import save_varqec_result

    (val_grad_fn, full_loss_fn, group_sizes, ppl, connections,
     full_weights, sampler_group_weights) = build_race(
        d, n, distance, layers, noise=noise, entangler=entangler,
        connectivity=connectivity, weighted=weighted, eta=eta, n_max=n_max,
        sigma_a2=sigma_a2, jitter_phi=jitter_phi)

    ones = full_weights  # full-batch weights: uniform, or the channel weights
    subsampling = sample_frac < 1.0
    total_ops = int(sum(group_sizes))

    # Sampler setup. group_weights are the per-operator loss weights the
    # importance sampler draws by: ones (unweighted -> Neyman by sqrt of
    # group size) or the channel weights (-> concentrate on probable errors).
    if sampler == "importance":
        from src.sampling.stratified_importance import (
            make_stratified_importance_weights)
        group_weights = sampler_group_weights
        is_budget = max(1, int(round(sample_frac * total_ops)))
    elif sampler != "uniform":
        raise ValueError(f"sampler must be 'uniform' or 'importance', got '{sampler}'")

    def _sample_weights(rng):
        if sampler == "importance":
            return make_stratified_importance_weights(group_weights, is_budget, rng)
        return stratified_weights(group_sizes, sample_frac, rng)

    def _op_evs(weights):
        # Device cost of a step = number of distinct operators evaluated
        # (paper Sec. V D); the loss itself still multiplies the zero-weight
        # operators by 0, so this is the accounting cost, not wall time.
        return int(sum(int(np.count_nonzero(np.asarray(w))) for w in weights))

    # Op-EV accounting across the whole race.
    race_op_evs = 0
    op_evs_at_first_certified = None
    # Trigger accounting: a "trigger" is a full-batch certification attempt
    # armed by the patience gate; a "false trigger" is one the full batch rejects.
    n_trigger_attempts = 0
    n_false_triggers = 0

    # The record starts empty unless the caller supplies a baseline with
    # explicit provenance; no run is stopped early before a record exists.
    record_steps = mercy_baseline
    record_from_baseline = mercy_baseline is not None
    winner = None  # dict(seed, steps, theta, losses, full_loss)
    best_uncert = None  # dict(seed, steps, theta, losses, full_loss)

    for seed in range(first_seed, first_seed + num_seeds):
        base_key = jax.random.PRNGKey(seed)
        init_key, _ = jax.random.split(base_key)
        theta = jax.random.uniform(init_key, (layers, ppl),
                                   minval=0.0, maxval=2 * np.pi)
        rng = np.random.default_rng(seed)

        # Three-stage LR schedule (lr / lr/5 below loss 0.5 / lr/50 below
        # 1e-3, i.e. 0.05 / 0.01 / 0.001 at the default lr), adapted from
        # U. Holzer's qutrit prototype; unlike the prototype the Adam
        # first/second moments are CARRIED across the switches via
        # optax.inject_hyperparams instead of re-initialising the optimizer
        # (a re-init zeroes the moments and restarts bias correction).
        optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=lr)
        opt_state = optimizer.init(theta)
        lr_stage = 0

        seed_losses = []
        seed_best_sampled = np.inf
        seed_best_theta = theta
        certified_here = False
        consec_below = 0  # consecutive steps under the trigger margin
        monitor_steps_l, monitor_losses_l = [], []
        mon_best_val, mon_best_theta = np.inf, None
        stage_lrs = (lr, lr / 5, lr / 50)
        f_lr = final_lr if final_lr is not None else lr / 500

        for step in range(steps):
            # Mercy rule: only once a certified record exists.
            if record_steps is not None and step >= record_steps:
                if verbose:
                    print(f"seed {seed}: eliminated at step {step} "
                          f"(record {record_steps})")
                break

            if subsampling:
                weights = _sample_weights(rng)
            else:
                weights = ones
            race_op_evs += _op_evs(weights)
            loss_val, grads = val_grad_fn(theta, weights)
            lv = float(loss_val)
            seed_losses.append(lv)
            if lv < seed_best_sampled:
                seed_best_sampled = lv
                seed_best_theta = theta

            # De-noised monitor trace: periodic full-batch eval of the
            # PRE-update theta the sampled loss belongs to.
            if monitor_every and (step % monitor_every == 0
                                  or step == steps - 1):
                mon_val = float(full_loss_fn(theta))
                race_op_evs += total_ops
                monitor_steps_l.append(step)
                monitor_losses_l.append(mon_val)
                if mon_val < mon_best_val:
                    mon_best_val = mon_val
                    mon_best_theta = theta

            # Trigger gating (paper Sec. IX B): the sampled loss must sit under
            # the margin for trigger_patience CONSECUTIVE steps before the
            # expensive full-batch certification is attempted. Suppresses false
            # triggers from a noisy single-step sampled estimate.
            if lv < target_loss * (1.0 + trigger_margin):
                consec_below += 1
            else:
                consec_below = 0

            # Victory check on the PRE-update theta the loss belongs to.
            if consec_below >= trigger_patience:
                certified, full_val = evaluate_candidate(
                    full_loss_fn, theta, target_loss)
                race_op_evs += total_ops  # a full-batch certification eval
                n_trigger_attempts += 1
                if not certified:
                    n_false_triggers += 1
                if verbose:
                    print(f"seed {seed} step {step}: sampled={lv:.3e}, "
                          f"full-batch={full_val:.3e} -> "
                          f"{'CERTIFIED' if certified else 'REJECTED'}")
                if certified:
                    certified_here = True
                    if record_steps is None or step < record_steps:
                        record_steps = step
                        record_from_baseline = False
                        if op_evs_at_first_certified is None:
                            op_evs_at_first_certified = race_op_evs
                        winner = {"seed": seed, "steps": step,
                                  "theta": np.array(theta),
                                  "losses": list(seed_losses),
                                  "full_loss": full_val,
                                  "monitor_steps": list(monitor_steps_l),
                                  "monitor_losses": list(monitor_losses_l),
                                  "selected_by": "certified"}
                    break
                # Not certified: the sampled trigger was a false alarm --
                # keep training this run. (No minimum gap to the next
                # confirmation beyond the patience count itself.)
                consec_below = 0

            updates, opt_state = optimizer.update(grads, opt_state, theta)
            theta = optax.apply_updates(theta, updates)

            # In-place hyperparams mutation is only sound in this EAGER
            # loop: opt_state is a concrete pytree here. If this update
            # step is ever moved inside jax.jit / lax.scan, the mutation
            # would be silently dropped (traced state is immutable) -- use
            # optax.tree_utils/functional replacement instead there.
            if lr_stage == 0 and lv < 0.5:
                opt_state.hyperparams['learning_rate'] = lr / 5
                lr_stage = 1
            if lr_stage <= 1 and lv < 1e-3:
                opt_state.hyperparams['learning_rate'] = lr / 50
                lr_stage = 2

            # Step-keyed final cosine anneal: overrides the loss-keyed stage LR
            # over the last anneal_steps steps.
            if anneal_steps and step >= steps - anneal_steps:
                t_frac = (step - (steps - anneal_steps)) / max(1, anneal_steps)
                base = stage_lrs[lr_stage]
                opt_state.hyperparams['learning_rate'] = (
                    f_lr + (base - f_lr) * 0.5 * (1.0 + np.cos(np.pi * t_frac)))

            if verbose and step % 200 == 0:
                print(f"seed {seed} step {step:5d} | loss={lv:.4e}")

        if not certified_here:
            _, full_val = evaluate_candidate(full_loss_fn, seed_best_theta,
                                             target_loss)
            race_op_evs += total_ops  # end-of-seed best-uncertified full eval
            sel_theta, sel_val, sel_by = seed_best_theta, full_val, "sampled_argmin"
            if mon_best_theta is not None and mon_best_val < full_val:
                # De-noised selection: the monitored full-batch minimum beats
                # the full-batch value of the noisy sampled argmin.
                sel_theta, sel_val, sel_by = mon_best_theta, mon_best_val, "monitor"
            if best_uncert is None or sel_val < best_uncert["full_loss"]:
                best_uncert = {"seed": seed, "steps": len(seed_losses),
                               "theta": np.array(sel_theta),
                               "losses": list(seed_losses),
                               "full_loss": sel_val,
                               "monitor_steps": list(monitor_steps_l),
                               "monitor_losses": list(monitor_losses_l),
                               "selected_by": sel_by}

    result = winner if winner is not None else best_uncert
    certified = winner is not None
    if verbose:
        if certified:
            print(f"\nWINNER: seed {result['seed']} certified "
                  f"full-batch loss {result['full_loss']:.3e} "
                  f"at step {result['steps']}")
        elif result is not None:
            print(f"\nNo seed certified; best uncertified full-batch loss "
                  f"{result['full_loss']:.3e} (seed {result['seed']})")

    if result is not None and out is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        save_varqec_result(
            out, result["theta"], result["losses"],
            f"native_d{d}", distance, layers,
            metadata={"K": d, "n_qudit": n, "d": d,
                      "seed": result["seed"],
                      "steps_to_target": result["steps"],
                      "certified": certified,
                      "certified_full_loss": result["full_loss"],
                      "target_loss": target_loss,
                      "sample_frac": sample_frac,
                      "noise": noise, "entangler": entangler,
                      "connectivity": connectivity,
                      "num_seeds": num_seeds,
                      "mercy_baseline": (mercy_baseline
                                         if mercy_baseline is not None
                                         else -1),
                      "mercy_baseline_used": record_from_baseline,
                      "backend": "jax",
                      "sigma_a2": sigma_a2,
                      "jitter_phi": jitter_phi,
                      "sampler": sampler,
                      "trigger_margin": trigger_margin,
                      "trigger_patience": trigger_patience,
                      "race_op_evs": race_op_evs,
                      "monitor_every": monitor_every,
                      "anneal_steps": anneal_steps,
                      "final_lr": (final_lr if final_lr is not None
                                   else (lr / 500 if anneal_steps else -1.0)),
                      "monitor_steps": np.asarray(
                          result.get("monitor_steps", []), dtype=int),
                      "monitor_losses": np.asarray(
                          result.get("monitor_losses", []), dtype=float),
                      "selected_by": result.get("selected_by", "sampled_argmin"),
                      "connections": np.asarray(connections, dtype=int)})

    return {"certified": certified, "result": result,
            "record_steps": record_steps,
            "sampler": sampler,
            "total_ops": total_ops,
            "race_op_evs": race_op_evs,
            "op_evs_at_first_certified": op_evs_at_first_certified,
            "n_trigger_attempts": n_trigger_attempts,
            "n_false_triggers": n_false_triggers}
