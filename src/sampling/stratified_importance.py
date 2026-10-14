#!/usr/bin/env python3
"""Stratified importance sampling of the detection set (paper Sec. IX A).

What it implements
------------------
The per-step sampling scheme behind the "importance" and "full-basis
importance" strategies of Sec. IX A. The channel-weighted detection loss
(Sec. IV D) is a sum over detection operators e of w_e f_e, where w_e is the
operator's channel probability and f_e its bracketed term in Eq. (12). The
operators are stored in wire groups (App. D: one group per set of qudits
acted on). Each gradient step this module

  1. splits a total draw budget B across the wire groups by Neyman
     allocation, B_g proportional to sqrt(sum_{e in g} w_e^2) (the stratum
     spread), with at least one draw for every group of non-zero mass
     (`neyman_group_budgets`);
  2. inside group g draws B_g operators with replacement with probability
     p_e = w_e / W_g, W_g = sum_{e in g} w_e (Rosalin-style importance
     sampling, Arrasmith et al. arXiv:2004.06252), and returns per-operator
     weights (W_g / B_g) * n_e, where n_e is how often e was drawn
     (`make_stratified_importance_weights`).

Unbiasedness. E[ sum_drawn (W_g/B_g) f_e ] = W_g sum_e p_e f_e
= sum_e w_e f_e, the exact group contribution; summing over groups gives the
full channel-weighted loss. Dividing each drawn term by its draw probability
is exactly the "each drawn operator's term is divided by its draw
probability" rule of Sec. IX A. Operators of zero channel weight have p_e = 0
and are never drawn, which is why they never count as op-EVs (Sec. V D).

The returned object is the `weights_per_group` tuple consumed by
`src.jax_backend.create_jax_loss_vmap_weighted` /
`build_varqec_loss_weighted` (the same interface as the uniform sampler
`src.jax_backend.stratified_weights`): passing all-ones recovers the full
loss, so the sampler is a drop-in replacement for the per-step weights.

`make_stratified_importance_weights_full_basis` is the "full-basis
importance" variant of Sec. IX A: the weight-1 and weight-2 sectors each get
their own budget and are both sampled, instead of evaluating the weight-1
sector in full.

Consumers
---------
src/seed_race.py (the "importance" sampler of run_race, single budget over
all groups); scripts/ansatz_study_train.py; the sampler-study driver
results/campaigns/r10_sampler/t1_pilot.py (weight-1 sector in full, weight-2
sampled; and the full-basis variant) and results/campaigns/sampling_budget_v2/;
tests/test_stratified_importance.py pins the budgets and unbiasedness.
"""
from __future__ import annotations
from typing import List

import numpy as np


def neyman_group_budgets(group_weights: List[np.ndarray], total_budget: int):
    """Split a total draw budget across wire groups by Neyman allocation.

    Args:
        group_weights: list aligned with the wire groups of
            `ErrorModel.build_grouped()`; entry g is an (n_g,) array of
            non-negative per-operator channel weights w_e.
        total_budget: total number of draws per step, B.

    Returns:
        list of ints, one per group: B_g = round(B * s_g / sum_h s_h) with
        s_g = sqrt(sum_{e in g} w_e^2). Empty or zero-mass groups get 0;
        every group with positive mass gets at least 1, so each stratum is
        represented every step. The realized total can therefore differ
        slightly from B (rounding plus the floor of 1).
    """
    spreads = np.array([float(np.sqrt(np.sum(np.asarray(Wg) ** 2)))
                        for Wg in group_weights])
    Z = spreads.sum()
    budgets = []
    for g, Wg in enumerate(group_weights):
        if len(Wg) == 0 or spreads[g] <= 0.0:
            budgets.append(0)
        else:
            budgets.append(max(1, int(round(total_budget * spreads[g] / Z))))
    return budgets


def make_stratified_importance_weights(group_weights: List[np.ndarray],
                                       total_budget: int,
                                       rng: np.random.Generator,
                                       dtype=None):
    """Draw one step's importance sample and return its per-operator weights.

    Args:
        group_weights: list aligned with the wire groups; group_weights[g]
            is the (n_g,) array of per-operator channel weights w_e (zero on
            operators the channel never produces; those get draw
            probability 0).
        total_budget: total draws per step, B, split across groups by
            `neyman_group_budgets`.
        rng: numpy random Generator (seeded per run by the caller).
        dtype: jnp dtype of the returned arrays (default jnp.float64).

    Returns:
        tuple of per-group jnp arrays of shape (n_g,). Operator e of group g
        drawn n_e times gets weight (W_g / B_g) * n_e, W_g = sum_{e in g}
        w_e; undrawn operators get 0. Used as `weights_per_group` in
        `create_jax_loss_vmap_weighted`, the weighted loss is then an
        unbiased single-step estimate of the full channel-weighted loss
        sum_e w_e f_e (see module docstring). The number of non-zero
        entries is the step's op-EV count (distinct operators evaluated).
    """
    import jax.numpy as jnp
    if dtype is None:
        dtype = jnp.float64

    budgets = neyman_group_budgets(group_weights, total_budget)
    out = []
    for g, Wg in enumerate(group_weights):
        Wg = np.asarray(Wg, dtype=float)
        n_g = len(Wg)
        w = np.zeros(n_g, dtype=float)
        Bg = budgets[g]
        total_w = Wg.sum()
        if n_g == 0 or Bg == 0 or total_w <= 0.0:
            out.append(jnp.asarray(w, dtype=dtype))
            continue
        p = Wg / total_w                      # within-group draw prob ∝ w_e
        draws = rng.choice(n_g, size=Bg, replace=True, p=p)
        counts = np.bincount(draws, minlength=n_g)
        # Each draw of e contributes f_e / (B_g p_e) = (W_g / B_g) f_e;
        # n_e draws multiply that by the count.
        w = (total_w / Bg) * counts
        out.append(jnp.asarray(w, dtype=dtype))
    return tuple(out)


def make_stratified_importance_weights_full_basis(group_weights: List[np.ndarray],
                                                  w1_mask: List[np.ndarray],
                                                  w2_mask: List[np.ndarray],
                                                  w1_budget: int,
                                                  w2_budget: int,
                                                  rng: np.random.Generator,
                                                  dtype=None):
    """Full-basis importance sampling (Sec. IX A): sample BOTH sectors.

    The plain "importance" strategy evaluates the weight-1 sector in full
    and samples only the weight-2 products. This variant gives each sector
    its own draw budget and runs `make_stratified_importance_weights`
    independently on the weight-1 and weight-2 portions of every group,
    then adds the two weight tuples. The two supports are disjoint inside a
    group, so each sector's estimate is unbiased for its own contribution
    and the sum is unbiased for the full channel-weighted loss.

    Args:
        group_weights: per-group (n_g,) channel weights, as above.
        w1_mask, w2_mask: per-group (n_g,) arrays, > 0 on the operators of
            weight 1 resp. weight 2 (an operator belongs to exactly one).
        w1_budget, w2_budget: draws per step for each sector.
        rng, dtype: as in `make_stratified_importance_weights`.

    Returns:
        tuple of per-group (n_g,) jnp weight arrays, the drop-in
        `weights_per_group` for the weighted loss.
    """
    import jax.numpy as jnp
    if dtype is None:
        dtype = jnp.float64
    w1_gw = [np.where(np.asarray(m) > 0, np.asarray(gw, dtype=float), 0.0)
             for gw, m in zip(group_weights, w1_mask)]
    w2_gw = [np.where(np.asarray(m) > 0, np.asarray(gw, dtype=float), 0.0)
             for gw, m in zip(group_weights, w2_mask)]
    w1_w = make_stratified_importance_weights(w1_gw, w1_budget, rng, dtype=dtype)
    w2_w = make_stratified_importance_weights(w2_gw, w2_budget, rng, dtype=dtype)
    return tuple(jnp.asarray(np.asarray(w1_w[g]) + np.asarray(w2_w[g]), dtype=dtype)
                 for g in range(len(group_weights)))
