"""unbiasedness + structural tests for stratified-importance sampling.

Pure-numpy / small-jnp; no training, fast even under machine contention.
"""
from __future__ import annotations
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.sampling.stratified_importance import (
    make_stratified_importance_weights, neyman_group_budgets,
    make_stratified_importance_weights_full_basis)


def _synthetic(rng, n_groups=6):
    """Random non-negative per-group weights (some ops zero) + per-op f."""
    gw, gf = [], []
    for _ in range(n_groups):
        n_g = int(rng.integers(1, 20))
        w = rng.random(n_g)
        w[rng.random(n_g) < 0.3] = 0.0   # some zero-weight ops
        gw.append(w)
        gf.append(rng.standard_normal(n_g))   # per-op loss contributions
    return gw, gf


def test_unbiased_estimator():
    """E[ Σ_g Σ_e weight_g[e]·f_e ] over RNG = Σ_g Σ_e w_e·f_e (the full loss)."""
    rng = np.random.default_rng(0)
    gw, gf = _synthetic(rng)
    true = sum(float(np.dot(w, f)) for w, f in zip(gw, gf))

    n_rep = 4000
    est = np.zeros(n_rep)
    srng = np.random.default_rng(123)
    for r in range(n_rep):
        wt = make_stratified_importance_weights(gw, total_budget=40, rng=srng)
        est[r] = sum(float(np.dot(np.asarray(w), f)) for w, f in zip(wt, gf))

    mean = est.mean()
    sem = est.std() / np.sqrt(n_rep)
    print(f"\n  true={true:.5f}, IS mean={mean:.5f} ± {sem:.5f} (n={n_rep}), "
          f"|bias|={abs(mean-true):.5f}, bias/sem={abs(mean-true)/sem:.2f}")
    # Unbiased: mean within ~4 SEM of the true value.
    assert abs(mean - true) < 4 * sem + 1e-9, (
        f"estimator biased: mean {mean} vs true {true} ({abs(mean-true)/sem:.1f} SEM)")


def test_neyman_budgets_sum_and_floor():
    """Budgets respect the total (approximately) and floor non-empty groups at ≥1."""
    rng = np.random.default_rng(1)
    gw, _ = _synthetic(rng)
    B = 50
    budgets = neyman_group_budgets(gw, B)
    # every group with positive mass gets ≥1
    for w, b in zip(gw, budgets):
        if np.asarray(w).sum() > 0:
            assert b >= 1
    # total is in the right ballpark (rounding + floors perturb it modestly)
    assert 0.5 * B <= sum(budgets) <= 2.0 * B, f"budgets sum {sum(budgets)} vs B={B}"
    print(f"\n  budgets={budgets}, sum={sum(budgets)} (target {B})")


def test_full_basis_unbiased():
    """make_..._full_basis (samples w1+w2 sectors) is unbiased for the FULL
    weighted sum Σ_g Σ_e w_e f_e across both sectors."""
    rng = np.random.default_rng(7)
    n_groups = 6
    gw, gf, w1m, w2m = [], [], [], []
    for _ in range(n_groups):
        n_g = int(rng.integers(2, 16))
        w = rng.random(n_g)
        gw.append(w); gf.append(rng.standard_normal(n_g))
        # split each group's ops into a w1 sub-mask and w2 sub-mask (disjoint)
        m1 = np.zeros(n_g); m2 = np.zeros(n_g)
        k = n_g // 2
        m1[:k] = 1.0; m2[k:] = 1.0
        w1m.append(m1); w2m.append(m2)
    true = sum(float(np.dot(w, f)) for w, f in zip(gw, gf))

    n_rep = 4000
    est = np.zeros(n_rep)
    srng = np.random.default_rng(321)
    for r in range(n_rep):
        wt = make_stratified_importance_weights_full_basis(
            gw, w1m, w2m, w1_budget=15, w2_budget=15, rng=srng)
        est[r] = sum(float(np.dot(np.asarray(w), f)) for w, f in zip(wt, gf))
    mean = est.mean(); sem = est.std() / np.sqrt(n_rep)
    print(f"\n  full-basis: true={true:.5f}, IS mean={mean:.5f} ± {sem:.5f}, "
          f"bias/sem={abs(mean-true)/sem:.2f}")
    assert abs(mean - true) < 4 * sem + 1e-9, (
        f"full-basis estimator biased: {mean} vs {true}")


def test_zero_weight_group_gets_zero():
    """A group of all-zero weights contributes a zero weight array (no draws)."""
    import jax.numpy as jnp
    gw = [np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.5])]
    wt = make_stratified_importance_weights(gw, total_budget=20,
                                            rng=np.random.default_rng(2))
    assert np.allclose(np.asarray(wt[0]), 0.0)
    assert float(np.asarray(wt[1]).sum()) > 0.0


def test_full_basis_is_wires_with_meth_weights():
    """B1: the full-basis-IS sampler (samples the weight-1 sector too) composes with
    the CORRECTED Meth channel weights + the default-basis w1/w2 masks -- the path the
    corrected Fig 8 uses for the cheapest strategy. Shapes align; realized op-EV budget
    tracks w1_budget+w2_budget; estimator is unbiased for the full weighted loss."""
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from src.errors import ErrorModel
    from src.meth_weights import compute_meth_group_weights
    from loss_split_helpers import build_w1_w2_masks
    d, n = 3, 5
    grouped = ErrorModel(d=d, n_qudit=n, distance=3, closed=True).build_grouped(verbose=False)
    sizes = [g['matrices'].shape[0] for g in grouped]
    sgw, flat = compute_meth_group_weights(d, n, distance=3)
    w1m, w2m = build_w1_w2_masks(grouped, d)
    assert [len(np.asarray(m)) for m in w1m] == sizes  # masks align with basis
    rng = np.random.default_rng(0)
    w1b, w2b = 20, 25
    # unbiasedness: average many single-draw estimates of Sum_i lambda_i f_i
    f = np.concatenate([np.asarray(x) for x in sgw]) * 0 + rng.random(sum(sizes))
    offs = np.concatenate([[0], np.cumsum(sizes)])
    true = float(np.sum(flat * f))
    ests, nz = [], []
    for s in range(4000):
        wt = make_stratified_importance_weights_full_basis(
            sgw, w1m, w2m, w1b, w2b, np.random.default_rng(s))
        wflat = np.concatenate([np.asarray(x) for x in wt])
        ests.append(float(np.sum(wflat * f)))
        nz.append(int(np.count_nonzero(wflat)))
    assert abs(np.mean(ests) - true) < 0.05 * max(true, 1e-9) + 1e-6, (np.mean(ests), true)
    assert np.mean(nz) <= w1b + w2b + 1  # realized op-EVs bounded by the budget
