"""Smoke + certification tests for scripts/seed_race.py.

Two contracts:
1. A small race (3 seeds x 50 steps, n=3/d=3/dist=2) completes quickly and
   persists an npz with full metadata whether or not a seed certifies.
2. The H1 fix: a theta whose SAMPLED loss is below target while its
   full-batch loss is not must NOT certify. Reuses the |kkk> fixture pattern
   from tests/test_full_vs_sampled_convergence.py .
"""
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

TARGET = 1e-6


def test_smoke_race_produces_npz(tmp_path):
    """3 seeds x 50 steps on (n=3, d=3, dist=2) in < 60 s, npz written."""
    from seed_race import run_race

    out = str(tmp_path / "race_smoke.npz")
    t0 = time.time()
    res = run_race(d=3, n=3, distance=2, layers=2, num_seeds=3, steps=50,
                   target_loss=TARGET, sample_frac=1.0, out=out,
                   verbose=False)
    elapsed = time.time() - t0
    assert elapsed < 60, f"smoke race took {elapsed:.0f}s"

    assert os.path.exists(out)
    data = np.load(out, allow_pickle=True)
    assert data["params"].shape == (2, 42)  # (5*3 + 2*3) * (d-1) = 42
    assert data["losses"].shape[0] <= 50
    for key in ("seed", "steps_to_target", "certified",
                "certified_full_loss", "sample_frac", "noise", "entangler",
                "connectivity", "target_loss"):
        assert key in data, f"missing metadata key {key}"
    assert str(data["noise"]) == "full"
    assert str(data["entangler"]) == "ms"
    assert str(data["connectivity"]) == "all-to-all"
    # 50 steps cannot reach 1e-6 here; the saved result must say so honestly,
    # and the recorded full-batch loss must agree with the certified flag.
    assert bool(data["certified"]) == (
        float(data["certified_full_loss"]) < TARGET)
    assert res["certified"] == bool(data["certified"])


def test_doctored_sampled_victory_is_rejected():
    """A sampled loss below target with full-batch loss above it must not
    certify. Fixture: |000>,|111>,|222> states whose full KL
    loss is far from zero while a 1-op Hoeffding draw reports ~0."""
    import jax.numpy as jnp
    from src.errors import ErrorModel
    from src.jax_backend import (create_jax_loss_vmap_weighted,
                                 hoeffding_weights)
    from seed_race import evaluate_candidate

    N, D, DIST = 3, 3, 3
    model = ErrorModel(d=D, n_qudit=N, distance=DIST, closed=False)
    grouped = model.build_grouped(verbose=False)
    state_loss = create_jax_loss_vmap_weighted(grouped, D, D, N, DIST)
    sizes = [g['matrices'].shape[0] for g in grouped]

    dim = D ** N
    cs = np.zeros((D, dim), complex)
    for k in range(D):
        cs[k, k * (D ** 2) + k * D + k] = 1.0  # |kkk>
    code_states = jnp.asarray(cs)

    ones = tuple(jnp.ones((m,)) for m in sizes)

    # Doctored run: find a seeded subsample whose sampled loss "converges".
    sampled_val = None
    for seed in range(500):
        rng = np.random.default_rng(seed)
        w = hoeffding_weights(sizes, 1, rng)
        v = float(state_loss(code_states, w))
        if v < TARGET:
            sampled_val = v
            break
    assert sampled_val is not None, \
        "fixture broken: expected some subsample to miss all violated ops"

    # The certification gate seed_race uses must reject this candidate.
    # (theta here is the state itself; the gate only needs a callable.)
    def full_loss_fn(states):
        return state_loss(states, ones)

    certified, full_val = evaluate_candidate(full_loss_fn, code_states,
                                             TARGET)
    assert full_val > 1e-2, "fixture broken: full loss should be large"
    assert not certified, \
        "sampled-only victory must be rejected by full-batch certification"


def test_importance_sampler_accounts_and_is_cheaper():
    """importance-sampled racing runs, reports op-EV accounting, and
    at equal steps/seeds costs fewer op-EVs than full-batch racing (the whole
    point of the sampling budget). Unreachable target so both run to budget."""
    from seed_race import run_race

    common = dict(d=3, n=3, distance=2, layers=2, num_seeds=2, steps=15,
                  target_loss=1e-12, out=None, verbose=False)

    fb = run_race(sample_frac=1.0, sampler="uniform", **common)
    imp = run_race(sample_frac=0.2, sampler="importance", **common)

    # accounting present and self-consistent
    for res in (fb, imp):
        assert res["total_ops"] > 0
        assert res["race_op_evs"] > 0
        assert res["op_evs_at_first_certified"] is None  # nothing certifies
    assert fb["total_ops"] == imp["total_ops"]
    # subsampled steps are cheaper per step -> strictly fewer op-EVs overall
    assert imp["race_op_evs"] < fb["race_op_evs"], (
        imp["race_op_evs"], fb["race_op_evs"])


def test_trigger_patience_requires_consecutive_below():
    """with a huge patience and a short budget, no full-batch
    certification is ever armed even at sample_frac<1, so no record forms."""
    from seed_race import run_race

    res = run_race(d=3, n=3, distance=2, layers=2, num_seeds=1, steps=10,
                   target_loss=1e-1,  # loose, but patience never satisfied
                   sample_frac=0.3, sampler="importance",
                   trigger_margin=10.0, trigger_patience=10_000,
                   out=None, verbose=False)
    assert res["record_steps"] is None
    assert not res["certified"]


def test_weighted_build_race_uses_meth_channel_weights():
    """weighted=True feeds the corrected Meth channel weights as the full-batch
    weights (non-uniform, normalised); unweighted keeps ones."""
    import numpy as np
    from src.seed_race import build_race
    _, _, _, _, _, fw_u, _ = build_race(3, 5, 3, 4, weighted=False)
    _, _, _, _, _, fw_w, sgw_w = build_race(3, 5, 3, 4, weighted=True)
    fu = np.concatenate([np.asarray(w) for w in fw_u])
    fw = np.concatenate([np.asarray(w) for w in fw_w])
    assert np.allclose(fu, 1.0), "unweighted full weights must be ones"
    assert abs(fw.sum() - 1.0) < 1e-6, "Meth weights must be normalised to 1"
    assert fw.std() / fw.mean() > 0.3, "Meth weights must be non-uniform"
    # sampler weights == full weights in the weighted arm (same per-op vector)
    sgw = np.concatenate([np.asarray(w) for w in sgw_w])
    assert np.allclose(sgw, fw)


def test_weighted_run_race_completes():
    """A small weighted race runs end-to-end with importance sampling."""
    from src.seed_race import run_race
    res = run_race(d=3, n=5, distance=3, layers=4, num_seeds=1, steps=25,
                   target_loss=1e-9, sample_frac=0.5, sampler="importance",
                   weighted=True, out=None, verbose=False)
    assert res["result"] is not None
    assert res["total_ops"] > 0 and res["race_op_evs"] > 0
    assert res["result"]["full_loss"] >= 0.0


def test_mercy_rule_not_armed_without_record():
    """Audit/04 H2: with no baseline and no certified record, every seed
    must exhaust its full step budget (no seed killed early)."""
    from seed_race import run_race

    steps = 12
    res = run_race(d=3, n=3, distance=2, layers=2, num_seeds=2, steps=steps,
                   target_loss=1e-12,  # unreachable -> no record forms
                   sample_frac=1.0, out=None, verbose=False)
    assert not res["certified"]
    assert res["record_steps"] is None
    # best_uncert stores len(seed_losses) == full budget for its seed
    assert res["result"]["steps"] == steps


def test_monitor_and_anneal_denoising(tmp_path):
    """De-noising options: monitor trace saved, selection by monitored
    full-batch minimum, anneal metadata recorded; defaults leave npz unchanged."""
    import numpy as np
    from src.seed_race import run_race

    out = tmp_path / "mon.npz"
    res = run_race(d=3, n=3, distance=2, layers=2, num_seeds=1, steps=40,
                   target_loss=1e-12, sample_frac=0.5, sampler="importance",
                   monitor_every=10, anneal_steps=15, out=str(out),
                   verbose=False)
    f = np.load(out)
    ms, ml = f["monitor_steps"], f["monitor_losses"]
    assert len(ms) == len(ml) and len(ms) >= 4
    assert ms[-1] == 39  # final step always monitored
    # saved full_loss is never worse than the best monitored value
    assert res["result"]["full_loss"] <= ml.min() + 1e-15
    assert str(f["selected_by"]) in ("monitor", "sampled_argmin")
    assert int(f["monitor_every"]) == 10 and int(f["anneal_steps"]) == 15
    assert float(f["final_lr"]) == 0.05 / 500  # default lr/500

    # Defaults off: legacy npz shape (empty monitor arrays, sentinel final_lr)
    out2 = tmp_path / "legacy.npz"
    run_race(d=3, n=3, distance=2, layers=2, num_seeds=1, steps=15,
             target_loss=1e-12, sample_frac=1.0, out=str(out2), verbose=False)
    f2 = np.load(out2)
    assert len(f2["monitor_steps"]) == 0
    assert float(f2["final_lr"]) == -1.0
    assert str(f2["selected_by"]) == "sampled_argmin"
