"""Paper-campaign reproduction regression .

Guards the last gap left by the train.py default flips (--noise full,
--connectivity all-to-all): reproducing the paper's campaign codes requires
`--noise dephasing --connectivity ring` explicitly. This test invokes the
same library path train.py takes for those flags (ErrorModel with
basis=None, ring connections, JAX backend) on the the recorded smoke config
(d=3, n=3, dist=2, 2 layers, 30 steps, seed 0, lr=0.05, lr_switch=0.01) and
pins the loss trajectory to the values the pre-flag script produced.

the recorded recorded final=4.8718e-02 for this config, verified identical to
the pre-flag script checked out from git; the full-precision references
below were regenerated 2026-07-04 from the current library path, whose
final matches the the recorded record to all recorded digits.
"""
import pytest

# step index -> loss (pre-flag paper-campaign trajectory, seed 0)
REFERENCE_TRAJECTORY = {
    0: 1.095034373608423e+00,
    1: 6.213076376244369e-01,
    5: 3.723588077676094e-01,
    10: 1.938552975709998e-01,
    20: 8.871593849170170e-02,
    29: 4.871769439316705e-02,
}
FINAL_LOSS = 4.871769439316705e-02
N_STEPS = 30


def _run_paper_smoke_config():
    from src.errors import ErrorModel
    from src.jax_backend import create_jax_encoder, build_varqec_loss, \
        train_jax

    n, d, dist = 3, 3, 2
    # --connectivity ring: [[i, (i+1) % n] for i in range(n)] (train.py:69)
    connections = [[0, 1], [1, 2], [2, 0]]
    # --noise dephasing -> basis=None; dist=2 -> closed=False (train.py:62,80)
    model = ErrorModel(d=d, n_qudit=n, distance=dist, closed=False,
                       basis=None)
    E_det_grouped = model.build_grouped(verbose=False)
    # 2 layers -> use_scan=False (train.py:186)
    enc, _, ppl = create_jax_encoder(n, d, connections=connections,
                                     use_scan=False)
    loss_fn = build_varqec_loss(encoder_fn=enc, K=d, d=d, n_qudit=n,
                                distance=dist, E_det_grouped=E_det_grouped)
    params, losses = train_jax(loss_fn, 2, ppl, n_steps=N_STEPS,
                               lr=0.05, lr_switch=0.01, seed=0)
    return losses


@pytest.fixture(scope="module")
def losses():
    return _run_paper_smoke_config()


def test_paper_campaign_smoke_trajectory_pinned(losses):
    """--noise dephasing --connectivity ring equivalents reproduce the
    pre-flag trajectory bit-for-bit (to 1e-10)."""
    assert len(losses) == N_STEPS
    assert abs(losses[-1] - FINAL_LOSS) < 1e-10, (
        f"final loss {losses[-1]:.15e} != pre-flag reference "
        f"{FINAL_LOSS:.15e} — the paper-campaign reproduction path has "
        f"drifted (the recorded recorded final=4.8718e-02)")
    for step, ref in REFERENCE_TRAJECTORY.items():
        assert abs(losses[step] - ref) < 1e-10, (
            f"loss at step {step}: {losses[step]:.15e} != {ref:.15e}")


def test_paper_campaign_loss_decreases(losses):
    """Sanity on the same run: trajectory is overall decreasing."""
    assert losses[-1] < losses[0] / 10
