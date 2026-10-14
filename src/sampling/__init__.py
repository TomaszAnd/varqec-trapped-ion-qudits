"""Per-step sampling of the detection set for the channel-weighted loss.

`stratified_importance` implements the importance-sampling strategies of
paper Sec. IX A: a Neyman split of the draw budget across wire groups
(App. D) and, inside each group, draws proportional to the operators'
channel weights, each drawn term divided by its draw probability so the
sampled loss is an unbiased estimate of the full one. The functions return
the per-group `weights_per_group` tuple consumed by
`src.jax_backend.create_jax_loss_vmap_weighted` /
`build_varqec_loss_weighted` (same interface as the uniform sampler
`src.jax_backend.stratified_weights`).

Consumed by src/seed_race.py (the "importance" sampler of the multi-start
race) and the training scripts under scripts/ and results/campaigns/.
"""
from src.sampling.stratified_importance import (
    make_stratified_importance_weights,
    make_stratified_importance_weights_full_basis,
    neyman_group_budgets,
)

__all__ = [
    "make_stratified_importance_weights",
    "make_stratified_importance_weights_full_basis",
    "neyman_group_budgets",
]
