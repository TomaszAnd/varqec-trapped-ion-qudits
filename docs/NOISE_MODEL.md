# Noise model: code ↔ paper conformance note

This note states what the code implements for the calibrated gate-error
channel of the paper (Sec. III, App. B) and how each element was checked.
Every check below is reproduced in `notebooks/central_comparison.ipynb`
(Sec. 4) and pinned by `tests/test_central_comparison.py`,
`tests/test_noise_model_paper.py`, `tests/test_channel_v2.py` and
`tests/test_exact_channel.py`.

## 1. The channel (paper Eq. 8)

One entangling-gate step on the coupled pair (control c, target t) is

    N_gate = A_{σ_a²} ∘ ( D^(c) ⊗ D^(t) ⊗ ⊗_{m∉{c,t}} D^(sp) ).

* `D^(c)`, `D^(t)`, `D^(sp)` are Gaussian-averaged stochastic-phase dephasing
  channels on the control, target and spectator qudits
  (`src/correlated_noise.py`: `control_qudit_kraus`, `target_qudit_kraus`,
  `spectator_qudit_kraus`).
* `A_{σ_a²}` is the Gaussian-averaged MS gate-angle fluctuation on the pair
  (`src/correlated_noise.py::ms_jitter_kraus`), applied after the dephasing
  factors, as Eq. (8) is written.

Calibrated constants (Sec. III): `σ_p² = 0.073` (`η = e^{−σ_p²} = 0.9296`),
`σ_a² = 0.047`. The strength grid scales both, `η(s) = e^{−s σ_p²}`,
`σ_a²(s) = s·0.047`; `s = 1` is the calibrated point
(`src/central_comparison.py`).

## 2. Dephasing Kraus families (paper App. B)

For a qudit whose level `k` carries the integer coupling `f_k`, the Kraus
family is, with `σ_p² = −ln η`,

    E_n = Σ_k (f_k σ_p)^n / √(n!) · e^{−σ_p² f_k² / 2} |k⟩⟨k| ,   n = 0, 1, 2, …

(`_kraus_diagonal`). The coupling patterns are

| role      | f_k                                        | code function                  |
|-----------|--------------------------------------------|--------------------------------|
| control   | 0 on the control level, k elsewhere        | `control_qudit_kraus`          |
| target    | 1 (control level), 2 (target level), 3 else | `target_qudit_kraus`           |
| spectator | k                                          | `spectator_qudit_kraus`        |

These are exactly the three families printed in App. B. App. B also quotes
the published control-qudit phase map of Meth et al., `c_i|i⟩ + e^{i2Φ}Σ_{j≠i} c_j|j⟩`
(uniform coupling 2 on every non-control level); the paper states that the
level-resolved `f_j = j` is its own re-derivation and prints the Kraus
family for that. The code's benchmarks use the level-resolved family. The
literal uniform-coupling variant exists as `control_qudit_kraus_simplified`
(`noise_model="simplified"`) and is not used by any reported number.

## 3. The exponent: e^{−½ σ_p² Δf²} — code and paper agree

The coherence between two levels whose couplings differ by `Δf` is damped
by `Σ_n E_n[i] E_n[j]* = e^{−½ σ_p² Δf²}`, the characteristic function of
`Φ ~ N(0, σ_p²)` evaluated at `Δf`. This is what the paper states (Sec. III
and App. B) and what the code implements: at `η = 0.9296` the multipliers
for `Δf = 0, 1, 2` are `1.0 / 0.9642 / 0.8642` (notebook Sec. 4; pinned by
`test_coherence_multipliers_are_half_sigma_squared`). The exact engine uses
the same factor as its analytic coherence mask
(`src/decoders/exact_channel.py::_mask_exact`).

Earlier project notes claimed a factor-two disagreement (`e^{−σ² Δf²}`)
between code and manuscript; that disagreement no longer exists in either
direction. Both the code and the current manuscript carry the Gaussian
`½`.

## 4. Series truncation

The App. B families are infinite series in `n`. The Monte-Carlo trajectory
samplers (`src/simulation.py::make_channel_v2_noise_fn`) truncate at
`n_max = 5` (dephasing) and `n_max = 10` (angle fluctuation); the
completeness defect `‖Σ_n E_n†E_n − I‖` of the truncated target family is
of order `1e-4` at the calibrated point and falls below `1e-10` by
`n_max = 12` (notebook Sec. 4). Every exact calibrated-channel number in
the paper (Table I, the channel panels of Figs. 3–7, 9, 10 and 12, App. F) is
computed by the exact density-matrix engine, which applies the untruncated
Gaussian masks and is exactly trace preserving.

## 5. Angle fluctuation

`U_MS(θ + δθ) U_MS(θ)† = exp[−i (δθ/4) S_φ²]` is θ-independent, so the
Gaussian average is a dephasing channel in the eigenbasis of `S_φ²`, with
eigenvalue tiers `g ∈ {1, 1/4, 0}` (double-active, single-active,
spectator). `ms_jitter_kraus` builds it with the same `_kraus_diagonal`
construction (couplings `g`, variance `σ_a²`) rotated back to the
computational basis. Stated conventions of the memory benchmark: the
driven transition is `(0, 1)` and the MS phase is `φ = 0` (`σ_φ = X_01`)
for every pair; the total error weight of the channel is φ-independent.
Single-qudit-gate angle fluctuation (`σ_a²/10`) is not applied: the memory
benchmark contains no single-qudit gate instances.

## 6. One noise round

One round = one gate step per coupled pair of the all-to-all graph,
`C(n,2)` steps (10 at `n = 5`); in each step the lower-indexed qudit of the
pair is the control, the higher-indexed one the target, all others are
spectators (`src/central_comparison.py::gate_pairs`). Every code of a
comparison receives the identical round, starting from perfectly prepared
codewords.

## 7. The discrete channel error basis (paper Sec. IV C)

The optimisation and the i.i.d. benchmark use the `2(q−1)` unitaries per
qudit `{Z_k}` (phase flip on level `k`, `k = 1..q−1`) and `{X_{k−1,k}}`
(adjacent-level swap), `src/errors.py::qudit_hardware_error_basis`; at
`q = 3`: `{Z_1, Z_2, X_01, X_12}`. The completed detection set at target
distance 3 adds the same-qudit products that leave the basis
(`close_error_basis`). The channel-weighted objective weights each operator
by its probability under the channel (`src/meth_weights.py`): the per-role
Pauli-twirl probabilities of the dephasing families lifted to products, with
the angle-fluctuation term contributing operators but no weights of its own,
as described in Sec. IV C.

## 8. Idle errors

Amplitude damping (`amplitude_damping_kraus`, paper App. B) and memory
dephasing are not applied in any benchmark (Sec. III); the lowering
operators enter only the extended detection basis of the unweighted
protocol variants (`basis="full"` in `src/errors.py`).

## 9. Which gates carry the noise, and where the Kraus families come from

**Which gates are noisy.** None of the gates of the variational encoder are
simulated noisily. Every calibrated-channel number (Table I, the channel
panels of Figs. 3–7, 9, 10 and 12, App. F) is a memory experiment: both codes
start from perfectly prepared codewords, one noise round is applied — one
gate step of Eq. (8) per coupled pair of the all-to-all graph, 10 steps at
n = 5, as if one MS gate had been applied on each pair — and the recovery
map follows (`src/central_comparison.py::noise_round_steps`). The round is
identical for both codes and does not depend on the encoder's angles or on
the compiled 36-MS circuit. Single-qudit gates carry no noise anywhere (the
σ_a²/10 term of Sec. III is never applied). The only gate-by-gate noisy
circuit in the paper is the noisy-encoding variant of Sec. V E, which uses
the *discrete i.i.d.* error (one basis operator with probability p after
each MS gate (p_step = 1 − (1−p)^{1/n_MS}, so the total budget is p;
results/campaigns/r2_campaign/t6a_noisy.py)), not the calibrated channel. During optimisation the channel
enters only through the operator weights of the loss (`src/meth_weights.py`).

**How the Kraus operators are derived.** Each role is a stochastic-phase
channel: level |k⟩ acquires phase f_k Φ with one Φ ~ N(0, σ_p²) per gate
step. Averaging ρ ↦ e^{iFΦ} ρ e^{−iFΦ}, F = diag(f_k), over Φ multiplies
each coherence ρ_jk by E[e^{i(f_j−f_k)Φ}] = e^{−½σ_p²(f_j−f_k)²}. The family
E_n = Σ_k (f_k σ_p)^n/√n! · e^{−σ_p² f_k²/2} |k⟩⟨k| is the Taylor expansion of
that Gaussian: Σ_n E_n[j]E_n[k]* = e^{−σ_p²(f_j²+f_k²)/2} Σ_n (σ_p² f_j f_k)^n/n!
= e^{−½σ_p²(f_j−f_k)²}, with Σ_n E_n†E_n = I. The exact engine skips the
expansion and applies the Gaussian mask directly. The angle-fluctuation
factor is the same construction with δθ ~ N(0, σ_a²) and F replaced by
S_φ²/4, whose eigenvalue tiers {1, 1/4, 0} play the role of f_k.

**Meth et al.'s maps, single MS gates, spectator ions, B2 vs B3.** What the
code does: on each gate step the driven pair receives the control map
(coupling 0 on the control level, k elsewhere) and the target map (1, 2, 3),
every other ion receives the spectator map (f_k = k), all with the fitted
σ_p² = 0.073, and the pair additionally receives the angle fluctuation with
σ_a² = 0.047. The code contains no derivation from a C-ROT decomposition.
The target map and the published control map are taken from the manuscript;
the level-resolved control coupling f_j = j is the manuscript's own
re-derivation (App. B); and the dephasing of ions outside the gate is the
manuscript's own extension of the same model with the same σ_p², which
Meth et al. did not fit separately. That the C-ROT calibration transfers
unchanged to a single MS instance and to spectator ions is therefore an
assumption of the paper (Sec. III, "we adopt their fitted values
unchanged"), not something the code checks.

B2 and B3 are two different models, not one model written twice. At q = 3
with control level 0, B2 (uniform 2Φ on every j ≠ i) gives couplings
(0, 2, 2): the (0,1) coherence decays by e^{−2σ_p²}, the (1,2) coherence not
at all. B3 (f_j = j) gives (0, 1, 2): both decay by e^{−½σ_p²}. Every
reported number uses B3 (`control_qudit_kraus`); B2 is available as
`control_qudit_kraus_simplified` and feeds nothing. Switching the control
map to B2 changes Table I. Under B3 at q = 3 with driven levels (0, 1) the
control (0,1,2), target (1,2,3) and spectator (0,1,2) masks have identical
pairwise differences, so the dephasing part of a gate step is the same on
every qudit; the roles differ only through the angle-fluctuation factor on
the pair (and through the couplings at q ≥ 4). Which control map the paper
means is an authors' decision; the code can run either.