# Native-gate variational error-correcting codes for trapped-ion qudits

Code, optimized codes and result records behind the paper *Native-gate
variational error-correcting codes for trapped-ion qudits* (T. Andrzejewski,
U. Holzer, N. K. H. Li, N. Friis).

*We specialize the variational error-correcting-code framework of Cao et al.
to trapped-ion qudits: encoding circuits built from the native gate set,
optimized against the calibrated gate-error channel of the processor
(correlated spectator dephasing composed with entangling-gate angle
fluctuation), and benchmarked at qudit dimensions q ∈ {3, 4, 5}, register
sizes n = 5–9 and target distances 2–4. On the discrete i.i.d. benchmark the
variational ((5,3,3))₃ code reaches a logical error rate 2.48 times below the
[[5,1,3]]_{Z₃} stabilizer code at nearly the same encoding cost; on the
calibrated channel the ordering reverses, and the stabilizer code leads at
every noise strength under near-optimal and exact optimal recovery.*

## Install

```bash
python3 -m venv env && source env/bin/activate      # Python 3.12
pip install -r requirements.txt                     # jax 0.4.35, pennylane 0.42.0, optax 0.2.4, ... (all pinned)
export PYTHONPATH=$(pwd)                            # so `import src...` resolves
python3 -m pytest tests/ -q                         # ~17 min on a laptop; 432 tests
```

Everything runs on CPU. Nothing in this repository retrains a code: every
number is regenerated from the committed parameter files and result records
under `results/`.

## Start here: `notebooks/central_comparison.ipynb`

The notebook recomputes the central comparison of the paper (Sec. VI A,
Table I; App. B, C, F) from `src/` and the committed parameters, in seven
linear steps: the two codes and their Knill–Laflamme residuals and weight
enumerators; the compilation of both encoders (the native-gate ansatz, and
the 36-MS encoder of [[5,1,3]]_{Z₃} with one CZ → 4 × MS decomposition
written out); one calibrated gate step of the noise model; the recovery
calculation (projective syndrome decoder, Petz recovery, certified optimal
recovery) at every strength of Table I; the i.i.d. certification benchmark
and the weight-conditioned probes; and a summary table of paper value vs
notebook value. It executes in about six minutes
(`jupyter nbconvert --to notebook --execute --inplace notebooks/central_comparison.ipynb`
from `notebooks/`); the committed copy carries its outputs.

## What reproduces

[`REPRODUCE.md`](REPRODUCE.md) maps every data figure (Figs. 3–12) and table
(I–VI) of the paper to the command that regenerates it, the parameter files
and records it reads, the printed value and the regenerated value. In short:

```bash
PYTHONPATH=. python3 scripts/render_paper_figures.py          # Figs. 3-12 -> results/figures/
PYTHONPATH=. python3 scripts/print_paper_tables.py --recompute  # Tables I-VI next to the printed values
```

[`docs/PROVENANCE.md`](docs/PROVENANCE.md) lists the record behind each
figure and table, the parameter file behind each code, and the shot budgets
and seeds. [`docs/NOISE_MODEL.md`](docs/NOISE_MODEL.md) is the element-by-element
conformance note of the noise model against the paper.
[`docs/HANDOFF.md`](docs/HANDOFF.md) lists every code ↔ manuscript
discrepancy and implicit convention found while preparing this package.

## Repository layout

```
src/                 library (gates, ansatz, error basis, loss, noise model, decoders, benchmarks)
notebooks/           central_comparison.ipynb (start here)
scripts/             figure and table generators, optimisation and benchmark drivers
  paper_figures/     per-figure render functions (one module per manuscript revision pass)
tests/               pytest suite guarding every module (numerics pinned to the paper's values)
results/
  campaigns/         committed parameter files (.npz) and result records (.json) behind the paper,
                     one directory per campaign, each with the script that produced it
  figures/           regenerated figures (not committed)
docs/                NOISE_MODEL.md, PROVENANCE.md, HANDOFF.md
```

## Module reference (`src/`)

- **`gates.py`** — native gate library for any q: `XY_gate`, `Z_gate` (Sec. II A 1),
  the Mølmer–Sørensen `MS_gate` (Sec. II A 2; Ringbauer et al. Eq. 2, verified to 2e-15),
  `zz_subspace_gate` (Eq. 4), `ls_global_gate` (Eq. 5), `CSUM_gate`/`CSUB_gate`, `CEX_gate`.
- **`encoder.py`** — the ansatz of Sec. IV A / Fig. 2 as a state-vector encoder
  (`create_native_encoder`) and its gate sequence (`layer_gate_sequence`).
- **`jax_backend.py`** — JIT-compiled encoder, the detection loss of Eq. (12) in
  wire-grouped form (App. D), its channel-weighted and sampled variants, Adam.
- **`loss.py`** — NumPy/PennyLane reference implementations of the loss; parameter-file I/O.
- **`errors.py`** — channel error basis `{Z_k, X_{k-1,k}}` (Sec. IV C), the completed
  detection set (`ErrorModel`, `close_error_basis`), wire grouping, and the i.i.d. noise
  function of the certification benchmark.
- **`correlated_noise.py`** — the Kraus families of App. B (control/target/spectator
  dephasing, MS angle fluctuation), amplitude damping.
- **`meth_weights.py`** — channel probabilities of the detection operators (the weights of
  the channel-weighted objective, Sec. IV D; also the decoder prior).
- **`simulation.py`** — Monte-Carlo logical error rate with the lookup decoder (Sec. V A),
  trajectory samplers of the calibrated channel.
- **`decoders/exact_channel.py`** — exact density-matrix evolution of one noise round
  (Sec. V B; n ≤ 7 at q = 3).
- **`decoders/projective_decode.py`** — the projective syndrome decoder of App. F, the Petz
  recovery of Eq. (F2), exact channel infidelities, the sampled estimator for n ≥ 8.
- **`decoders/petz.py`**, **`decoders/weighted_map.py`**, **`decoders/priors.py`** — shot-based Petz and weight-≤2 MAP decoders with a channel prior
  (the MAP decoder of the Sec. V A footnote).
- **`sampling/stratified_importance.py`** — importance sampling of the detection set
  (Sec. IX A): Neyman per-group budgets, within-group draws ∝ channel weight, unbiased weights.
- **`seed_race.py`** — the multi-run optimisation with the certified stopping rule (Sec. IX B).
- **`catalog.py`** — code loader; `central_553_3` is the central code, `five_qudit_d3` the
  analytical [[5,1,3]]_{Z₃} of Eq. (13).
- **`central_comparison.py`** — constants of Sec. III, one noise round, Table I cells
  (projective / Petz / certified optimal), i.i.d. benchmark and probes, Table III counts.
- **`stabilizer_encoder.py`** — the 36-MS encoding circuit of App. C.
- **`analysis.py`** — Knill–Laflamme residuals, distance check, Shor–Laflamme enumerators (App. A).

## Noise model (Sec. III, App. B)

One entangling-gate step is `N_gate = A_{σ_a²} ∘ (D^(c) ⊗ D^(t) ⊗ ⊗_sp D^(sp))`: Gaussian-
averaged stochastic-phase dephasing on every qudit of the register (coupling f(j) per level:
control 0 on the control level and j elsewhere; target 1 / 2 / 3; spectator j), composed with
the Gaussian-averaged MS angle fluctuation on the driven pair. Coherences between levels with
coupling difference Δf decay as `e^{−½ σ_p² Δf²}` (1.0 / 0.9642 / 0.8642 at the calibrated
η = 0.9296). Calibrated constants σ_p² = 0.073 and σ_a² = 0.047; the strength grid scales
both by s. One noise round is one gate step per coupled pair (10 at n = 5). Exact
calibrated-channel numbers come from the density-matrix engine; the i.i.d. benchmark injects
one operator of the discrete basis per erring qudit. Details and checks: `docs/NOISE_MODEL.md`.

## Gate set

MS is the entangling gate of every reported code (`MS_gate`, Ringbauer Eq. 2, two
parameters per (edge, transition)); the subspace-ZZ, global light-shift, CSUM and CEX gates
enter only the entangling-gate study of Sec. VIII B (`jax_backend.create_jax_encoder_variant`).
The ansatz applies gates on all pairs of the register (all-to-all connectivity) in every
reported code; parameter files store their edge list, which the loaders use.

## Decoders

- **Lookup decoder** (Sec. V A): identity + every single-qudit inverse, closest to the code
  space wins, then projection; used for every i.i.d. LER and for the weight-conditioned probes.
- **Projective syndrome decoder** (App. F): prior-ordered weight-≤2 corrections claim
  orthogonalized syndrome sectors; CPTP; exact at n ≤ 7, sampled estimator at n ≥ 8.
- **Petz recovery** (Eq. F2) of the code and one noise round, evaluated exactly.
- **Certified optimal recovery** (App. F): fixed-point iteration on the channel-fidelity
  SDP with rigorous upper and lower bounds.
- **Weight-≤2 MAP decoder** with channel prior (Sec. V A footnote).

## Notes for developers

- The default optimisation loss is `jax_backend.build_varqec_loss_weighted(..., fast=True)`
  (cross-group batched einsum core); `fast=False` is the per-group vmap core, pinned equal to
  1e-12.
- The detection set is *completed* (`closed=True`) for target distance ≥ 3: the channel error
  basis is not a group, so the same-qudit products the Knill–Laflamme conditions require are
  added explicitly.
- Every reported code uses all-to-all connectivity; `create_native_encoder` and
  `create_jax_encoder` default to a ring only when no edge list is given.
- Import `src.jax_backend` before building any JAX encoder: it enables 64-bit JAX, and
  encoders built in 32-bit mode fail the distance-3 residual checks.
- Exact channel evaluation stores the full density matrix: n ≤ 7 at q = 3, q ≤ 5 at n = 5;
  above that use the sampled estimator of `projective_decode`.
- The lookup decoder runs factored (per-wire contractions) whenever q^n > 500, q ≥ 5 or the
  target distance is ≥ 3, i.e. for every distance-3 benchmark.
- The entangling-gate audit (MS ≡ Ringbauer Eq. 2 to 2e-15, `zz_subspace_gate` ≡ Eq. 4,
  `ls_global_gate` ≡ Eq. 5) is pinned by `tests/test_gates_physical.py`.
- Recommended optimisation protocol (Sec. IX C): channel-weighted objective, importance
  sampling at f = 0.05, L = 2, certified stop at 1.5× the capacity floor, selection by the
  weight-2/weight-3 probes. The unweighted objective diverges under sampling below f ≈ 0.4.
- The open problem of the paper's conclusion — an ancilla circuit realizing the projective
  syndrome measurement — has no code here.

## References

- Cao et al., *Quantum variational learning for quantum error-correcting codes*,
  [arXiv:2204.03560](https://arxiv.org/abs/2204.03560) — the VarQEC framework.
- Meth et al., *Simulating 2D lattice gauge theories on a qudit quantum computer*,
  [arXiv:2310.12110](https://arxiv.org/abs/2310.12110) — the calibrated noise model (App. J).
- Ringbauer et al., Nat. Phys. **18**, 1053 (2022) — the qudit Mølmer–Sørensen gate (Eq. 2).
- Hrmo et al., Nat. Commun. **14**, 2242 (2023) — the global light-shift gate (Eq. 3).
- Chau (1997), Rains (1997) — the [[5,1,3]]_{Z_q} code.
- Petz (1986); Barnum & Knill (2002); Fletcher, Shor & Win (2007) — recovery maps and the
  channel-fidelity SDP.
