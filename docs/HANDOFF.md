# Handoff: code ↔ manuscript discrepancies and conventions

Everything below was found while preparing this repository against the
manuscript dated 30 August 2026 (*Native-gate variational error-correcting
codes for trapped-ion qudits*). Each item states what the code does and
what the manuscript says; nothing was changed on either side to make them
agree. Items that would change a printed number come first; then items that
affect a stated definition or claim; then conventions a reader of the code
should know.

## A. Items that change a printed number

1. **The "full" detection set is smaller than its definition, and the
   Table IV residuals are sums over it.** The wire-grouped builder
   de-duplicates operators by testing whether `M M2†` is proportional to
   the identity (`src/errors.py::factored_to_grouped`,
   `_is_proportional_to_identity`). For the non-unitary lowering/raising
   operators of the extended basis (`basis="full"`: the unweighted protocol
   variants of Sec. VIII A and the "full basis" rows of Table IV) a vanishing
   product passes that test, so pairs such as L_01, L_12 on the same qudit
   are treated as duplicates and one is dropped. At n = 5, q = 3, distance 3
   the completed full set has 1191 operators and the builder keeps 801; an
   exact proportionality test keeps all 1191 (there are no genuine
   duplicates). The "residual" column of Table IV is the detection loss
   summed over that set, so it depends on which set is used:

   | Table IV row | printed (801 operators) | full set (1191 operators) |
   |---|---|---|
   | central, channel-weighted L = 2 | 4.6 | 5.81 |
   | channel basis, full batch, L = 16 | 2.90 | 3.74 |
   | full basis, residual plateau, L = 16 | 0.449 | 0.668 |
   | same run, converged | 0.360 | 0.536 |
   | full basis, deep, L = 28 | 1.95 × 10⁻⁴ | 2.71 × 10⁻⁴ |
   | [[5,1,3]]_{Z_3} | ~10⁻³⁰ | ~10⁻³⁰ |

   The ordering of the rows and every conclusion drawn from them in
   Sec. VII are unchanged (the ratios are 1.26–1.49), and the unweighted
   variants were optimized on, and certified against, the 801-operator set.
   **Unaffected:** the channel error basis is unitary, so 221 of 221
   operators are kept; the channel-weighted protocol, the completed channel
   basis loss (0.9888 for the central code), every Table I cell, every
   i.i.d. LER and probe, and every figure are untouched. The authors must
   decide whether to print the residuals over the set the manuscript
   defines (1191) or to state the reduced set.

## B. Items that affect a stated definition or claim (printed numbers unchanged)1. **B2 versus B3 control-qudit map: Table I is sensitive to the choice.**
   App. B quotes Meth et al.'s published control map (uniform phase 2Φ on
   every non-control level, couplings (0, 2, 2) at q = 3: "B2") and then
   prints the Kraus family with the level-resolved coupling f_j = j
   ((0, 1, 2): "B3", the manuscript's re-derivation). These are different
   channels (see `docs/NOISE_MODEL.md` §9 for the two masks). Every
   reported number uses B3 (`control_qudit_kraus`); B2 is
   `control_qudit_kraus_simplified` (`noise_model="simplified"`) and feeds
   nothing. Re-running the Table I cells with the control map switched to
   B2 (everything else unchanged; `channel_cell` with
   `build_dose_steps(..., noise_model="simplified")`):

   | s | map | central proj / Petz | stabilizer proj / Petz | Petz ratio |
   |---|---|---|---|---|
   | 1.0 | B3 (printed) | 39.62 / 16.22 | 37.16 / 10.43 | 1.56 |
   | 1.0 | B2 | 43.15 / 18.37 | 40.33 / 12.02 | 1.53 |
   | 0.5 | B3 (printed) | 23.17 / 6.03 | 21.05 / 3.08 | 1.96 |
   | 0.5 | B2 | 26.21 / 7.43 | 23.77 / 3.85 | 1.93 |

   Both codes get worse by a similar factor and the ordering and ratios are
   essentially unchanged, but every Table I entry moves. Which control map
   the paper means is the authors' decision; the code runs either.

2. **Spectator-ion dephasing is an extension of the fitted model.** The
   code applies the spectator family (f_k = k, the same σ_p²) to every ion
   outside the driven pair at every gate step (`gate_pairs`,
   `build_dose_steps`). The manuscript presents this register-wide
   dephasing as part of the calibrated model (Sec. I, Sec. III: "dephasing
   every qudit of the register"), with the phase pattern "taken from the
   noise-model appendix of Ref. meth2024". Whether Meth et al. fitted a
   spectator-ion term separately could not be checked here (their appendix
   is not in this repository); the transfer of the single-gate calibration
   to spectator ions and to a single MS instance is stated in
   `docs/NOISE_MODEL.md` §9 as an assumption of the paper. At q = 3 with
   driven levels (0, 1) the control (B3), target and spectator masks
   coincide, so the extension changes nothing numerically within one gate
   step; it matters because every one of the 10 steps dephases all five
   ions.

3. **Noisy-encoding variant (Sec. V E).** The text says the variant spreads
   the same p = 0.05 budget "across every gate of the encoding circuit".
   The code (`results/campaigns/r2_campaign/t6a_noisy.py`; record
   `t6a_results.json`) injects one basis operator per qudit with
   probability p_step = 1 − (1−p)^{1/n_MS} after each **MS gate only**
   (36 for the stabilizer encoder, 40 for the central code), never after a
   single-qudit gate; the printed 1.29× is 0.0427 vs 0.0333 from that
   record.

4. **Weight-enumerator normalisation (App. A).** The manuscript defines the
   projector as P = (1/K) Σ_k |ψ_k⟩⟨ψ_k| and then A_j = K⁻² Σ |Tr(PE)|²,
   B_j = K⁻¹ Σ Tr(PEPE†), which gives A_0 = 1/K², contradicting the table
   caption "A_0 = B_0 = 1 throughout". The code
   (`src/analysis.py`, `scripts/reproduce_tables.py`) uses the unnormalised
   projector P = Σ_k |ψ_k⟩⟨ψ_k| (trace K), the standard Shor–Laflamme
   convention, which gives A_0 = B_0 = 1 and the printed Table II. The
   manuscript's definition of P should drop the 1/K.

5. **Closure example (Sec. IV C).** The text says "at q = 3, Z_1 X_01 =
   −|1⟩⟨0| is not a basis operator". With the operators as defined
   (Z_1 = diag(1, −1, 1), X_01 the swap of levels 0, 1 with identity on
   level 2) the product is |0⟩⟨1| − |1⟩⟨0| + |2⟩⟨2|. The conclusion (the
   product leaves the basis and is added by the completion) stands.

6. **Spectator family (App. B).** The displayed D_n sums "over spectator
   levels"; the code's spectator family (`spectator_qudit_kraus`) couples
   every level of a spectator *qudit* with f_k = k, which is also what the
   Sec. III text describes. Reading "spectators" as spectator qudits
   matches the code.

7. **"One draw per gate" (Sec. III footnote).** The trajectory samplers
   draw one independent phase per qudit per gate step
   (`src/simulation.py::make_meth_mc_noise_fn`), consistent with the
   tensor-product form of Eq. (8); the footnote could be read as one phase
   shared by control and target. The exact engine does not sample at all.

8. **What the channel weights are (Sec. IV C/D).** The manuscript describes
   the weight of a detection operator as "the probability the calibrated
   channel assigns to it in the decomposition of App. B". The implementation
   (`src/meth_weights.py`) uses the Pauli-twirl projection
   (1/q²) Σ_a |Tr(P† K_a)|² of the *spectator* dephasing family for every
   qudit, lifted to products over qudits and normalised over the detection
   set; the angle-fluctuation term adds operators but no weight (as stated).
   Not a contradiction, but the twirl and the spectator-role approximation
   are not described in the manuscript.

9. **CSUB.** `src/gates.py` provides CSUB = CSUM† as a library gate; the
   manuscript's gate library lists five two-qudit gates. CSUB is used only
   in the compiled [[5,1,3]] encoder of App. C (as CZ† conjugated by
   Fourier gates), where it costs the same four MS gates.

10. **Which "importance sampling" optimized the central code.** Sec. IX A
   defines strategy 2 ("Importance") as the weight-1 sector evaluated in
   full and the weight-2 sector sampled, and Sec. VI A says the central
   code was optimized "with importance sampling at f = 0.2". The library
   optimizer that produced it (`src/seed_race.py::run_race`,
   `sampler="importance"`) draws over *all* wire groups, weight-1 and
   weight-2 alike, with one shared budget of f·|E_det| operators per step
   (Neyman per-group allocation, within-group draws ∝ channel weight), i.e.
   it is closest to strategy 3 ("full-basis importance") with a single
   budget. The strategies exactly as defined in Sec. IX A are implemented
   in the sampler-study driver `results/campaigns/r10_sampler/t1_pilot.py`
   behind Figs. 10–11 and Table VI. The central code's parameters are
   committed, so no number changes; the sentence in Sec. VI A describing
   its sampler is the item to check.

## C. Conventions the manuscript leaves implicit
1. **Role assignment in a noise round.** Sec. V B says every qudit dephases
   "at the rate the calibrated channel assigns to its part in that step
   (control, target, or spectator)". The code's convention
   (`src/central_comparison.py::gate_pairs`, and every campaign script):
   in the step of pair (i, j) with i < j, qudit i is the control, qudit j
   the target, the driven transition is (0, 1) in every step, and the MS
   phase of the angle-fluctuation factor is φ = 0. At q = 3 with driven
   levels (0, 1) the control couplings (0, 1, 2), the target couplings
   (1, 2, 3) and the spectator couplings (0, 1, 2) have identical pairwise
   differences, so the three dephasing masks coincide; the roles differ in
   the dephasing part only at q ≥ 4. This is a property of the model, not
   a bug, but the manuscript's wording suggests role-dependent rates at
   every q.

2. **Series truncation.** App. B writes the Kraus families as infinite
   series. The trajectory samplers truncate at n_max = 5 (dephasing) and
   10 (angle fluctuation); the exact engine behind every calibrated-channel
   number uses the analytic Gaussian coherence masks, i.e. the untruncated
   channel. Completeness defect of the truncated target family at the
   calibrated point: 6.4e-5 at n_max = 5, 3.7e-13 at n_max = 12
   (`notebooks/central_comparison.ipynb`, Sec. 4).

3. **Exponent of the dephasing.** Code and manuscript agree:
   coherences decay as e^{−½ σ_p² Δf²} (1.0 / 0.9642 / 0.8642 at η = 0.9296
   for Δf = 0, 1, 2), see `docs/NOISE_MODEL.md` §3. Older project notes and
   an older in-code comment claimed a factor-two disagreement with the
   manuscript; that claim referred to an earlier draft and is obsolete.

4. **Single-qudit gates of the compiled [[5,1,3]] encoder.** App. C counts
   36 MS gates "plus single-qudit gates". The compiled circuit
   (`src/stabilizer_encoder.py`) carries 174 single-qudit unitaries
   (Fourier gates, subspace Hadamards, level permutations, diagonal
   corrections) as explicit 3 × 3 matrices; they are not further decomposed
   into the XY / Z rotations of Sec. II A. The MS count does not depend on
   that decomposition. Codewords are reproduced up to a global phase per
   codeword (residual ≤ 1e-14, as stated).

5. **Logical label preparation.** Sec. IV A prepares |k, 0, …, 0⟩. The
   state-vector encoder (`src/encoder.py`, manual path, used by the catalog
   and by all campaign codes through the JAX backend) does exactly that.
   The PennyLane path for q = 3 reaches |k⟩ by a chain of XY(π) rotations,
   which adds a global phase (−i)^k per codeword; no reported number uses
   that path.

6. **Lookup decoder.** Sec. V A: "applies every candidate correction (the
   identity and each single-qudit inverse E†) and keeps the candidate whose
   output lies closest to the code space"; the code scores each candidate by
   Σ_k |⟨E ψ_k | noisy⟩|² (the squared norm of the corrected state's
   projection onto the code space) and then projects the winner onto the
   code space before comparing with the input
   (`src/simulation.py::simulate_ler_with_correction_factored`). Ties go to
   the first candidate in the order identity, then (qudit, basis operator).

7. **Certified optimal recovery (App. F).** The manuscript says the
   optimum was computed "for the cells of the Sec. VI A comparison" and
   quotes Petz-over-optimal overheads 1.42–1.46 (stabilizer) and 1.28–1.36
   (variational codes). The committed record
   (`results/campaigns/r12_scrutiny/t2_results.json`) covers s ∈ {0.25, 0.5}
   only, for four codes. The notebook recomputes the certified optimum on
   the full strength grid for the two codes of Table I: the stabilizer
   code leads in every cell (ratio of certified bounds 3.28, 2.65, 2.13,
   1.70, 1.47 at s = 0.125 … 1.5) and the overheads are 1.39–1.47
   (stabilizer) and 1.29–1.35 (central code). The committed numbers were
   produced by a certified fixed-point iteration
   (`t2_jrf.py`, ported to `src/central_comparison.py`), not by the generic
   SDP solver script `t2_sdp.py`, which failed at this problem size.

8. **Stabilizer-code weight-3 probe.** The value LER|_{w=3} = 0.729 for
   [[5,1,3]]_{Z_3} (Sec. VI A, Table IV) is not stored in any committed
   record; it is reproduced by the notebook (100 000 conditioned shots,
   seed 1003: 0.7287). The central code's 0.289 is both recorded and
   reproduced (0.28916).

9. **Table I caption.** "The term costs the variational code 37% and the
    stabilizer code 76%": with the unrounded rates the costs are 37.4% and
    76.8%; the manuscript rounds the second down.

10. **Where the certified stopping rule lives.** Sec. IX B's rule (five
    consecutive sampled values below the target, then a full-batch
    confirmation, at least 50 steps between confirmations) is implemented
    in the campaign script that produced Table V
    (`results/campaigns/sampling_budget_v2/b_policy_queue.py`, policy
    `hysteresis`). The library optimiser `src/seed_race.py::run_race`
    implements only the trigger (sampled loss below
    `target·(1 + trigger_margin)` for `trigger_patience` consecutive steps;
    defaults 0 and 1) followed by a full-batch confirmation, with no
    minimum gap between confirmations; its docstring says so. The central
    code and the depth/pruning codes were optimized with
    `trigger_margin = 0.25`, `trigger_patience = 5` and a target of 1e-9
    that was never reached (preset 3000 steps), so the stopping rule did
    not act on them. The library default is `trigger_patience = 1`
    (one sampled value below the margin triggers a confirmation); the
    50-step minimum gap appears in `b_policy_queue.py` and in
    `results/campaigns/r11_gapfill/t7_budget.py`, which produced the
    certified budgets quoted in Sec. IX C.

11. **Default sampling fraction.** `src/seed_race.py` and
    `scripts/seed_race.py` default to f = 0.5 (a knee measured for the
    *unweighted* objective); the paper's protocol recommends f = 0.05 for
    the channel-weighted objective, and the central code used f = 0.2.
    Every campaign script passes its fraction explicitly.

12. **Optimizer schedule.** Sec. IV D names Adam and App. D "staged Adam"
    without stages. `src/seed_race.py::run_race`: Adam at 0.05, lowered to
    0.01 when the sampled loss falls below 0.5 and to 0.001 below 1e-3,
    moments carried across switches; `src/jax_backend.py::train_jax`
    (used by `scripts/train.py`): 0.05 → 0.01 once the loss is below 0.1,
    with the optimizer re-initialised. Every reported code came through
    `run_race`.

13. **"Factored fallback" (App. D).** A factored loss path exists
    (`src/jax_backend.py::create_jax_loss_factored`) but no optimisation
    driver falls back to it; every reported code used the wire-grouped
    path, which compiled for every setting of the paper.

14. **Numerical tolerances of App. F.** The sector construction's "fixed
    numerical tolerance" is an absolute singular-value threshold of 1e-8
    (`src/decoders/projective_decode.py::build_syndrome_sectors`). The
    Petz inverse square root N(P)^{-1/2} is taken on the support of N(P):
    the exact-channel Petz keeps eigenvalues above 1e-12 × max
    (`build_petz_exact_dose`) and scores mass outside the support as
    unrecovered (the reported leak is at machine precision for every cell);
    the shot-based Petz of `src/decoders/petz.py` uses a relative singular-
    value cutoff of 1e-10. Eq. (F2) states neither.

15. **Figure-file names.** The figure files carry the tag of the manuscript
    revision that last changed them (`_w5`, `_w6`, `_w7`, `_w8`); the mapping
    to the paper's figure numbers is in `REPRODUCE.md` and
    `scripts/render_paper_figures.py`.
