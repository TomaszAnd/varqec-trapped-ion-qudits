"""Decoders and recovery maps for the variational qudit codes.

Modules (paper references in parentheses):

  - ``weighted_map``        prior-weighted MAP decoder over weight-<=2
                            corrections, Monte-Carlo LER (Sec. V A, footnote
                            on the lookup decoder).
  - ``petz``                low-rank Petz transpose recovery built against a
                            weighted-Pauli channel, Monte-Carlo LER
                            (Sec. VII B, the Petz re-examination of the
                            loss--LER decoupling).
  - ``exact_channel``       exact density-matrix evolution through one round
                            of the calibrated gate-error channel (Sec. V B).
  - ``projective_decode``   the projective syndrome decoder of App. F, the
                            exact Petz recovery of App. F Eq. (F2) on the
                            full noise round, and the exact / sampled
                            entanglement-fidelity figures of merit (Sec. V B).
  - ``priors``              channel-prior vectors over the closure basis.
  - ``_common``             wire-grouped operator application, flat-index
                            bookkeeping, checkpoint -> codewords.

The unweighted lookup decoder of the i.i.d. benchmark lives in
``src/simulation.py`` (``lookup_table_decoder``), not here.

``simulate_ler_with_weighted_map`` and ``simulate_ler_with_petz`` return
per-shot Bernoulli outcome arrays (1 = logical error) compatible with the
bootstrap CI helper in ``scripts/benchmark_ler_meth_pauli.py``.
"""
from src.decoders.weighted_map import (
    build_weighted_correction_set,
    simulate_ler_with_weighted_map,
)
from src.decoders.priors import (
    uniform_depolarizing_prior,
    meth_pauli_prior,
)
from src.decoders.petz import (
    build_petz_recovery,
    simulate_ler_with_petz,
    fidelity_under_petz,
)

__all__ = [
    "build_weighted_correction_set",
    "simulate_ler_with_weighted_map",
    "uniform_depolarizing_prior",
    "meth_pauli_prior",
    "build_petz_recovery",
    "simulate_ler_with_petz",
    "fidelity_under_petz",
]
