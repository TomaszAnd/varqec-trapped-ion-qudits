"""Channel-prior builders for the weighted MAP and Petz decoders.

A *prior* is a flat per-operator weight vector aligned with the weight-<=2
closure basis ``ErrorModel(d, n_qudit, distance=3, closed=True).build_grouped()``
(paper Sec. IV C, "Channel error basis and completion"), in the flat
ordering of ``src.decoders._common.flat_group_offsets``.  Two priors are
provided:

* ``meth_pauli_prior``          the Pauli-twirled calibrated-channel weights
                                 of the n = 9 qutrit register (685 operators),
                                 read from ``results/campaigns/n9_reference/
                                 meth_pauli_weights.npz`` (written by
                                 ``scripts/meth_pauli_weights.py``);
* ``uniform_depolarizing_prior`` the analytic prior of i.i.d. per-qudit
                                 depolarizing noise at strength p (paper
                                 Sec. V A benchmark), used by the decoder
                                 tests as a reference prior.

Priors enter the decoders only through their ordering and relative size:
``weighted_map`` scores corrections by ``w_C * ||P_C C noisy||^2`` and the
projective decoder of ``projective_decode`` orders corrections by descending
prior.  (The projective decoder of the paper's App. F uses the live
calibrated-channel prior from ``src.meth_weights`` instead of this module;
see ``projective_decode.build_projective_instrument``.)

Consumers: ``tests/test_petz.py``, ``tests/test_weighted_map.py``,
``tests/test_benchmark_ler_meth.py``, and campaign scripts under
``results/campaigns/r1_campaign``.
"""
from __future__ import annotations
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def meth_pauli_prior(weights_npz_path: str = None) -> np.ndarray:
    """Load the Pauli-twirled calibrated-channel per-operator weights.

    Args:
      weights_npz_path: path to an ``.npz`` with a ``'weights'`` field; the
        default is the committed n = 9, q = 3 file used to train the
        channel-weighted codes and to drive the MAP-LER sweeps.

    Returns:
      Flat (685,) float vector, normalized to sum 1, in the flat order of the
      weight-<=2 closure basis at d = 3, n_qudit = 9.
    """
    if weights_npz_path is None:
        weights_npz_path = os.path.join(
            REPO, "results/campaigns/n9_reference/meth_pauli_weights.npz")
    return np.asarray(np.load(weights_npz_path, allow_pickle=True)['weights'])


def uniform_depolarizing_prior(E_full_grouped: list, n_qudit: int, d: int,
                                p: float, n_single: int) -> np.ndarray:
    """Flat per-op priors for uniform per-qudit Pauli depolarizing at strength p
    (matches src.simulation.make_pauli_depolarizing_noise_fn).

    Args:
      E_full_grouped: grouped weight-<=2 closure basis from
        ``ErrorModel.build_grouped``.
      n_qudit, d: register size and local dimension.
      p: per-qudit error probability.
      n_single: number of *generating* single-qudit operators (the first
        ``n_single`` matrices of every single-wire group are the operators
        the i.i.d. channel actually draws; the remaining ones are closure
        products the channel never produces).

    Returns:
      (n_total,) float vector with the probability the i.i.d. channel assigns
      to each operator:

      Identity:                      (1-p)^n
      Single-qudit (e < n_single):   (1-p)^(n-1)*p*(1/n_single)
      Single-qudit closure (e >= n_single): 0  (not produced by single-Pauli channel)
      Two-qudit:                     (1-p)^(n-2)*(p/n_single)^2
      Weight->=3 mass omitted (outside the weight-<=2 basis), so the vector
      does not sum to 1.
    """
    n_total = sum(int(g['matrices'].shape[0]) for g in E_full_grouped)
    w = np.zeros(n_total)
    flat = 0
    for g in E_full_grouped:
        wires = g['wires']
        n_g = int(g['matrices'].shape[0])
        if len(wires) == 0:
            w[flat] = (1 - p) ** n_qudit
        elif len(wires) == 1:
            for e in range(min(n_single, n_g)):
                w[flat + e] = (1 - p) ** (n_qudit - 1) * p * (1.0 / n_single)
        else:
            for e in range(n_g):
                w[flat + e] = (1 - p) ** (n_qudit - 2) * (p / n_single) ** 2
        flat += n_g
    return w
