"""Helpers for the central five-qutrit comparison (paper Sec. VI A, Table I).

The central comparison measures the variational ((5,3,3))_3 code (MS 40)
against the analytical [[5,1,3]]_{Z_3} stabilizer code (MS 36) on the
calibrated gate-error channel of Sec. III and on the discrete i.i.d.
certification benchmark of Sec. V.  Everything needed to redo that
comparison from the committed parameters is collected here, built on the
lower-level modules:

* ``src.decoders.exact_channel``      exact density-matrix evolution of one
                                       noise round (Eq. 8, one gate step per
                                       coupled pair; App. B Kraus families);
* ``src.decoders.projective_decode``  the projective syndrome decoder (App. F)
                                       and the Petz recovery (Eq. F2);
* ``src.simulation`` / ``src.errors``  the lookup-decoder i.i.d. benchmark
                                       (Sec. V A) and the weight-conditioned
                                       probes (Sec. V C);
* ``src.catalog``                      the codewords of both codes.

The certified optimal recovery of App. F (fixed-point iteration with
rigorous two-sided bounds on the channel-fidelity SDP) is ported here from
the campaign script that produced the committed numbers
(``results/campaigns/r12_scrutiny/t2_jrf.py``).

Conventions (paper Sec. III and App. F)
---------------------------------------
* Calibrated constants: sigma_p^2 = 0.073 (eta = e^{-sigma_p^2} = 0.9296),
  sigma_a^2 = 0.047.  The strength grid scales both: eta(s) = e^{-s sigma_p^2},
  sigma_a^2(s) = s * 0.047; s = 1 is the calibrated point.
* One noise round = one gate step per coupled pair of the all-to-all graph,
  C(n, 2) steps (10 at n = 5).  In each step the lower-indexed qudit of the
  pair takes the control role, the higher-indexed one the target role, every
  other qudit is a spectator; the driven levels are (0, 1) and the MS phase
  of the angle-fluctuation term is phi = 0.  These are stated conventions of
  the benchmark (the paper applies the per-step channel "at the rate the
  calibrated channel assigns to its part in that step").
* Figure of merit: channel infidelity 1 - F_avg of recovery composed with
  the noise round on the logical space, with F_avg = (K F_e + 1)/(K + 1).

Consumers: ``notebooks/central_comparison.ipynb``,
``tests/test_central_comparison.py``.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ── calibrated constants (paper Sec. III) ────────────────────────────────

#: Fitted stochastic-phase variance of Meth et al.; eta = e^{-sigma_p^2}.
SIGMA_P2 = 0.073
#: Fitted entangling-gate angle-fluctuation variance.
SIGMA_A2 = 0.047
#: eta at the calibrated point.
ETA_CAL = math.exp(-SIGMA_P2)
#: Strength grid of Table I (s = 1 calibrated).
S_GRID: Tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 1.5)
#: Benchmark injection probability of the i.i.d. certification benchmark.
P_BENCH = 0.05
#: Dimension and register size of the central comparison.
D, N = 3, 5


def eta_of_s(s: float) -> float:
    """eta(s) = e^{-s sigma_p^2} (paper Sec. III, strength grid)."""
    return math.exp(-s * SIGMA_P2)


def sigma_a2_of_s(s: float) -> float:
    """sigma_a^2(s) = s * 0.047 (paper Sec. III, strength grid)."""
    return s * SIGMA_A2


def coherence_multipliers(sigma_p2: float, delta_f: Sequence[int]) -> np.ndarray:
    """Gaussian coherence decay e^{-sigma_p^2 Delta_f^2 / 2} for each Delta_f.

    This is the factor by which the Gaussian average of the stochastic-phase
    model damps the coherence between two levels whose couplings differ by
    Delta_f (paper Sec. III and App. B).  At eta = 0.9296 and
    Delta_f = 0, 1, 2 it equals 1.0 / 0.9642 / 0.8642.
    """
    df = np.asarray(delta_f, dtype=float)
    return np.exp(-0.5 * sigma_p2 * df ** 2)


# ── geometry of one noise round ──────────────────────────────────────────

def all_to_all_edges(n: int = N) -> List[List[int]]:
    """Edges [i, j], i < j, of the all-to-all connectivity graph."""
    return [[i, j] for i in range(n) for j in range(i + 1, n)]


def gate_pairs(n: int = N, ctrl_level: int = 0, tgt_level: int = 1
               ) -> List[Tuple[int, int, int, int]]:
    """(control qudit, target qudit, control level, target level) per edge.

    One entry per coupled pair; the lower-indexed qudit is the control.
    Levels (0, 1) are the driven transition of every step (stated
    convention of the benchmark).
    """
    return [(i, j, ctrl_level, tgt_level) for i in range(n) for j in range(i + 1, n)]


def noise_round_steps(n: int = N, d: int = D, s: float = 1.0,
                      with_angle_fluctuation: bool = True,
                      mode: str = "exact") -> List[dict]:
    """Precomputed gate steps of one noise round at strength s (Eq. 8).

    Returns the step list consumed by ``src.decoders.exact_channel``:
    one step per coupled pair, each carrying the per-qudit control/target/
    spectator coherence masks at eta(s) and, unless
    ``with_angle_fluctuation`` is False (the sigma_a^2 = 0 check of the
    Table I caption), the MS angle-fluctuation mask at sigma_a^2(s).
    """
    from src.decoders.exact_channel import build_dose_steps
    sa2 = sigma_a2_of_s(s) if with_angle_fluctuation else 0.0
    return build_dose_steps(n, d, gate_pairs(n), eta_of_s(s), sigma_a2=sa2,
                            mode=mode)


# ── the two codes ────────────────────────────────────────────────────────

def central_code_states() -> np.ndarray:
    """Codewords (3, 243) of the central ((5,3,3))_3 code via ``src.catalog``."""
    from src.catalog import load_code
    return np.asarray(load_code("central_553_3")["code_states"])


def stabilizer_code_states() -> np.ndarray:
    """Codewords (3, 243) of [[5,1,3]]_{Z_3} from paper Eq. (13)."""
    from src.catalog import five_qudit_code_states
    return np.asarray(five_qudit_code_states(D))


# ── calibrated-channel cell: projective decoder, Petz, optimal ───────────

def favg_rate(F_e: float, K: int) -> float:
    """Channel infidelity 1 - F_avg from the entanglement fidelity F_e."""
    return 1.0 - (K * F_e + 1.0) / (K + 1.0)


def _fidelity_sdp_objective(blocks: dict, K: int, dim: int) -> np.ndarray:
    """A = (1/K^2) sum_ij M_ij^T (x) |j><i| with M_ij = N(|psi_i><psi_j|).

    Tr[A X] over the Choi matrix X of a recovery R: B(H) -> B(C^K) is the
    entanglement fidelity of R composed with the noise round (channel-
    fidelity SDP of Fletcher, Shor and Win; paper App. F).
    """
    A = np.zeros((dim * K, dim * K), dtype=complex)
    for i in range(K):
        for j in range(K):
            M = blocks[(i, j)] if i <= j else blocks[(j, i)].conj().T
            B = np.zeros((K, K))
            B[j, i] = 1.0
            A += np.kron(M.T, B)
    A /= K ** 2
    return 0.5 * (A + A.conj().T)


def certified_optimal_fidelity(A: np.ndarray, dim: int, K: int,
                               max_iter: int = 3000, gap_target: float = 2e-6,
                               check_every: int = 10) -> dict:
    """Optimal entanglement fidelity max Tr[A X] over CPTP Choi matrices X
    (X >= 0, Tr_out X = I_dim) with certified two-sided bounds.

    Fixed-point iteration for linear objectives over CPTP Choi matrices
    (Reimpell-Werner / Jezek-Rehacek-Fiurasek family):

        W = A' X A',  Lam = Tr_out W,  X <- (Lam^{-1/2} (x) I_K) W (Lam^{-1/2} (x) I_K),

    with A' = A + c I >= 0, c = max(0, -lambda_min(A)).  Since Tr X = dim is
    fixed by trace preservation, the shift changes the objective by the
    constant dim * c and not the maximiser.  Every iterate is exactly trace
    preserving and positive, so lb = Tr[A X] is a rigorous lower bound on the
    optimum at every step; the dual candidate Y = Lam^{1/2} - c I, shifted to
    dual feasibility (Y (x) I_K >= A), gives a rigorous upper bound Tr[Y].
    Iteration stops when ub - lb < ``gap_target``.

    Returns dict with 'F_e_lb', 'F_e_ub', 'iters', 'wall_s'.
    """
    nbig = dim * K

    def tr_out(X):
        return np.einsum("akbk->ab", X.reshape(dim, K, dim, K))

    def kron_IK_mul(Y, X):
        # (Y (x) I_K) @ X without forming the Kronecker product
        return (Y @ X.reshape(dim, K * nbig)).reshape(nbig, nbig)

    lamA = float(np.linalg.eigvalsh(A)[0])
    c = max(0.0, -lamA) + 1e-12
    Ash = A + c * np.eye(nbig)
    X = np.eye(nbig, dtype=complex) / K
    best_lb, best_ub = -np.inf, np.inf
    t0 = time.time()
    it_done = 0
    for it in range(1, max_iter + 1):
        W = Ash @ X @ Ash
        Lam = tr_out(W)
        w, V = np.linalg.eigh(Lam)
        w = np.maximum(w, 1e-300)
        Lm = (V * (1.0 / np.sqrt(w))[None, :]) @ V.conj().T
        X = kron_IK_mul(Lm, kron_IK_mul(Lm, W.conj().T).conj().T)
        X = 0.5 * (X + X.conj().T)
        it_done = it
        if it % check_every == 0 or it == max_iter:
            lb = float(np.real(np.trace(A @ X)))
            Yc = (V * np.sqrt(w)[None, :]) @ V.conj().T - c * np.eye(dim)
            Smat = kron_IK_mul(Yc, np.eye(nbig, dtype=complex)) - A
            lam = float(np.linalg.eigvalsh(Smat)[0])
            ub = float(np.real(np.trace(Yc))) + dim * max(0.0, -lam)
            best_lb = max(best_lb, lb)
            best_ub = min(best_ub, ub)
            if best_ub - best_lb < gap_target:
                break
    return {'F_e_lb': best_lb, 'F_e_ub': best_ub, 'iters': it_done,
            'wall_s': time.time() - t0}


def channel_cell(code_states: np.ndarray, s: float, d: int = D, n: int = N,
                 with_angle_fluctuation: bool = True,
                 optimal: bool = False, steps: Optional[List[dict]] = None
                 ) -> dict:
    """One cell of Table I: exact infidelities after one noise round.

    Args:
        code_states: (K, d^n) orthonormal codewords.
        s: noise strength on the grid of Sec. III.
        with_angle_fluctuation: False reproduces the sigma_a^2 = 0 check
            of the Table I caption.
        optimal: also compute the certified optimal recovery (App. F).
        steps: precomputed ``noise_round_steps`` (optional, for reuse).

    Returns dict with 'projective_rate', 'petz_rate', the sector report of
    the projective decoder ('n_sectors', 'sector_dims', 'rest_dim'), and,
    if requested, 'optimal_rate_lb' / 'optimal_rate_ub' with the certified
    F_e bounds.  Rates are 1 - F_avg.
    """
    from src.decoders.exact_channel import evolve_code_blocks
    from src.decoders.projective_decode import (
        build_petz_exact_dose, build_projective_instrument,
        entanglement_fidelity_petz, entanglement_fidelity_projective)
    K, dim = code_states.shape
    eta = eta_of_s(s)
    sa2 = sigma_a2_of_s(s) if with_angle_fluctuation else 0.0
    if steps is None:
        steps = noise_round_steps(n, d, s, with_angle_fluctuation)
    blocks = evolve_code_blocks(code_states, steps, d, n)
    inst = build_projective_instrument(code_states, d, n, eta, sigma_a2=sa2)
    proj = entanglement_fidelity_projective(inst, blocks)
    petz = build_petz_exact_dose(code_states, steps, d, n)
    pz = entanglement_fidelity_petz(petz, code_states, blocks)
    rep = inst['report']
    dims = rep['sector_dims']
    out = {
        's': s, 'eta': eta, 'sigma_a2': sa2,
        'projective_rate': proj['rate_1mFavg'],
        'petz_rate': pz['rate_1mFavg'],
        'projective_F_e': proj['F_e'], 'petz_F_e': pz['F_e'],
        'n_sectors': rep['n_sectors'],
        'sector_dims': dims,
        'sector_dim_counts': {int(m): int(sum(1 for x in dims if x == m))
                              for m in sorted(set(dims))},
        'claimed_dim': rep['claimed_dim_R'], 'rest_dim': rep['rest_dim'],
        'petz_support_rank': pz['support_rank'],
    }
    if optimal:
        A = _fidelity_sdp_objective(blocks, K, dim)
        opt = certified_optimal_fidelity(A, dim, K)
        out.update({
            'F_e_opt_lb': opt['F_e_lb'], 'F_e_opt_ub': opt['F_e_ub'],
            'optimal_rate_lb': favg_rate(opt['F_e_ub'], K),
            'optimal_rate_ub': favg_rate(opt['F_e_lb'], K),
            'optimal_iters': opt['iters'], 'optimal_wall_s': opt['wall_s'],
            'certified_opt_ge_petz': bool(opt['F_e_ub'] >= pz['F_e'] - 1e-9),
            'certified_opt_ge_projective': bool(opt['F_e_ub'] >= proj['F_e'] - 1e-9),
        })
    return out


# ── i.i.d. certification benchmark and weight-conditioned probes ─────────

def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95% binomial interval for k successes in n trials."""
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    dn = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / dn
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / dn
    return (max(0.0, c - h), c + h)


def weight_conditioned_noise_fn(d: int, n_qudit: int, w: int):
    """Noise function with exactly w erring qudits per shot (Sec. V C probe).

    Draws w distinct qudits uniformly and applies to each one operator drawn
    uniformly from the channel error basis of Sec. IV C
    (``src.errors.qudit_hardware_error_basis``).
    """
    from src.errors import qudit_hardware_error_basis
    from src.simulation import _apply_factored_op
    S = [np.asarray(E, complex) for E in qudit_hardware_error_basis(d)]

    def fn(state, rng):
        qs = rng.choice(n_qudit, size=w, replace=False)
        s = state
        for q in qs:
            s = _apply_factored_op(s, S[rng.integers(len(S))], int(q), n_qudit, d)
        nrm = np.linalg.norm(s)
        return s / nrm if nrm > 1e-15 else s
    return fn


def _ler_summary(result: dict, n_shots: int) -> dict:
    ler = float(result['logical_error_rate'])
    k = int(round(ler * n_shots))
    lo, hi = wilson_interval(k, n_shots)
    return {'ler': ler, 'k': k, 'n_shots': n_shots, 'wilson95': (lo, hi),
            'mean_fidelity': float(result['mean_fidelity'])}


def iid_logical_error_rate(code_states: np.ndarray, p: float = P_BENCH,
                           n_shots: int = 200_000, seed: int = 3000,
                           d: int = D, n: int = N) -> dict:
    """Lookup-decoder LER on the i.i.d. benchmark (Sec. V A) with Wilson CI.

    Each qudit independently suffers, with probability p, one operator
    drawn uniformly from the channel error basis; the decoder is the lookup
    table over {I} + weight-1 corrections (``src.simulation``).
    """
    from src.errors import make_hardware_noise_fn, qudit_hardware_error_basis
    from src.simulation import simulate_ler_with_correction_factored
    S = [np.asarray(E, complex) for E in qudit_hardware_error_basis(d)]
    r = simulate_ler_with_correction_factored(
        code_states, make_hardware_noise_fn(d, n, p), S, n, d,
        n_shots=n_shots, seed=seed)
    out = _ler_summary(r, n_shots)
    out['p'] = p
    return out


def weight_conditioned_probe(code_states: np.ndarray, w: int,
                             n_shots: int = 100_000, seed: int = 1000,
                             d: int = D, n: int = N) -> dict:
    """LER|_{w}: lookup-decoder failure rate with exactly w erring qudits."""
    from src.errors import qudit_hardware_error_basis
    from src.simulation import simulate_ler_with_correction_factored
    S = [np.asarray(E, complex) for E in qudit_hardware_error_basis(d)]
    r = simulate_ler_with_correction_factored(
        code_states, weight_conditioned_noise_fn(d, n, w), S, n, d,
        n_shots=n_shots, seed=seed)
    out = _ler_summary(r, n_shots)
    out['w'] = w
    return out


# ── encoder resource counts (paper App. D, Table III) ────────────────────

def ansatz_resource_counts(n: int, q: int, L: int = 2,
                           n_edges: Optional[int] = None) -> Dict[str, int]:
    """Per-code gate and parameter counts of the ansatz of Sec. IV A.

    Per layer: two rounds of XY rotations (2 (q-1) gates, 4 (q-1) parameters
    per qudit), one MS gate per (edge, transition) (2 (q-1) parameters per
    edge), one Z phase per (qudit, transition) ((q-1) parameters per qudit):
    P_l = (5 n + 2 |E|)(q-1), params = L P_l, MS budget = L |E| (q-1).
    """
    E = math.comb(n, 2) if n_edges is None else n_edges
    return {
        'n': n, 'q': q, 'L': L, 'E': E,
        'P_layer': (5 * n + 2 * E) * (q - 1),
        'params': L * (5 * n + 2 * E) * (q - 1),
        'XY_per_layer': 2 * n * (q - 1),
        'MS_per_layer': E * (q - 1),
        'Z_per_layer': n * (q - 1),
        'MS_budget': L * E * (q - 1),
    }
