#!/usr/bin/env python3
"""Pauli-frame MC LER against the twirled Meth channel.

This benchmark draws ONE error per shot from the flat 685-op distribution
`weights[]` stored in `meth_pauli_weights.npz`. That distribution is the
Pauli channel whose ℓ_2 cost upper-bounds the code's ε-correctability via
Cao Prop 3 (with m = 685), and it is exactly the channel-weighted loss objective the n = 9 reference code was
trained against. So pipeline (a) is the matched yardstick for the training
objective.

Default sampler — flat 685-vector
  Per shot, draw idx ~ Categorical(weights/Σweights); apply the lifted
  multi-qudit Pauli operator at that flat index to the encoded state;
  decode with the weight-1 lookup-table corrector (same decoder used in
  R12/R13/R14 LER benchmarks). The α axis rescales the non-identity
  weights by α and renormalizes — α=1 is the calibrated channel, α<1 is
  weaker noise, α>1 is stronger.

Optional `--multi_event` sampler — per-gate per-role product
  Per shot, for each gate (q_a, q_b) in gate_pairs (36 gates for the n=9
  a2a register): sample one Pauli on q_a from `control_weights`, one on
  q_b from `target_weights`, one on every other qudit from
  `spectator_weights`; apply the tensor product. Composes over all 36
  gates per shot. This is NOT the channel the n = 9 reference code was optimized against —
  the per-gate product can produce weight-≥3 ops outside the closure
  basis the decoder is built for, biasing LER. Sensitivity-only; never
  the headline. Quantifies the gap between "Pauli channel matching the
  loss" and "per-gate-independent Pauli channel."

Notes
- α-axis here is NOT the same as the η-axis in pipeline (b). η rescales
  the underlying Kraus channel (a structural change to the noise);
  α rescales the twirled Pauli distribution (a rescaling of the
  post-twirl probabilities). Document both in the head-to-head writeup.
- Phase-only matches the twirl scope of the weight derivation (
  audit §1a/§1e); subspace_depolarizing is not in the weights.

src/ untouched.
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from typing import Callable, Optional

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# Canonical primitives live in the decoder package.
from src.decoders._common import (
    apply_local_np, flat_group_offsets, encoder_forward)


def _apply_single_qudit_np(state: np.ndarray, op: np.ndarray, q: int,
                            n_qudit: int, d: int) -> np.ndarray:
    """Apply d×d op to qudit q. Mirrors src.simulation._apply_factored_op."""
    shape = (d,) * n_qudit
    s = state.reshape(shape)
    s = np.tensordot(op, s, axes=([1], [q]))
    s = np.moveaxis(s, 0, q)
    return s.reshape(-1)


# ─────────── per-shot factored-decoder LER (returns bernoulli outcomes) ───────────

def simulate_per_shot_factored(code_states: np.ndarray,
                                noise_fn: Callable,
                                single_errors: list,
                                n_qudit: int, d: int,
                                n_shots: int,
                                seed: int) -> np.ndarray:
    """Mirror of src.simulation.simulate_ler_with_correction_factored
    that returns the per-shot pass/fail array (1 = logical error, 0 = ok)
    so callers can bootstrap on the bernoulli.

    Same decoder logic: correction set is {I} ∪ {E_q for q∈[n], E∈single_errors};
    score by codeword-overlap; project; fid<0.5 → logical error.
    """
    K = code_states.shape[0]
    rng = np.random.default_rng(seed)
    outcomes = np.zeros(n_shots, dtype=np.int8)

    def _code_overlap(state):
        return sum(np.abs(np.vdot(code_states[k], state)) ** 2
                   for k in range(K))

    for shot in range(n_shots):
        alpha = rng.standard_normal(K) + 1j * rng.standard_normal(K)
        alpha /= np.linalg.norm(alpha)
        logical_state = alpha @ code_states

        noisy = noise_fn(logical_state, rng)

        best_overlap = _code_overlap(noisy)
        best_corrected = noisy
        for q in range(n_qudit):
            for E in single_errors:
                corrected = _apply_single_qudit_np(
                    noisy, E.conj().T, q, n_qudit, d)
                ov = _code_overlap(corrected)
                if ov > best_overlap:
                    best_overlap = ov
                    best_corrected = corrected

        coeffs = np.array([np.vdot(code_states[k], best_corrected)
                            for k in range(K)])
        projected = coeffs @ code_states
        norm = np.linalg.norm(projected)
        if norm > 1e-10:
            projected /= norm
        fid = float(np.abs(np.vdot(logical_state, projected)) ** 2)
        if fid < 0.5:
            outcomes[shot] = 1
    return outcomes


# ─────────── bootstrap CI ───────────

def bootstrap_ler_ci(outcomes: np.ndarray, n_resamples: int = 1000,
                     confidence_level: float = 0.95,
                     rng: Optional[np.random.Generator] = None
                     ) -> tuple:
    """Return (mean, ci_lo, ci_hi) by percentile bootstrap on bernoulli outcomes."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(outcomes)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(outcomes))
    # If all zeros or all ones, percentile bootstrap is degenerate; report (mean, mean, mean)
    if mean == 0.0 or mean == 1.0:
        return mean, mean, mean
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = outcomes[idx].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return mean, lo, hi


# ─────────── noise-fn factories ───────────

def make_flat_noise_fn(weights: np.ndarray, E_full: list,
                        group_starts: np.ndarray, d: int, n_qudit: int
                        ) -> Callable:
    """Per shot: idx ~ Cat(weights/sum), apply lifted op E_idx to state.

    Matches the channel the n = 9 reference code was optimized against (the twirled Meth
    Pauli channel on the weight-≤2 closure basis).
    """
    p = weights / weights.sum()
    n_ops = len(p)
    starts = np.asarray(group_starts)

    def noise_fn(state, rng):
        idx = int(rng.choice(n_ops, p=p))
        g = int(np.searchsorted(starts[1:], idx, side='right'))
        e = idx - int(starts[g])
        group = E_full[g]
        wires = tuple(int(w) for w in group['wires'])
        inv = tuple(int(x) for x in group['inverse_perm'])
        M = np.asarray(group['matrices'][e])
        return apply_local_np(state, M, wires, inv, d, n_qudit)
    return noise_fn


def make_multi_event_noise_fn(W_data: dict, gate_pairs: list,
                               d: int, n_qudit: int,
                               alpha: float = 1.0) -> Callable:
    """Per shot: per gate (q_a, q_b), sample per-role Paulis and compose.

    SENSITIVITY ONLY — per-gate-independent product is NOT the channel
    the n = 9 reference code was optimized against. Use to quantify the gap between
    "matched-loss channel" (default flat-685 sampler) and
    "per-gate-independent" composition.

    α-axis: rescale non-identity entries of each per-role 13-vector by α
    and renormalize per role.
    """
    from src.errors import qudit_hardware_error_basis, close_error_basis
    Id = np.eye(d, dtype=complex)
    base = qudit_hardware_error_basis(d)
    cross = close_error_basis(base)
    pauli_ops = [Id] + [np.asarray(o, dtype=complex) for o in base] + \
                [np.asarray(o, dtype=complex) for o in cross]
    n_paulis = len(pauli_ops)

    def _scaled(v):
        w = np.asarray(v, dtype=float).copy()
        if alpha != 1.0:
            w[1:] *= alpha
        return w / w.sum()
    w_ctrl = _scaled(W_data['control_weights'])
    w_tgt = _scaled(W_data['target_weights'])
    w_spec = _scaled(W_data['spectator_weights'])

    def noise_fn(state, rng):
        s = state.copy()
        for q_a, q_b in gate_pairs:
            for q in range(n_qudit):
                if q == q_a:
                    pa = int(rng.choice(n_paulis, p=w_ctrl))
                elif q == q_b:
                    pa = int(rng.choice(n_paulis, p=w_tgt))
                else:
                    pa = int(rng.choice(n_paulis, p=w_spec))
                if pa == 0:
                    continue  # identity
                s = _apply_single_qudit_np(s, pauli_ops[pa], q, n_qudit, d)
        return s
    return noise_fn


def make_flat_alpha_weights(weights: np.ndarray, alpha: float,
                              id_index: int = 0) -> np.ndarray:
    """Rescale non-identity entries by α, renormalize. The identity at
    flat index 0 is preserved in raw value and the total is renormalized."""
    w = weights.copy()
    mask = np.ones_like(w, dtype=bool)
    mask[id_index] = False
    w[mask] *= alpha
    s = w.sum()
    if s <= 0:
        raise ValueError(f"α={alpha} produced non-positive total weight")
    return w / s


# ─────────── main ───────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="Path to a code .npz checkpoint")
    p.add_argument("--weights",
                   default="results/round14_scoping/meth_pauli_weights.npz",
                   help="Path to Meth Pauli weights .npz (the 685-vector)")
    p.add_argument("--n_shots", type=int, default=3000)
    p.add_argument("--alpha_sweep", default="0.5,0.75,1.0,1.5,2.0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="Output CSV path (default: stdout-only)")
    p.add_argument("--label", default=None,
                   help="Code label written into CSV rows (default: checkpoint basename)")
    p.add_argument("--multi_event", action="store_true",
                   help="Sensitivity-only sampler: per-gate per-role product. "
                        "NOT the channel the n = 9 reference code was optimized against.")
    args = p.parse_args()

    weights_path = os.path.join(REPO, args.weights) \
        if not os.path.isabs(args.weights) else args.weights
    ckpt_path = os.path.join(REPO, args.checkpoint) \
        if not os.path.isabs(args.checkpoint) else args.checkpoint

    print(f"=== Pauli-frame Meth LER ({'multi_event' if args.multi_event else 'flat-685'}) ===")
    print(f"  checkpoint: {ckpt_path}")
    print(f"  weights:    {weights_path}")
    print(f"  n_shots:    {args.n_shots}")
    alphas = [float(x) for x in args.alpha_sweep.split(",")]
    print(f"  α sweep:    {alphas}")

    # Load weights
    W = np.load(weights_path, allow_pickle=True)
    weights = np.asarray(W['weights'])
    d_from_w = int(W['d'])
    n_qudit_from_w = int(W['n_qudit'])

    # Connections (a2a default). Read from checkpoint.
    data = np.load(ckpt_path, allow_pickle=True)
    connections = np.asarray(data['connections']).tolist()
    connections = [list(c) for c in connections]
    label = args.label or os.path.basename(ckpt_path).replace(".npz", "")

    # Encoder forward
    print(f"  building encoder + code_states ...")
    t0 = time.time()
    code_states, d, n_qudit, K = encoder_forward(ckpt_path, connections)
    print(f"  code_states {code_states.shape} dtype={code_states.dtype} "
          f"({time.time()-t0:.1f}s)")
    assert d == d_from_w and n_qudit == n_qudit_from_w

    # Structural basis (matches the weights ordering)
    from src.errors import ErrorModel, qudit_hardware_error_basis
    model = ErrorModel(d=d, n_qudit=n_qudit, distance=3, closed=True)
    E_full = model.build_grouped(verbose=False)
    n_full = sum(int(g['matrices'].shape[0]) for g in E_full)
    assert n_full == len(weights), \
        f"|E_full|={n_full} ≠ |weights|={len(weights)}"
    group_starts = flat_group_offsets(E_full)
    single_errors = [np.asarray(E, dtype=complex)
                     for E in qudit_hardware_error_basis(d)]

    # Sweep
    bootstrap_rng = np.random.default_rng(args.seed)
    rows = []
    for alpha in alphas:
        if args.multi_event:
            noise_fn = make_multi_event_noise_fn(
                {'control_weights': W['control_weights'],
                 'target_weights': W['target_weights'],
                 'spectator_weights': W['spectator_weights']},
                connections, d, n_qudit, alpha=alpha)
        else:
            w_alpha = make_flat_alpha_weights(weights, alpha)
            noise_fn = make_flat_noise_fn(
                w_alpha, E_full, group_starts, d, n_qudit)

        t0 = time.time()
        outcomes = simulate_per_shot_factored(
            code_states, noise_fn, single_errors, n_qudit, d,
            n_shots=args.n_shots, seed=args.seed + int(1000 * alpha))
        elapsed = time.time() - t0
        ler, lo, hi = bootstrap_ler_ci(outcomes, n_resamples=1000,
                                        rng=bootstrap_rng)
        rel = (hi - lo) / max(ler, 1e-12)
        print(f"  α={alpha:.3f}: LER={ler:.4e} [{lo:.4e}, {hi:.4e}] "
              f"(rel_w={rel:.2%}, {elapsed:.1f}s)")
        rows.append({
            'code': label, 'sampler': 'multi_event' if args.multi_event else 'flat685',
            'alpha': alpha, 'ler_mean': ler, 'ler_lo': lo, 'ler_hi': hi,
            'n_shots': args.n_shots, 'seed': args.seed,
        })

    if args.out:
        out_path = os.path.join(REPO, args.out) \
            if not os.path.isabs(args.out) else args.out
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        write_header = not os.path.exists(out_path)
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"  appended {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
