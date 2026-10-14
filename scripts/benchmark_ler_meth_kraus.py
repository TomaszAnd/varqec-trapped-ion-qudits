#!/usr/bin/env python3
"""Kraus-MC LER against the physical-mode Meth phase channel.

Thin wrapper around `src.simulation.make_correlated_dephasing_noise_fn`
(which constructs control/target/spectator dephasing Kraus per gate from
`src.correlated_noise`) and `simulate_ler_with_correction_factored`. The
benchmark sweeps η, the Meth phase-noise parameter (η = exp(-σ_p²)),
across a small grid bracketing the calibrated value 0.9296.

**Phase-only, by design.** The weight derivation's twirl processed only the phase
component of `combined_gate_kraus` (by design).
`subspace_depolarizing_kraus` exists in `src/correlated_noise.py` but is
NOT composed in here; including it would compare a phase-trained code
against a richer channel, which is not the matched head-to-head.

**Physical mode** (level-dependent f_k = k for the control qudit) per
the stated design choice.

η axis vs α axis (pipeline a)
  η rescales the underlying Gaussian phase variance σ_p² = -log(η); a
  structural change to the Kraus operators. α (in pipeline a) rescales
  the post-twirl Pauli probabilities. These are different axes.

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
sys.path.insert(0, os.path.join(REPO, "scripts"))

# Canonical encoder-forward lives in the decoder package.
from src.decoders._common import encoder_forward

# bootstrap_ler_ci: the canonical copy lives in benchmark_ler_meth_pauli (its
# downstream consumers already import it from there). Re-exported here so the
# name stays importable from this module for any historical caller.
from benchmark_ler_meth_pauli import bootstrap_ler_ci  # noqa: F401


# ─────────── per-shot factored-decoder LER (matches pipeline (a)'s helper) ───────────

def _apply_single_qudit_np(state: np.ndarray, op: np.ndarray, q: int,
                            n_qudit: int, d: int) -> np.ndarray:
    """Apply d×d op to qudit q. Mirrors src.simulation._apply_factored_op."""
    shape = (d,) * n_qudit
    s = state.reshape(shape)
    s = np.tensordot(op, s, axes=([1], [q]))
    s = np.moveaxis(s, 0, q)
    return s.reshape(-1)


def simulate_per_shot_factored(code_states: np.ndarray,
                                noise_fn: Callable,
                                single_errors: list,
                                n_qudit: int, d: int,
                                n_shots: int,
                                seed: int) -> np.ndarray:
    """Per-shot bernoulli outcomes (1 = logical error). Mirrors
    src.simulation.simulate_ler_with_correction_factored."""
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


# ─────────── gate_pairs construction (matches scripts/train_r14_3_meth.py) ───────────

def build_gate_pairs(n_qudit: int, connections: list) -> list:
    """For each (q_a, q_b) edge, emit (q_a, q_b, ctrl_level=0, tgt_level=1).

    Matches the encoder's gate layout: each MS-style gate is parametrized by
    a level pair (j, k); for d=3 the canonical choice is the (0, 1) sublevel
    (ctrl_level=0, tgt_level=1). This is what the channel applies per gate.
    """
    return [(int(qa), int(qb), 0, 1) for qa, qb in connections]


# ─────────── main ───────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="Path to a code .npz checkpoint")
    p.add_argument("--n_shots", type=int, default=3000)
    p.add_argument("--eta_sweep",
                   default="0.90,0.9296,0.95,0.97,0.99",
                   help="η values to sweep; η = exp(-σ_p²). "
                        "Meth-calibrated 0.9296 is the default centerpoint.")
    p.add_argument("--n_max", type=int, default=5,
                   help="Kraus truncation order (default 5; the weight derivation used 2)")
    p.add_argument("--noise_model", default="physical",
                   choices=["physical", "simplified"],
                   help="physical=level-dependent f_k=k (default); "
                        "simplified=Meth literal Eq.J3 uniform f=2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="Output CSV path (default: stdout-only)")
    p.add_argument("--label", default=None,
                   help="Code label written into CSV rows")
    args = p.parse_args()

    ckpt_path = os.path.join(REPO, args.checkpoint) \
        if not os.path.isabs(args.checkpoint) else args.checkpoint

    print(f"=== Kraus-MC physical-phase Meth LER ===")
    print(f"  checkpoint:  {ckpt_path}")
    print(f"  n_shots:     {args.n_shots}")
    etas = [float(x) for x in args.eta_sweep.split(",")]
    print(f"  η sweep:     {etas}")
    print(f"  n_max:       {args.n_max}")
    print(f"  noise_model: {args.noise_model} (phase-only, no subspace_depol)")

    # Build encoder + code_states
    data = np.load(ckpt_path, allow_pickle=True)
    connections = [list(c) for c in np.asarray(data['connections']).tolist()]
    label = args.label or os.path.basename(ckpt_path).replace(".npz", "")

    print(f"  building encoder + code_states ...")
    t0 = time.time()
    code_states, d, n_qudit, K = encoder_forward(ckpt_path, connections)
    print(f"  code_states {code_states.shape} dtype={code_states.dtype} "
          f"({time.time()-t0:.1f}s)")

    # Build gate_pairs + decoder correction set
    from src.simulation import make_correlated_dephasing_noise_fn
    from src.errors import qudit_hardware_error_basis
    gate_pairs = build_gate_pairs(n_qudit, connections)
    single_errors = [np.asarray(E, dtype=complex)
                     for E in qudit_hardware_error_basis(d)]

    # Sweep
    bootstrap_rng = np.random.default_rng(args.seed)
    rows = []
    for eta in etas:
        noise_fn = make_correlated_dephasing_noise_fn(
            n_qudits=n_qudit, d=d, gate_pairs=gate_pairs,
            eta=eta, n_max=args.n_max, noise_model=args.noise_model)

        t0 = time.time()
        outcomes = simulate_per_shot_factored(
            code_states, noise_fn, single_errors, n_qudit, d,
            n_shots=args.n_shots, seed=args.seed + int(10000 * eta))
        elapsed = time.time() - t0
        ler, lo, hi = bootstrap_ler_ci(outcomes, n_resamples=1000,
                                        rng=bootstrap_rng)
        rel = (hi - lo) / max(ler, 1e-12)
        print(f"  η={eta:.4f}: LER={ler:.4e} [{lo:.4e}, {hi:.4e}] "
              f"(rel_w={rel:.2%}, {elapsed:.1f}s)")
        rows.append({
            'code': label, 'noise_model': args.noise_model,
            'eta': eta, 'ler_mean': ler, 'ler_lo': lo, 'ler_hi': hi,
            'n_max': args.n_max,
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
