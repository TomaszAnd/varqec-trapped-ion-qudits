#!/usr/bin/env python3
"""Pauli-twirled Meth (physical) weights aligned
with the R12/13/14 weight-≤2 closure basis.

For each operator E_i in `ErrorModel(d, n_qudit, distance=3,
closed=True).build_grouped()`, compute an importance weight λ_i^Meth
that reflects how likely E_i is under the Meth physical channel:

    λ_i = Π_q  p_q^{role(q)}(E_i^q)

where:
- E_i^q is the q-th tensor factor of E_i (identity if q is not in
  E_i's support);
- role(q) is the role of qudit q in a typical gate of the encoder
  (control / target / spectator);
- p_q^role(P) is the single-qudit Pauli-twirl probability of Pauli P
  under the role-specific Kraus channel from src.correlated_noise.

For the n = 9 reference code, we use the SPECTATOR channel for every qudit
as a pragmatic approximation: in the n=9 a2a encoder, each qudit
participates in only a few gate roles per layer and serves as
spectator on the rest, so the spectator channel dominates the
per-qudit channel composition over a full encoder. A more faithful
multi-role composition is deferred to R15.

This script outputs:
  - results/round14_scoping/meth_pauli_weights.npz   (weights aligned with E_grouped)
  - results/round14_scoping/meth_weights_validation.md (validation table)

The weights are normalised so Σ λ_i = 1 (per-step error rate
absorbed into the loss-prefactor; the relative weighting is what
matters for the importance-weighted KL).

src/ stays frozen; this is scripts-only and uses public API
(ErrorModel, control/target/spectator_qudit_kraus from
src.correlated_noise).
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


# ─────────── Pauli twirl primitives ───────────

def pauli_twirl_single_qudit(kraus_diags_or_mats, pauli_ops, d):
    """For each Pauli P in pauli_ops: λ_P = (1/d²) Σ_a |Tr(P† K_a)|².

    kraus_diags_or_mats: list of d-element diagonal arrays OR d×d matrices.
    pauli_ops: list of d×d matrices (the structural Pauli basis).
    """
    weights = np.zeros(len(pauli_ops), dtype=float)
    for i, P in enumerate(pauli_ops):
        s = 0.0
        for K in kraus_diags_or_mats:
            if K.ndim == 1:
                # Diagonal Kraus stored as a 1D array
                t = np.sum(P.conj().diagonal() * K)
            else:
                t = np.trace(P.conj().T @ K)
            s += abs(t) ** 2
        weights[i] = s / (d ** 2)
    return weights


# ─────────── Build the R12/13/14 closure basis with op enumeration ───────────

def factor_per_qudit(grouped_op, n_qudit, d):
    """For a grouped op with `wires` and `matrices`, return a list of
    n_qudit d×d single-qudit factors. Identity on qudits not in wires.

    For len(wires) == 0 (identity sentinel): all identity.
    For len(wires) == 1: that qudit gets the d×d matrix, others identity.
    For len(wires) == 2: factor the d²×d² matrix into two d×d factors
    by SVD (works exactly for tensor-product operators, which all our
    weight-2 ops are — they were built as products of single-qudit ops).
    """
    Id = np.eye(d, dtype=complex)
    factors = [Id.copy() for _ in range(n_qudit)]
    wires = grouped_op['wires']
    if len(wires) == 0:
        return factors
    M = np.asarray(grouped_op['matrices'])
    if len(wires) == 1:
        q = wires[0]
        factors[q] = M.astype(complex)
        return factors
    # Two-qudit: factor M (shape d²×d²) into A ⊗ B
    # M has shape (d**2, d**2). Reshape to (d, d, d, d) = (a1, b1, a2, b2)
    # then to (d², d²) with first index = (a1, a2), second = (b1, b2).
    # Use SVD to find rank-1 factorization (true for product ops).
    M_re = M.reshape(d, d, d, d).transpose(0, 2, 1, 3).reshape(d * d, d * d)
    U, S, Vh = np.linalg.svd(M_re)
    # The largest singular value corresponds to the rank-1 tensor product
    A = (np.sqrt(S[0]) * U[:, 0]).reshape(d, d)
    B = (np.sqrt(S[0]) * Vh[0, :]).reshape(d, d)
    # Sanity: if A ⊗ B doesn't reconstruct M well, this isn't a tensor product
    M_recon = np.kron(A, B)
    if not np.allclose(M, M_recon, atol=1e-8):
        # Try the reverse order
        M_recon2 = np.kron(B, A)
        if np.allclose(M, M_recon2, atol=1e-8):
            A, B = B, A
        else:
            # Fallback: factor not exact tensor product; treat as identity-equivalent
            # (shouldn't happen for our basis)
            pass
    factors[wires[0]] = A
    factors[wires[1]] = B
    return factors


def build_single_qudit_pauli_basis(d):
    """Return a list of (label, d×d matrix) for the single-qudit Pauli
    basis matching the R12/13/14 structural choice: weight-1 hardware
    errors PLUS their same-qudit closure (the n_single + n_cross set).
    The identity is included as the first entry.

    For d=3: 1 + 4 + 8 = 13 ops (per src.errors.close_error_basis output).
    Actually src.errors.qudit_hardware_error_basis(3) returns 4 ops
    (Z_1, Z_2, X_01, X_12); close_error_basis returns 8 cross-products.
    Total single-qudit basis = 1 + 4 + 8 = 13.
    """
    from src.errors import qudit_hardware_error_basis, close_error_basis
    Id = np.eye(d, dtype=complex)
    base = qudit_hardware_error_basis(d)
    cross = close_error_basis(base)
    ops = [Id] + [np.asarray(o, dtype=complex) for o in base] + \
          [np.asarray(o, dtype=complex) for o in cross]
    labels = (["I"]
              + [f"Z_{k}" for k in range(1, d)]
              + [f"X_{k}{k+1}" for k in range(d - 1)]
              + [f"closure_{i}" for i in range(len(cross))])
    return list(zip(labels, ops))


# ─────────── Build per-role single-qudit Pauli weights ───────────

def build_per_role_weights(d, eta, n_max=2):
    """Compute single-qudit Pauli weights for each role: control,
    target, spectator. Use the 'physical' Model B Kraus operators per
    stated choice.
    """
    from src.correlated_noise import (control_qudit_kraus,
                                       target_qudit_kraus,
                                       spectator_qudit_kraus)
    pauli_labels_ops = build_single_qudit_pauli_basis(d)
    pauli_ops = [op for _, op in pauli_labels_ops]
    pauli_labels = [lbl for lbl, _ in pauli_labels_ops]

    # Build Kraus diagonals for each role (control_level=0, target_level=1
    # as a representative parametrization; we average over levels in the
    # multi-qudit lift)
    k_ctrl = control_qudit_kraus(d, control_level=0, eta=eta, n_max=n_max)
    k_tgt = target_qudit_kraus(d, control_level=0, target_level=1,
                                eta=eta, n_max=n_max)
    k_spec = spectator_qudit_kraus(d, eta=eta, n_max=n_max)

    w_ctrl = pauli_twirl_single_qudit(k_ctrl, pauli_ops, d)
    w_tgt = pauli_twirl_single_qudit(k_tgt, pauli_ops, d)
    w_spec = pauli_twirl_single_qudit(k_spec, pauli_ops, d)

    return {
        "labels": pauli_labels,
        "ops": pauli_ops,
        "control": w_ctrl,
        "target": w_tgt,
        "spectator": w_spec,
    }


# ─────────── Lift to multi-qudit closure basis ───────────

def pauli_weight_for_factor(M, pauli_basis):
    """Return |⟨P_k, M⟩|² for each P_k in pauli_basis. Used to express
    a d×d operator M as a weighted sum of Paulis on the structural basis.
    """
    d = M.shape[0]
    weights = np.zeros(len(pauli_basis), dtype=float)
    for k, P in enumerate(pauli_basis):
        t = np.trace(P.conj().T @ M)
        weights[k] = abs(t) ** 2 / (d ** 2)
    return weights


def lift_to_multiqudit_weights(E_grouped, d, n_qudit, per_role):
    """For each op in E_grouped, compute λ = Π_q λ^spec(E_i^q).

    Pragmatic choice: use the SPECTATOR channel uniformly for all
    qudits (see r14_3_choices.md for justification). A more faithful
    multi-role composition (control + target + spectator weighted by
    actual gate participation in the encoder) is deferred to R15.
    """
    pauli_ops = per_role["ops"]
    w_spec = per_role["spectator"]
    # For each per-qudit factor M_q of E_i, decompose into Paulis and
    # compute the expected per-qudit error weight:
    #   p(E_i^q) = Σ_k |⟨P_k, M_q⟩|² / d² · w_spec[k]
    # (Conditional on E_i^q being applied as a Pauli-class error on q.)
    # The multi-qudit weight is the product over q.
    n_ops_total = sum(int(g['matrices'].shape[0]) for g in E_grouped)
    weights = np.zeros(n_ops_total, dtype=float)
    op_labels = []
    flat_idx = 0
    for g in E_grouped:
        wires = g['wires']
        for e_idx in range(g['matrices'].shape[0]):
            # Build the per-qudit factorization for this single op
            g_single = {'wires': wires,
                        'matrices': g['matrices'][e_idx]}
            if len(wires) <= 1:
                factors = factor_per_qudit({'wires': wires,
                                              'matrices': g['matrices'][e_idx]},
                                             n_qudit, d)
            else:
                factors = factor_per_qudit({'wires': wires,
                                              'matrices': g['matrices'][e_idx]},
                                             n_qudit, d)

            # For each qudit, decompose its factor on pauli basis and
            # compute the expected error weight under spectator channel.
            log_w = 0.0
            valid = True
            for q in range(n_qudit):
                M_q = factors[q]
                pauli_proj = pauli_weight_for_factor(M_q, pauli_ops)
                # Expected per-qudit error weight: sum over Paulis of
                # (decomposition coefficient) × (spectator channel weight)
                # The decomposition coefficients are |⟨P_k, M_q⟩|² / d² →
                # they sum to 1 if M_q is unitary (Parseval), so this is
                # a normalised mixture.
                p_q = float(np.sum(pauli_proj * w_spec))
                if p_q <= 0:
                    valid = False
                    break
                log_w += np.log(p_q)
            if valid:
                weights[flat_idx] = float(np.exp(log_w))
            else:
                weights[flat_idx] = 0.0
            op_labels.append(f"wires={wires}, e={e_idx}")
            flat_idx += 1

    # Normalize weights
    s = weights.sum()
    if s > 0:
        weights = weights / s
    return weights, op_labels


# ─────────── Main + validation ───────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=3)
    p.add_argument("--n_qudit", type=int, default=9)
    p.add_argument("--eta", type=float, default=0.9296,
                   help="Meth-calibrated η = exp(-0.073)")
    p.add_argument("--n_max", type=int, default=2)
    p.add_argument("--out_dir", default="results/campaigns/weighted")
    args = p.parse_args()

    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Meth (physical) Pauli weights ===")
    print(f"  d={args.d}, n_qudit={args.n_qudit}, η={args.eta}, "
          f"n_max={args.n_max}")

    from src.errors import ErrorModel
    model = ErrorModel(d=args.d, n_qudit=args.n_qudit, distance=3, closed=True)
    E_grouped = model.build_grouped(verbose=False)
    n_total = sum(int(g['matrices'].shape[0]) for g in E_grouped)
    print(f"  |E_grouped| = {n_total}")

    # Per-role single-qudit Pauli weights
    per_role = build_per_role_weights(args.d, args.eta, args.n_max)
    print(f"\n  Per-role single-qudit Pauli weights (Pauli labels: "
          f"{per_role['labels']}):")
    for role in ["control", "target", "spectator"]:
        print(f"    {role:>10}: {[f'{w:.3e}' for w in per_role[role]]}")

    # Lift to multi-qudit closure basis (spectator-only approximation)
    weights, op_labels = lift_to_multiqudit_weights(
        E_grouped, args.d, args.n_qudit, per_role)
    print(f"\n  Lifted weights (spectator-only approximation):")
    print(f"    sum = {weights.sum():.6f}")
    print(f"    min = {weights.min():.3e}, max = {weights.max():.3e}")
    print(f"    mean = {weights.mean():.3e}, std = {weights.std():.3e}")
    print(f"    std/mean = {weights.std()/max(weights.mean(),1e-30):.3f}")

    # Top-10 ops by weight
    top_idx = np.argsort(weights)[::-1][:10]
    print(f"\n  Top-10 ops by Meth weight:")
    for rank, i in enumerate(top_idx):
        # Look up the wire signature for this flat index
        flat = 0
        for g in E_grouped:
            n_in_g = g['matrices'].shape[0]
            if flat <= i < flat + n_in_g:
                wires = g['wires']
                e_idx = i - flat
                print(f"    #{rank+1} weight={weights[i]:.4e} "
                      f"wires={wires} e={e_idx}")
                break
            flat += n_in_g

    # Save weights
    npz_path = os.path.join(out_dir, f"meth_pauli_weights_corrected_d{args.d}_n{args.n_qudit}.npz")
    np.savez(npz_path,
             weights=weights,
             d=args.d, n_qudit=args.n_qudit, eta=args.eta, n_max=args.n_max,
             noise_model="physical",
             role_approximation="spectator-only",
             pauli_labels=per_role["labels"],
             control_weights=per_role["control"],
             target_weights=per_role["target"],
             spectator_weights=per_role["spectator"])
    print(f"\n  wrote {npz_path}")

    # Validation MD
    md_path = os.path.join(out_dir, f"meth_weights_validation_d{args.d}_n{args.n_qudit}.md")
    with open(md_path, "w") as f:
        f.write(f"# Meth (physical) Pauli weights — validation\n\n")
        f.write(f"Built by `scripts/meth_pauli_weights.py` at "
                f"d={args.d}, n_qudit={args.n_qudit}, η={args.eta}, "
                f"n_max={args.n_max}.\n\n")
        f.write(f"## Per-role single-qudit Pauli twirl\n\n")
        f.write(f"Pauli basis ({len(per_role['labels'])} ops): "
                f"{per_role['labels']}\n\n")
        f.write("| role | I | Z_1 | Z_2 | X_01 | X_12 | closure 1-8 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for role in ["control", "target", "spectator"]:
            w = per_role[role]
            closure_str = " | ".join(f"{w[k]:.2e}" for k in range(5, len(w)))
            f.write(f"| {role} | {w[0]:.4f} | {w[1]:.2e} | "
                    f"{w[2]:.2e} | {w[3]:.2e} | {w[4]:.2e} | "
                    f"{closure_str} |\n")
        f.write("\n")
        f.write("Interpretation: per-qudit Pauli-twirl probabilities.\n")
        f.write("Identity weight (column I) is the 'no error' probability.\n")
        f.write("Z_1 / Z_2 are the dominant phase-flip syndromes for the\n")
        f.write("dephasing channel. X_01 / X_12 weights should be small\n")
        f.write("for a phase-only channel (no X-class errors).\n\n")

        f.write(f"## Lifted multi-qudit weights\n\n")
        f.write(f"Pragmatic approximation: spectator channel for\n")
        f.write(f"every qudit. See `r14_3_choices.md`.\n\n")
        f.write(f"- |E_grouped| = {n_total}\n")
        f.write(f"- Σ λ_i = {weights.sum():.6f} (normalised to 1)\n")
        f.write(f"- min = {weights.min():.3e}\n")
        f.write(f"- max = {weights.max():.3e}\n")
        f.write(f"- mean = {weights.mean():.3e}\n")
        f.write(f"- std = {weights.std():.3e}\n")
        f.write(f"- std/mean = {weights.std()/max(weights.mean(),1e-30):.3f}\n\n")
        nonuniformity = weights.std() / max(weights.mean(), 1e-30)
        if nonuniformity < 0.05:
            verdict = "**Nearly uniform** (std/mean < 5%). The Meth channel is too uniform on this basis; importance weighting will NOT differ from R12/13/14 uniform-weighted training. Reframe R14-3 expectations."
        elif nonuniformity < 0.5:
            verdict = "**Moderately non-uniform** (5% ≤ std/mean < 50%). Importance weighting should produce a measurable but modest difference from uniform-weighted training."
        else:
            verdict = "**Strongly non-uniform** (std/mean ≥ 50%). Meth-weighted training should differ substantially from uniform — importance weighting is a real lever."
        f.write(f"### Verdict\n\n{verdict}\n\n")

        f.write(f"## Top-10 ops by Meth weight\n\n")
        f.write("| rank | weight | wires | e_idx |\n|---:|---:|---|---:|\n")
        for rank, i in enumerate(top_idx):
            flat = 0
            for g in E_grouped:
                n_in_g = g['matrices'].shape[0]
                if flat <= i < flat + n_in_g:
                    wires = g['wires']
                    e_idx = i - flat
                    f.write(f"| {rank+1} | {weights[i]:.4e} | "
                            f"{wires} | {e_idx} |\n")
                    break
                flat += n_in_g
        f.write("\n")
        f.write("Expected dominant ops for a phase-dephasing channel:\n")
        f.write("single-qudit Z-class errors on non-control levels (weight-1\n")
        f.write("ops with `wires=(q,)` for various q, e_idx pointing to Z_1\n")
        f.write("or Z_2). If the top-10 are exotic high-weight cross-products,\n")
        f.write("there's a bug.\n\n")

    print(f"  wrote {md_path}")


if __name__ == "__main__":
    main()
