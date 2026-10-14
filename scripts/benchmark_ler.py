#!/usr/bin/env python3
"""
Benchmark logical error rate for any code in the catalog.

Usage:
  python3 scripts/benchmark_ler.py --code qutrit_d3 --n_shots 5000
  python3 scripts/benchmark_ler.py --code five_qudit_d3 --noise hardware --n_shots 5000
  python3 scripts/benchmark_ler.py --code dephasing_d3 --noise hardware --n_shots 2000
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.catalog import load_code, list_codes
from src.errors import make_hardware_noise_fn
from src.errors import qudit_hardware_error_basis, _embed_single_qudit
from src.simulation import (
    simulate_ler_with_detection, simulate_ler_with_correction,
    simulate_ler_with_correction_factored, sweep_logical_error_rate,
)


def main():
    parser = argparse.ArgumentParser(description="Benchmark LER for a VarQEC code")
    parser.add_argument("--code", required=True, help=f"Code name. Available: {list_codes()}")
    parser.add_argument("--noise", default="hardware", choices=["hardware"],
                        help="Noise model for testing")
    parser.add_argument("--p_range", default="0.001,0.005,0.01,0.02,0.05,0.1,0.15,0.2",
                        help="Comma-separated physical error rates")
    parser.add_argument("--n_shots", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decoder", default="auto",
                        choices=["auto", "dense", "factored"],
                        help="Correction decoder. 'auto' uses factored when "
                             "dim>500 or d>=5 or dist>=3 (same predicate as "
                             "scripts/train.py); 'dense' / 'factored' override.")
    args = parser.parse_args()

    p_rates = [float(p) for p in args.p_range.split(",")]

    print(f"Loading code: {args.code}")
    code_data = load_code(args.code)
    code_states = code_data['code_states']
    meta = code_data['metadata']
    d = meta['d']
    n_qudit = meta['n_qudit']
    distance = meta['distance']
    K = meta['K']

    dim = d ** n_qudit
    print(f"  d={d}, n={n_qudit}, K={K}, distance={distance}, dim={dim}")
    if 'final_loss' in meta:
        print(f"  Training loss: {meta.get('final_loss', 'N/A')}")

    # Decoder selection: same predicate as scripts/train.py's use_factored.
    factored_needed = dim > 500 or d >= 5 or distance >= 3
    if args.decoder == "auto":
        decoder_mode = "factored" if factored_needed else "dense"
    else:
        decoder_mode = args.decoder
    print(f"  decoder: {args.decoder} -> {decoder_mode} "
          f"(dim={dim}, factored needed: {'yes' if factored_needed else 'no'})")

    # Build correction operators for the selected path.
    E_corr_ops = []
    single_errors = []
    if distance >= 3:
        single_errors = qudit_hardware_error_basis(d)
        if decoder_mode == "dense":
            for q in range(n_qudit):
                for E in single_errors:
                    E_corr_ops.append(_embed_single_qudit(E, q, n_qudit, d))

    # Run benchmarks
    results = {'p_rates': p_rates, 'code': args.code, 'distance': distance,
               'd': d, 'n_qudit': n_qudit, 'n_shots': args.n_shots}

    if distance == 2:
        print(f"\nDistance-2 detection benchmark ({args.n_shots} shots):")
        print(f"{'p':>7} | {'det_frac':>8} | {'undet_err':>9} | {'ps_fid':>7}")
        print("-" * 42)
        det_fracs, undet_errs, ps_fids, raw_fids = [], [], [], []
        for p in p_rates:
            noise = make_hardware_noise_fn(d, n_qudit, p)
            r = simulate_ler_with_detection(code_states, noise, args.n_shots, seed=args.seed)
            det_fracs.append(r['detected_fraction'])
            undet_errs.append(r['undetected_error_rate'])
            ps_fids.append(r['post_selected_fidelity'])
            raw_fids.append(r['mean_raw_fidelity'])
            print(f"{p:7.3f} | {r['detected_fraction']:8.4f} | {r['undetected_error_rate']:9.4f} | {r['post_selected_fidelity']:7.4f}")
        results.update({'det_fracs': det_fracs, 'undet_errs': undet_errs,
                        'ps_fids': ps_fids, 'raw_fids': raw_fids})

    elif distance >= 3:
        print(f"\nDistance-{distance} correction benchmark ({args.n_shots} shots, {decoder_mode}):")
        print(f"{'p':>7} | {'LER':>7} | {'F_corr':>7} | {'F_raw':>7}")
        print("-" * 38)
        lers, fid_corrs, fid_raws = [], [], []
        for p in p_rates:
            noise = make_hardware_noise_fn(d, n_qudit, p)
            if decoder_mode == "factored":
                r = simulate_ler_with_correction_factored(
                    code_states, noise, single_errors, n_qudit, d,
                    args.n_shots, seed=args.seed)
            else:
                r = simulate_ler_with_correction(
                    code_states, noise, E_corr_ops,
                    args.n_shots, seed=args.seed)
            lers.append(r['logical_error_rate'])
            fid_corrs.append(r['mean_fidelity'])
            fid_raws.append(r['mean_raw_fidelity'])
            marker = " *QEC*" if r['logical_error_rate'] < p else ""
            print(f"{p:7.3f} | {r['logical_error_rate']:7.4f} | {r['mean_fidelity']:7.4f} | {r['mean_raw_fidelity']:7.4f}{marker}")
        results.update({'lers': lers, 'fid_corrs': fid_corrs, 'fid_raws': fid_raws,
                        'decoder': decoder_mode})

    # Save results
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "simulations")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"ler_{args.code}_{args.noise}.npz")
    np.savez(save_path, **{k: np.array(v) if isinstance(v, list) else v
                           for k, v in results.items()})
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
