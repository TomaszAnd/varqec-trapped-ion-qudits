#!/usr/bin/env python3
"""Multi-start "seed race" for VarQEC code discovery -- CLI wrapper.

The core protocol (evaluate_candidate, build_race, run_race) lives in
src/seed_race.py; the seed race is multi-start orchestration above the
differentiable backend, so it gets its own module. This file is a thin
argparse wrapper and re-exports the core functions for backward compatibility
(tests import `evaluate_candidate` / `run_race` from `seed_race`).

Usage:
    python3 scripts/seed_race.py --d 3 --n 5 --distance 3 --layers 2 \
        --num-seeds 100 --steps 3000 --sample-frac 0.02
"""
import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Re-export the core so `from seed_race import run_race, evaluate_candidate`
# keeps working (tests/test_seed_race.py, and any downstream scripts).
from src.seed_race import (evaluate_candidate, build_race, run_race,  # noqa: F401
                           CAMPAIGN_SAMPLE_FRAC, CAMPAIGN_SAMPLER)


def main():
    p = argparse.ArgumentParser(
        description="Multi-start seed race for VarQEC code discovery")
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--distance", type=int, default=2)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--num-seeds", type=int, default=100)
    p.add_argument("--steps", type=int, default=3000,
                   help="Per-seed step budget; honored in full until the "
                        "first certified record exists (audit/04 H2).")
    p.add_argument("--target-loss", type=float, default=1e-6)
    p.add_argument("--sample-frac", type=float, default=CAMPAIGN_SAMPLE_FRAC,
                   help="Per-group sampling fraction for the per-step loss "
                        "(1.0 = full batch). Sampled losses only trigger "
                        "certification, never grant it (audit/04 H1). Default "
                        "is the Stage-H' campaign budget (knee f*=0.4 + margin); "
                        "do NOT go below ~0.4 (gradient-corruption cliff).")
    p.add_argument("--noise", choices=["full", "dephasing"], default="full")
    p.add_argument("--entangler", choices=["ms", "ls"], default="ms")
    p.add_argument("--connectivity", choices=["all-to-all", "ring"],
                   default="all-to-all")
    p.add_argument("--mercy-baseline", type=int, default=None,
                   help="Optional externally-known record (in steps) to seed "
                        "the mercy rule. Default: none -- no seed is killed "
                        "before a certified record exists.")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--sampler", choices=["uniform", "importance"],
                   default=CAMPAIGN_SAMPLER,
                   help="Subsampled-loss sampler when --sample-frac < 1: "
                        "'uniform' per-group subsampling or 'importance' "
                        "(Neyman + within-group importance sampling). Campaign "
                        "default = importance.")
    p.add_argument("--trigger-margin", type=float, default=0.0,
                   help="Sampled loss must be below target*(1+margin) to arm a "
                        "full-batch certification (H false-trigger suppression).")
    p.add_argument("--trigger-patience", type=int, default=1,
                   help="Consecutive under-margin steps required before a "
                        "full-batch certification is attempted.")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    out = args.out
    if out is None:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = os.path.join(
            repo, "results", "params",
            f"seed_race_d{args.d}_n{args.n}_dist{args.distance}"
            f"_{args.layers}L.npz")

    t0 = time.time()
    res = run_race(args.d, args.n, args.distance, args.layers,
                   args.num_seeds, args.steps,
                   target_loss=args.target_loss,
                   sample_frac=args.sample_frac, noise=args.noise,
                   entangler=args.entangler,
                   connectivity=args.connectivity,
                   mercy_baseline=args.mercy_baseline, out=out, lr=args.lr,
                   first_seed=args.first_seed, sampler=args.sampler,
                   trigger_margin=args.trigger_margin,
                   trigger_patience=args.trigger_patience)
    print(f"race finished in {time.time() - t0:.0f}s; "
          f"certified={res['certified']}; sampler={res['sampler']}; "
          f"race_op_evs={res['race_op_evs']}; "
          f"op_evs_at_first_certified={res['op_evs_at_first_certified']}")


if __name__ == "__main__":
    main()
