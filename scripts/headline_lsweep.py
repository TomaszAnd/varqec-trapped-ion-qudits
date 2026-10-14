#!/usr/bin/env python3
"""STAGE 2 — headline 2x recovery: best-of-N + L-sweep at n=5 (CPU, NOT GPU-blocked).

The 1.46x headline used the Stage-H certified n=5 code (LER 5.23e-4). n=5 is dim 243 --
a proper best-of-8 x L-sweep is minutes/seed on CPU. Train ((5,3,3))_3, all-to-all, MS,
corrected --noise full, L in {16,20,24,28}, best-of-8 seeds each (keep best-by-loss per L).
Measure each L-winner's i.i.d. hardware-error LER at p=0.05 (300k shots); keep the
lowest-LER code. Re-measure THAT code vs [[5,1,3]] at >=300k shots (->1M if CIs still
touch the 2x, i.e. VarQEC upper CI not clearly below ~3.8e-4).

Verdict: does a deeper/best-of-N n=5 code reach <=3.8e-4 (recovering ~2x)? If yes -> update
headline; if the floor is genuinely ~5e-4 -> the 1.46x is the real best-practice result and
the published 3.67e-4 was a favourable seed/connectivity. Either way CPU-settled.

Writes -> results/campaigns/headline_p05_Lsweep/.
"""
from __future__ import annotations
import os, sys, time, json, math
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import numpy as np
from src.seed_race import run_race
from src.decoders._common import encoder_forward
from src.catalog import load_code
from src.errors import qudit_hardware_error_basis, make_hardware_noise_fn
from src.simulation import simulate_ler_with_correction_factored

OUTDIR = os.path.join(REPO, "results", "campaigns", "headline_p05_Lsweep")
D, N, P = 3, 5, 0.05
LAYERS = [16, 20, 24, 28]
SEEDS = 8
STEPS = 3000
TARGET_2X = 3.8e-4  # VarQEC LER at/below this recovers ~2x vs [[5,1,3]] (~7.6e-4)


def wilson(k, n, z=1.96):
    ph = k / n; d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), c + h)


def ler_at(cs, single, n_shots, seed):
    r = simulate_ler_with_correction_factored(
        cs, make_hardware_noise_fn(D, N, P), single, N, D, n_shots=n_shots, seed=seed)
    ler = float(r["logical_error_rate"]); k = int(round(ler * n_shots))
    return ler, k, wilson(k, n_shots)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    conns = [[i, j] for i in range(N) for j in range(i + 1, N)]
    single = [np.asarray(E, complex) for E in qudit_hardware_error_basis(D)]
    log = {"config": {"layers": LAYERS, "seeds": SEEDS, "steps": STEPS, "p": P,
                      "target_2x": TARGET_2X, "noise": "corrected full, a2a, MS"},
           "per_L": []}

    best = None  # (ler, L, npz, cs)
    for L in LAYERS:
        out = os.path.join(OUTDIR, f"n5_L{L}.npz")
        t = time.time()
        print(f"\n=== L={L}: best-of-{SEEDS} training ===", flush=True)
        res = run_race(d=D, n=N, distance=3, layers=L, num_seeds=SEEDS, steps=STEPS,
                       target_loss=1e-9, sample_frac=1.0, sampler="uniform",
                       trigger_margin=0.25, trigger_patience=5, out=out, first_seed=0,
                       verbose=False, connectivity="all-to-all", entangler="ms", noise="full")
        floss = float(res["result"]["full_loss"])
        cs, *_ = encoder_forward(out, conns)
        ler, k, ci = ler_at(cs, single, 300_000, seed=42)
        row = {"L": L, "best_full_loss": floss, "seed": int(res["result"]["seed"]),
               "ler_300k": ler, "wilson95": list(ci), "wall_s": round(time.time() - t, 1)}
        log["per_L"].append(row)
        print(f"  L={L}: loss={floss:.3e} LER={ler:.4e} CI=[{ci[0]:.3e},{ci[1]:.3e}] "
              f"({row['wall_s']}s)", flush=True)
        if best is None or ler < best[0]:
            best = (ler, L, out, cs)
        json.dump(log, open(os.path.join(OUTDIR, "lsweep.json"), "w"), indent=2)

    # best code vs [[5,1,3]], escalate shots until CIs separate around the 2x
    best_ler, best_L, best_npz, cs_best = best
    cs_513 = load_code("five_qudit_d3")["code_states"]
    log["best"] = {"L": best_L, "npz": os.path.relpath(best_npz, REPO), "ler_300k": best_ler}
    print(f"\n=== best code: L={best_L} LER={best_ler:.4e}; re-measuring vs [[5,1,3]] ===",
          flush=True)
    for n_shots in (300_000, 1_000_000):
        a_ler, a_k, a_ci = ler_at(cs_best, single, n_shots, seed=7)
        b_ler, b_k, b_ci = ler_at(cs_513, single, n_shots, seed=7)
        ratio = b_ler / a_ler if a_ler > 0 else float("inf")
        sep = a_ci[1] < b_ci[0]
        cell = {"n_shots": n_shots,
                "VarQEC_bestL": {"ler": a_ler, "wilson95": list(a_ci)},
                "[[5,1,3]]": {"ler": b_ler, "wilson95": list(b_ci)},
                "ratio": round(ratio, 2), "separated": bool(sep),
                "varqec_at_or_below_2x_target": bool(a_ci[1] <= TARGET_2X)}
        log.setdefault("headline", []).append(cell)
        json.dump(log, open(os.path.join(OUTDIR, "lsweep.json"), "w"), indent=2)
        print(f"  {n_shots:,}: VarQEC {a_ler:.4e} [{a_ci[0]:.3e},{a_ci[1]:.3e}] vs "
              f"[[5,1,3]] {b_ler:.4e} -> {ratio:.2f}x, sep={sep}", flush=True)
        # stop at 300k unless the 2x question is still ambiguous (VarQEC CI straddles target)
        if not (a_ci[0] < TARGET_2X < a_ci[1]):
            break

    recovered = best_ler <= TARGET_2X
    log["verdict"] = (
        f"RECOVERED ~2x: best n=5 code (L={best_L}) reaches {best_ler:.3e} <= {TARGET_2X:.1e}"
        if recovered else
        f"FLOOR ~{best_ler:.1e}: deeper/best-of-{SEEDS} does NOT beat {TARGET_2X:.1e}; the "
        f"1.46x is the real best-practice result and the published 3.67e-4 was a favourable "
        f"seed/connectivity. CPU-settled, not GPU-blocked.")
    json.dump(log, open(os.path.join(OUTDIR, "lsweep.json"), "w"), indent=2)
    print("\n" + log["verdict"] + "\nDONE", flush=True)


if __name__ == "__main__":
    main()
