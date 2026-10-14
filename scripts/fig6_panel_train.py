#!/usr/bin/env python3
"""Fig-6 one-parameter-panel codes: train the missing K/q/distance points.

Four one-parameter panels (register size, K, q, distance), each
varying exactly one of n, K, q, dist. The n-panel uses the existing campaign
codes; this script trains the codes the other three panels need:

  K-panel    ((5,2,3))_3   K=2, q=3, dist=3   (K=3 point = campaign d3_n5)
  dist-panel ((5,3,2))_3   dist=2             (dist=3 point = campaign d3_n5)
  q-panel    ((5,3,3))_4   q=4, K=3           (q=3 point = campaign d3_n5)
             ((5,3,3))_5   q=5, K=3

K != q is not expressible through src.seed_race.build_race (it hard-wires
K=d), so this trainer drives build_varqec_loss_weighted with K explicit while
mirroring the campaign protocol EXACTLY (campaign_train.py -> run_race with
f_campaign=1.0 full-batch): L=16, all-to-all, XX/YY MS, --noise full basis,
corrected Gaussian Meth channel semantics untouched, best-of-3 seeds x 2000
steps, Adam lr=0.05 with the three-stage schedule, theta init
uniform(0, 2pi) from split(PRNGKey(seed))[0]. target_loss=1e-9 like the
campaign (nothing certifies; achieved loss is reported honestly).

npz output is decoder-compatible (params + d + n_qudit + K + connections,
via src.loss.save_varqec_result) so src/decoders/_common.encoder_forward
rebuilds the code states exactly as the LER benchmark does.

Writes ONLY to results/campaigns/fig6_panels/ (or --outdir).
Usage: fig6_panel_train.py [--seeds 3] [--steps 2000] [--only TAG] [--outdir DIR]
"""
from __future__ import annotations
import os, sys, time, json, argparse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import src.jax_backend  # noqa: F401  x64 footgun: flip jax_enable_x64 FIRST
import numpy as np

# (tag, d, K, distance) — n=5, L=16, a2a, MS for every panel code.
CODES = [
    ("K2_d3_n5_dist3_16L", 3, 2, 3),
    ("d3_n5_dist2_16L",    3, 3, 2),
    ("d4K3_n5_dist3_16L",  4, 3, 3),
    ("d5K3_n5_dist3_16L",  5, 3, 3),
]
N, LAYERS, LR = 5, 16, 0.05


def train_code(tag, d, K, dist, seeds, steps, outdir):
    import jax, jax.numpy as jnp, optax
    from src.errors import ErrorModel
    from src.jax_backend import create_jax_encoder, build_varqec_loss_weighted
    from src.loss import save_varqec_result

    conns = [[i, j] for i in range(N) for j in range(i + 1, N)]
    enc, _, ppl = create_jax_encoder(N, d, connections=conns, use_scan=True)
    closed = dist >= 3
    grouped = ErrorModel(d=d, n_qudit=N, distance=dist, closed=closed,
                         basis="full").build_grouped(verbose=False)
    loss_w = build_varqec_loss_weighted(encoder_fn=enc, K=K, d=d, n_qudit=N,
                                        distance=dist, E_det_grouped=grouped)
    full_w = tuple(jnp.ones((g['matrices'].shape[0],), dtype=jnp.float64)
                   for g in grouped)
    val_grad = jax.jit(jax.value_and_grad(lambda th: loss_w(th, full_w)))

    best = None  # (full_loss, seed, theta, losses)
    for seed in range(seeds):
        init_key, _ = jax.random.split(jax.random.PRNGKey(seed))
        theta = jax.random.uniform(init_key, (LAYERS, ppl),
                                   minval=0.0, maxval=2 * np.pi)
        opt = optax.inject_hyperparams(optax.adam)(learning_rate=LR)
        st = opt.init(theta); lr_stage = 0
        losses = []
        best_here = (np.inf, theta)
        for step in range(steps):
            lv, g = val_grad(theta)
            lvf = float(lv); losses.append(lvf)
            if lvf < best_here[0]:
                best_here = (lvf, theta)
            upd, st = opt.update(g, st, theta)
            theta = optax.apply_updates(theta, upd)
            if lr_stage == 0 and lvf < 0.5:
                st.hyperparams['learning_rate'] = LR / 5; lr_stage = 1
            if lr_stage <= 1 and lvf < 1e-3:
                st.hyperparams['learning_rate'] = LR / 50; lr_stage = 2
            if step % 200 == 0:
                print(f"  [{tag}] seed {seed} step {step:5d} loss={lvf:.4e}",
                      flush=True)
        fl = float(loss_w(best_here[1], full_w))  # full-batch loss of best theta
        print(f"  [{tag}] seed {seed} best full loss {fl:.4e}", flush=True)
        if best is None or fl < best[0]:
            best = (fl, seed, np.array(best_here[1]), list(losses))

    out = os.path.join(outdir, tag + ".npz")
    save_varqec_result(
        out, best[2], best[3], f"native_d{d}", dist, LAYERS,
        metadata={"K": K, "n_qudit": N, "d": d, "seed": best[1],
                  "certified": False, "certified_full_loss": best[0],
                  "target_loss": 1e-9, "sample_frac": 1.0,
                  "noise": "full", "entangler": "ms",
                  "connectivity": "all-to-all", "num_seeds": seeds,
                  "backend": "jax", "sampler": "uniform",
                  "steps_to_target": len(best[3]),
                  "connections": np.asarray(conns, dtype=int)})
    return out, best[0], best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--only", default=None, help="train a single tag")
    ap.add_argument("--outdir", default=os.path.join(
        REPO, "results", "campaigns", "fig6_panels"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    codes = [c for c in CODES if args.only is None or c[0] == args.only]
    summary_path = os.path.join(args.outdir, "panel_train_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        summary = json.load(open(summary_path))
    for tag, d, K, dist in codes:
        t = time.time()
        print(f"\n[panel {tag}] ((n={N},K={K},dist={dist}))_{d} training ...",
              flush=True)
        try:
            out, fl, seed = train_code(tag, d, K, dist, args.seeds, args.steps,
                                       args.outdir)
            summary[tag] = {"best_full_loss": fl, "seed": seed,
                            "wall_s": round(time.time() - t, 1),
                            "npz": os.path.basename(out),
                            "protocol": f"best-of-{args.seeds} x {args.steps} "
                                        "full-batch, L=16 a2a MS, noise=full"}
            print(f"[panel {tag}] best_full_loss={fl:.4e} "
                  f"({summary[tag]['wall_s']}s)", flush=True)
        except Exception as e:
            summary[tag] = {"FAILED": str(e),
                            "wall_s": round(time.time() - t, 1)}
            print(f"[panel {tag}] FAILED: {e}", flush=True)
        json.dump(summary, open(summary_path, "w"), indent=2)
    print("\n[panel] DONE", flush=True)


if __name__ == "__main__":
    main()
