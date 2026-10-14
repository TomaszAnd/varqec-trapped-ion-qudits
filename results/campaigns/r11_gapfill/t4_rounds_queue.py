#!/usr/bin/env python3
"""Repeated-round lambda data over the FULL s grid (Fig. 5).

Extends r9_campaign/t3_rounds.py from s in {0.25, 0.5} to the full
strength grid {0.125, 0.25, 0.5, 1.0, 1.5}: per-round F_e / rate for the
n=7 trio (d2/d3/d4) + the n=5 central code + [[5,1,3]]_Z3 through the
exact repeated-round PROJECTIVE-MAP instrument (round = v2 dose at s,
then the instrument as a CPTP channel; identical machinery to
r9_campaign/t3_rounds.py, dense-gemm application validated there to
4e-16 and re-checked here at startup). R = 12 rounds or F_e < 0.5.

Existing (code, s) cells are imported verbatim from r9_campaign
t3_results.json with provenance recorded -- no recompute of committed
numbers.

OPTIONAL PETZ ROWS: round = full-dose Petz recovery of the round dose,
R(X) = P_C N^dag(S^{-1/2} X S^{-1/2}) P_C, applied per round.
n=5 references at s in {0.25, 0.5}; n=7 trio at s in {0.5, 1.0}
(s=0.25 at n=7 needs ~12 rounds x 3.5 min x 3 codes -- omitted for cost,
stated). Petz recovery is trace-nonincreasing on support leak; the leak
is scored unrecovered (trace_block00 recorded per round).

Round-1 rates ARE the single-round rates through the same instrument
(reproduce r9_campaign t1/t2b cells exactly where they overlap --
checked in the analysis step); the single-round inset needs no separate
run.

Writes t4_rounds_results.json (resumable at (code, instrument, s)
granularity).
Launch: cd results/campaigns/r11_gapfill
        nohup nice -n 19 python3 t4_rounds_queue.py >> t4_rounds.log 2>&1 &
"""
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

import numpy as np

import src.jax_backend  # noqa: F401  (x64 first)
from src.catalog import five_qudit_code_states
from src.decoders._common import apply_local_np, encoder_forward
from src.decoders.exact_channel import apply_dose, build_dose_steps
from src.decoders.projective_decode import (
    apply_instrument_dense, build_projective_instrument,
    build_petz_exact_dose, favg_from_fe,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "t4_rounds_results.json")
STATUS = os.path.join(HERE, "t4_rounds_status.json")
R9_T3 = os.path.join(REPO, "results", "campaigns", "r9_campaign",
                     "t3_results.json")
BR = os.path.join(REPO, "results", "campaigns")

SIGMA_P2, SIGMA_A2_CAL = 0.073, 0.047
D = 3
S_ALL = (0.125, 0.25, 0.5, 1.0, 1.5)
R_MAX = 12
FE_FLOOR = 0.5

CODES = [
    ("headline", 5, os.path.join(BR, "r1_campaign", "t1_codes",
                                 "A_L2_s100.npz")),
    ("five_one_three_Z3", 5, None),
    ("d2", 7, os.path.join(BR, "r3_campaign", "t2c_codes", "d2_L2_s100.npz")),
    ("d3", 7, os.path.join(BR, "r3_campaign", "t2c_codes", "d3_L2_s100.npz")),
    ("d4", 7, os.path.join(BR, "r3_campaign", "t2c_codes", "d4_L2_s102.npz")),
]
PETZ_S = {5: (0.25, 0.5), 7: (0.5, 1.0)}


def set_status(**kw):
    kw["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(kw, open(STATUS, "w"), indent=2)


def instrument_channel_ops(inst):
    Ball = inst['Ball']
    d, n = inst['d'], inst['n_qudit']
    R = Ball.shape[1]
    W = np.empty_like(Ball)
    mask = np.zeros((R, R))
    for s_ in inst['sectors']:
        sl = slice(s_['offset'], s_['offset'] + s_['dim'])
        for c in range(s_['dim']):
            W[:, s_['offset'] + c] = apply_local_np(
                np.ascontiguousarray(Ball[:, s_['offset'] + c]),
                s_['C_dag'], s_['wires'], s_['inv'], d, n)
        mask[sl, sl] = 1.0
    return Ball, W, mask


def apply_instrument_channel(sigma, Ball, W, mask):
    Y = Ball.conj().T @ sigma
    G = Y @ Ball
    out = W @ ((G * mask) @ W.conj().T)
    SB = sigma @ Ball
    out += sigma - Ball @ Y - SB @ Ball.conj().T + Ball @ (G @ Ball.conj().T)
    return out


def consistency_check():
    s = 0.5
    eta, sa2 = float(np.exp(-s * SIGMA_P2)), s * SIGMA_A2_CAL
    cs = five_qudit_code_states(D)
    inst = build_projective_instrument(cs, D, 5, eta, sigma_a2=sa2)
    Ball, W, mask = instrument_channel_ops(inst)
    rng = np.random.default_rng(3)
    A = rng.standard_normal((243, 243)) + 1j * rng.standard_normal((243, 243))
    sigma = A @ A.conj().T
    sigma /= np.trace(sigma).real
    ref = apply_instrument_dense(inst, sigma)
    new = apply_instrument_channel(sigma, Ball, W, mask)
    dev = float(np.max(np.abs(ref - new)))
    print(f"consistency check: max dev {dev:.2e}", flush=True)
    assert dev < 1e-10, dev
    return dev


def load_code(name, n, path):
    conns = [[i, j] for i in range(n) for j in range(i + 1, n)]
    if path is None:
        cs = five_qudit_code_states(D)
    else:
        cs, d_, n_, K_ = encoder_forward(path, conns)
        assert (d_, n_) == (D, n)
    return np.asarray(cs)


def run_rounds(cs, n, steps, round_fn):
    """Common round loop: blocks -> per-round F_e list."""
    K = cs.shape[0]
    blocks = {}
    for i in range(K):
        for j in range(i, K):
            blocks[(i, j)] = np.outer(cs[i], cs[j].conj())
    rounds = []
    for r in range(1, R_MAX + 1):
        t0 = time.time()
        f = np.zeros((K, K), dtype=complex)
        for i in range(K):
            for j in range(i, K):
                sig = round_fn(blocks[(i, j)])
                blocks[(i, j)] = sig
                f[i, j] = cs[i].conj() @ (sig @ cs[j])
                if j > i:
                    f[j, i] = np.conj(f[i, j])
        F_e = float(np.real(f.sum()) / K ** 2)
        F_avg = favg_from_fe(F_e, K)
        tr = float(np.real(np.trace(blocks[(0, 0)])))
        rounds.append({"round": r, "F_e": F_e, "F_avg": F_avg,
                       "rate_1mFavg": 1.0 - F_avg, "trace_block00": tr,
                       "wall_s": round(time.time() - t0, 1)})
        print(f"    round {r}: F_e={F_e:.6f} rate={1.0-F_avg:.4e} "
              f"({rounds[-1]['wall_s']}s)", flush=True)
        if F_e < FE_FLOOR:
            break
    return rounds


def unit_projective(name, n, cs, s, res):
    key = f"{name}@s{s:.2f}"
    if key in res["cells"]:
        return
    set_status(state="running", unit="proj:" + key, done=len(res["cells"]))
    eta, sa2 = float(np.exp(-s * SIGMA_P2)), s * SIGMA_A2_CAL
    gp = [(i, j, 0, 1) for i in range(n) for j in range(i + 1, n)]
    t0 = time.time()
    steps = build_dose_steps(n, D, gp, eta, sigma_a2=sa2)
    inst = build_projective_instrument(cs, D, n, eta, sigma_a2=sa2)
    Ball, W, mask = instrument_channel_ops(inst)
    print(f"{key}: setup {time.time()-t0:.0f}s "
          f"({inst['report']['n_sectors']} sectors)", flush=True)

    def round_fn(sig):
        return apply_instrument_channel(apply_dose(sig, steps, D, n),
                                        Ball, W, mask)

    rounds = run_rounds(cs, n, steps, round_fn)
    res["cells"][key] = {
        "code": name, "n": n, "s": s, "eta": eta, "sigma_a2": sa2,
        "instrument": "projective-MAP",
        "n_sectors": inst['report']['n_sectors'],
        "claimed_dim_R": inst['report']['claimed_dim_R'],
        "rounds": rounds}
    json.dump(res, open(OUT, "w"), indent=2)


def unit_petz(name, n, cs, s, res):
    key = f"{name}@s{s:.2f}"
    if key in res["petz_cells"]:
        return
    set_status(state="running", unit="petz:" + key, done=len(res["cells"]))
    eta, sa2 = float(np.exp(-s * SIGMA_P2)), s * SIGMA_A2_CAL
    gp = [(i, j, 0, 1) for i in range(n) for j in range(i + 1, n)]
    t0 = time.time()
    steps = build_dose_steps(n, D, gp, eta, sigma_a2=sa2)
    petz = build_petz_exact_dose(cs, steps, D, n)
    S_half_inv = petz["inv_sqrt"]
    P_C = cs.T @ cs.conj()
    print(f"petz {key}: setup {time.time()-t0:.0f}s "
          f"(support {petz['support_rank']}/{D**n})", flush=True)

    def round_fn(sig):
        X = apply_dose(sig, steps, D, n)
        Y = S_half_inv @ X @ S_half_inv
        Z = apply_dose(Y, steps, D, n, adjoint=True)
        return P_C @ Z @ P_C

    rounds = run_rounds(cs, n, steps, round_fn)
    res["petz_cells"][key] = {
        "code": name, "n": n, "s": s, "eta": eta, "sigma_a2": sa2,
        "instrument": "full-dose Petz (R(X)=P_C N^dag(S^-1/2 X S^-1/2) P_C; "
                      "leak scored unrecovered)",
        "support_rank": petz["support_rank"],
        "rounds": rounds}
    json.dump(res, open(OUT, "w"), indent=2)


def main():
    res = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            res = json.load(f)
    res.setdefault("meta", {
        "d": D, "s_grid": list(S_ALL), "R_max": R_MAX, "fe_floor": FE_FLOOR,
        "round": "rho -> instrument(dose_v2(rho)), exact",
        "convention": "eta(s)=e^{-s*0.073}, sigma_a2(s)=s*0.047",
        "metric": "per-round F_e / F_avg / rate=1-F_avg (channel infidelity)",
        "single_round_note": "round-1 rate = the single-round rate through "
                             "the same instrument (matches r9 t1/t2b cells)",
        "petz_scope": "n=5 refs at s in {0.25,0.5}; n=7 trio at s in "
                      "{0.5,1.0}; n=7 s=0.25 omitted for cost (stated)"})
    res.setdefault("cells", {})
    res.setdefault("petz_cells", {})

    # import committed r9 cells verbatim (provenance recorded)
    with open(R9_T3) as f:
        r9 = json.load(f)
    for key, cell in r9["cells"].items():
        if key not in res["cells"]:
            cell = dict(cell)
            cell["instrument"] = "projective-MAP"
            cell["reused_from"] = "r9_campaign/t3_results.json"
            res["cells"][key] = cell
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"imported {len(r9['cells'])} r9 cells", flush=True)

    res["meta"]["consistency_dev"] = consistency_check()

    # cheap n=5 first, then n=7 projective, then Petz rows
    for name, n, path in sorted(CODES, key=lambda c: c[1]):
        cs = load_code(name, n, path)
        for s in S_ALL:
            unit_projective(name, n, cs, s, res)
    for name, n, path in sorted(CODES, key=lambda c: c[1]):
        cs = load_code(name, n, path)
        for s in PETZ_S[n]:
            unit_petz(name, n, cs, s, res)
    set_status(state="done", done=len(res["cells"]),
               petz_done=len(res["petz_cells"]))
    print("done", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        set_status(state="FAILED", error=str(e))
        raise
