#!/usr/bin/env python3
"""Stopping-rule comparison (Table V): total op-EVs to a CERTIFIED stop.

Setting: UNWEIGHTED hardware-closure-basis KL, n=5, L=16 a2a MS, i.e.
((5,3,3))_3 in the repo's ((n,K,d))_q notation (= [[5,1,3]] in [[n,k,d]]
notation with k=1 logical qutrit, K=q=3 codewords); a literal K=1 in the
repo loss is degenerate (Term 1 has no i<j pairs and Term 2's codeword
variance over one codeword is 0, so the loss is identically zero).
A K-EXPLICIT rebuild (fig6_panel_train machinery, K=3) is verified against
build_race's loss at startup and recorded in the JSON header.

Certification threshold: the paper's 1e-6 full-batch criterion is
UNATTAINABLE in this setting — the L=16 hw-basis capacity floor is
7.0-8.5e-3 across all 8 loss_ler_diag/t1g seeds (and 3000-step deep runs
never pass 7.2e-5 even at L=28 on any basis) — so at 1e-6 every policy
would report certified=NO with identical (maximal) budgets and the
comparison would be vacuous. Policies are therefore compared at an
ATTAINABLE tier tau chosen just above the measured capacity floor (U arm:
1e-2; W arm: 1e-4 vs the ((6,3,3))_3 weighted ms floor 4.35e-5), and every
step's sampled loss is also tested against 1e-6 (dips counted) to document
on-record that the paper-tier trigger never fires here.

Policies (4 seeds each, metric = TOTAL op-EVs incl. confirmation costs):
  fixed_final : fixed 3000 steps + ONE final full-batch confirm on the
                best-sampled theta (the standard protocol);
  naive       : full-batch confirm EVERY step the sampled loss dips below
                tau (the runaway candidate; confirms counted);
  hysteresis  : confirm only after m=5 CONSECUTIVE sampled dips AND >=50
                steps since the last confirm;
  growing     : start f=0.05, double f when the windowed sampled loss
                plateaus (<5% relative improvement over 100 steps); once
                f=1 every step is exact -> accept when the (full-batch)
                loss < tau. No separate confirms.
Grid: {fixed_final, naive, hysteresis} x f in {0.1, 0.2, 0.35} x sampler
{stratified, importance} + growing x {stratified, importance} + the
full-batch baseline (fixed_final @ f=1). W arm: the 4 policies at
importance@0.2 only. min-viable f is read off the results (smallest f with
4/4 certified yield), not run as a separate probe.

Per cell x seed: steps_to_stop, realized ops/step, confirm count (+false
confirms), train/confirm/total op-EVs, certified y/n, achieved full loss,
sampled dips below 1e-6, wall seconds. Writes b_results.json (incremental,
resume per unit). Feeds Table V.
Launch detached: nohup nice -n 19 python3 b_policy_queue.py >> b.log 2>&1 &
"""
import os, sys, json, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import src.jax_backend  # x64 first (must precede any jnp array creation)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'b_results.json')
STATUS = os.path.join(HERE, 'b_status.json')
MAX_STEPS, LR, SEEDS = 3000, 0.05, [0, 1, 2, 3]
PAPER_TIER = 1e-6


def set_status(**kw):
    kw['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    json.dump(kw, open(STATUS, 'w'), indent=2)


def build_arm(arm):
    from src.seed_race import build_race
    if arm == 'U':
        (vg, fl, gs, ppl, _, fw, sgw) = build_race(3, 5, 3, 16, noise='hw')
    else:
        (vg, fl, gs, ppl, _, fw, sgw) = build_race(3, 6, 3, 16, weighted=True)
    return dict(val_grad=vg, full_loss_fn=fl, group_sizes=gs, ppl=ppl,
                full_weights=fw, sgw=sgw, total_ops=int(sum(gs)))


def verify_k_explicit(u_arm):
    """K-explicit (fig6_panel_train machinery) rebuild == build_race, K=3."""
    import jax, jax.numpy as jnp
    from src.errors import ErrorModel
    from src.jax_backend import create_jax_encoder, build_varqec_loss_weighted
    n, d = 5, 3
    conns = [[i, j] for i in range(n) for j in range(i + 1, n)]
    enc, _, ppl = create_jax_encoder(n, d, connections=conns, use_scan=True)
    grouped = ErrorModel(d=d, n_qudit=n, distance=3, closed=True,
                         basis=None).build_grouped(verbose=False)
    loss_k = build_varqec_loss_weighted(encoder_fn=enc, K=3, d=d, n_qudit=n,
                                        distance=3, E_det_grouped=grouped)
    ones = tuple(jnp.ones((g['matrices'].shape[0],), dtype=jnp.float64)
                 for g in grouped)
    theta = jax.random.uniform(jax.random.PRNGKey(7), (16, ppl),
                               minval=0.0, maxval=2 * np.pi)
    a = float(loss_k(theta, ones))
    b = float(u_arm['full_loss_fn'](theta))
    rel = abs(a - b) / max(abs(b), 1e-300)
    return {'k_explicit_loss': a, 'build_race_loss': b, 'rel_diff': rel,
            'agree_1e-12': rel < 1e-12}


def draw(sampler, f, arm, rng):
    from src.jax_backend import stratified_weights
    from src.sampling.stratified_importance import (
        make_stratified_importance_weights)
    if sampler == 'full':
        return arm['full_weights'], arm['total_ops']
    if sampler == 'stratified':
        w = stratified_weights(arm['group_sizes'], f, rng)
    else:
        budget = max(1, int(round(f * arm['total_ops'])))
        w = make_stratified_importance_weights(arm['sgw'], budget, rng)
    return w, int(sum(int(np.count_nonzero(np.asarray(x))) for x in w))


def run_unit(arm, policy, sampler, f, seed, tau):
    import jax, jax.numpy as jnp, optax
    t0 = time.time()
    theta = jax.random.uniform(
        jax.random.split(jax.random.PRNGKey(seed))[0],
        (16, arm['ppl']), minval=0.0, maxval=2 * np.pi)
    rng = np.random.default_rng(seed)
    opt = optax.inject_hyperparams(optax.adam)(learning_rate=LR)
    st = opt.init(theta); stage = 0
    total_ops = arm['total_ops']
    train_ev = confirm_ev = 0
    n_confirms = n_false = consec = 0
    dips_paper_tier = 0
    last_confirm_step = -10**9
    best_sampled, best_theta = np.inf, theta
    certified, stop_step, full_at_stop = False, None, None
    cur_f = 0.05 if policy == 'growing' else f
    win, prev_win_mean = [], None

    for step in range(MAX_STEPS):
        eff_sampler = ('full' if (policy == 'growing' and cur_f >= 1.0)
                       else sampler)
        w, c = draw(eff_sampler, cur_f, arm, rng)
        train_ev += c
        lv, g = arm['val_grad'](theta, w)
        lvf = float(lv)
        if lvf < best_sampled:
            best_sampled, best_theta = lvf, theta
        if lvf < PAPER_TIER:
            dips_paper_tier += 1

        if policy == 'naive' and lvf < tau:
            fv = float(arm['full_loss_fn'](theta)); confirm_ev += total_ops
            n_confirms += 1
            if fv < tau:
                certified, stop_step, full_at_stop = True, step, fv
                break
            n_false += 1
        elif policy == 'hysteresis':
            consec = consec + 1 if lvf < tau else 0
            if consec >= 5 and (step - last_confirm_step) >= 50:
                fv = float(arm['full_loss_fn'](theta)); confirm_ev += total_ops
                n_confirms += 1; last_confirm_step = step; consec = 0
                if fv < tau:
                    certified, stop_step, full_at_stop = True, step, fv
                    break
                n_false += 1
        elif policy == 'growing':
            win.append(lvf)
            if cur_f >= 1.0 and lvf < tau:
                certified, stop_step, full_at_stop = True, step, lvf
                break
            if len(win) == 100:
                m = float(np.mean(win))
                if (prev_win_mean is not None and cur_f < 1.0
                        and (prev_win_mean - m) / max(prev_win_mean, 1e-300)
                        < 0.05):
                    cur_f = min(1.0, 2 * cur_f)
                prev_win_mean = m; win = []

        upd, st = opt.update(g, st, theta)
        theta = optax.apply_updates(theta, upd)
        if stage == 0 and lvf < 0.5:
            st.hyperparams['learning_rate'] = LR / 5; stage = 1
        if stage <= 1 and lvf < 1e-3:
            st.hyperparams['learning_rate'] = LR / 50; stage = 2

    if not certified:
        # fixed_final's single confirm; fallback final confirm for the rest
        fv = float(arm['full_loss_fn'](best_theta)); confirm_ev += total_ops
        n_confirms += 1
        certified = fv < tau
        stop_step, full_at_stop = MAX_STEPS, fv
        if not certified:
            n_false += 1
    steps_run = stop_step if stop_step is not None else MAX_STEPS
    return {'policy': policy, 'sampler': sampler, 'f': f, 'seed': seed,
            'tau': tau, 'certified': certified,
            'steps_to_stop': int(steps_run),
            'realized_ops_per_step': round(train_ev / max(1, min(
                steps_run + 1, MAX_STEPS)), 2),
            'n_confirms': n_confirms, 'n_false_confirms': n_false,
            'train_op_evs': int(train_ev), 'confirm_op_evs': int(confirm_ev),
            'total_op_evs': int(train_ev + confirm_ev),
            'full_loss_at_stop': full_at_stop,
            'sampled_dips_below_1e-6': int(dips_paper_tier),
            'wall_s': round(time.time() - t0, 1)}


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {'units': {}}
    print('[b] building arms ...', flush=True)
    arms = {'U': build_arm('U'), 'W': build_arm('W')}
    res['header'] = {
        'U': {'setting': '((5,3,3))_3, n=5, L=16, all-to-all, MS, '
                         'unweighted completed channel basis',
              'total_ops': arms['U']['total_ops'], 'tau': 1e-2,
              'tau_rationale': 'attainable tier just above the L=16 hw '
                               'capacity floor 7.0-8.5e-3 (t1g, 8 seeds); '
                               'paper tier 1e-6 unattainable here (dips '
                               'counted per unit)'},
        'W': {'setting': '((6,3,3))_3 weighted Meth KL (Sec IX line)',
              'total_ops': arms['W']['total_ops'], 'tau': 1e-4,
              'tau_rationale': 'above the study ms floor 4.35e-5'},
        'k_explicit_verification': verify_k_explicit(arms['U']),
        'max_steps': MAX_STEPS, 'paper_tier': PAPER_TIER}
    json.dump(res, open(OUT, 'w'), indent=2)
    print('[b] K-explicit check:',
          res['header']['k_explicit_verification'], flush=True)

    cells = []
    for pol in ('fixed_final', 'naive', 'hysteresis'):
        for f in (0.1, 0.2, 0.35):
            for smp in ('stratified', 'importance'):
                cells.append(('U', pol, smp, f, 1e-2))
    cells.append(('U', 'growing', 'stratified', None, 1e-2))
    cells.append(('U', 'growing', 'importance', None, 1e-2))
    cells.append(('U', 'fixed_final', 'full', 1.0, 1e-2))
    for pol in ('fixed_final', 'naive', 'hysteresis'):
        cells.append(('W', pol, 'importance', 0.2, 1e-4))
    cells.append(('W', 'growing', 'importance', None, 1e-4))

    units = [(a, p, s, f, tau, seed) for (a, p, s, f, tau) in cells
             for seed in SEEDS]
    for i, (a, p, s, f, tau, seed) in enumerate(units):
        key = f'{a}|{p}|{s}|f{f}|s{seed}'
        if key in res['units']:
            continue
        set_status(state='running', unit=key, done=len(res['units']),
                   total=len(units))
        row = run_unit(arms[a], p, s, f, seed, tau)
        row['arm'] = a
        res['units'][key] = row
        json.dump(res, open(OUT, 'w'), indent=2)
        print(f"[b] UNIT DONE {key}: cert={row['certified']} "
              f"steps={row['steps_to_stop']} confirms={row['n_confirms']} "
              f"(false {row['n_false_confirms']}) "
              f"total={row['total_op_evs']} full={row['full_loss_at_stop']:.3e} "
              f"[{row['wall_s']}s]", flush=True)

    set_status(state='done', done=len(res['units']), total=len(units))
    print('[b] QUEUE DONE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        set_status(state='FAILED', error=str(e))
        raise
