#!/usr/bin/env python3
"""Random-encoder control baseline.

8 untrained random-theta encoders on the exact central-code ansatz
(((5,3,3))_3, L=2, a2a, MS; theta ~ U[0,2pi), the training initializer),
measured through the identical benchmark: w1/w2/w3 conditioned probes +
iid LER @ p=0.05 (300k shots, escalated to >=30 errors, cap 2M).
Splits the central code's gain into training vs physical-resource parts.
Writes t4_random_baseline.json (supporting control; not a paper figure).
"""
import os, sys, json, math, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0,REPO); os.chdir(REPO)
import src.jax_backend
import numpy as np
import jax
from src.seed_race import build_race
from src.simulation import simulate_ler_with_correction_factored, _apply_factored_op
from src.errors import make_hardware_noise_fn, qudit_hardware_error_basis

HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(HERE,'t4_random_baseline.json')
D,N,L=3,5,2
S=[np.asarray(E,complex) for E in qudit_hardware_error_basis(D)]
def wk_noise(w):
    def fn(state,rng):
        qs=rng.choice(N,size=w,replace=False)
        s=state
        for q in qs:
            s=_apply_factored_op(s,S[rng.integers(len(S))],int(q),N,D)
        nrm=np.linalg.norm(s)
        return s/nrm if nrm>1e-15 else s
    return fn
def wilson(k,n,z=1.96):
    ph=k/n; dn=1+z*z/n
    c=(ph+z*z/(2*n))/dn
    h=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/dn
    return (max(0.0,c-h),c+h)

# encoder via build_race machinery (same ansatz path as training)
from src.jax_backend import create_jax_encoder
conns=[[i,j] for i in range(N) for j in range(i+1,N)]
enc,_,ppl=create_jax_encoder(N,D,connections=conns,use_scan=False)
res=json.load(open(OUT)) if os.path.exists(OUT) else {'units':{}}
for seed in range(500,508):
    key=f'rand_s{seed}'
    if key in res['units']: continue
    t0=time.time()
    k0=jax.random.PRNGKey(seed)
    ik,_=jax.random.split(k0)
    theta=np.asarray(jax.random.uniform(ik,(L,ppl),minval=0.0,maxval=2*np.pi))
    cs=np.stack([np.asarray(enc(theta, k)).reshape(-1) for k in range(D)])
    # orthonormality check (isometry by construction)
    g=np.conj(cs)@cs.T
    assert np.allclose(g,np.eye(D),atol=1e-10)
    row={'seed':seed}
    for w,sd,shots in ((1,1001,200_000),(2,1002,100_000),(3,1003,100_000)):
        rr=simulate_ler_with_correction_factored(cs,wk_noise(w),S,N,D,n_shots=shots,seed=sd)
        k=int(round(float(rr['logical_error_rate'])*shots))
        row[f'ler_w{w}']={'k':k,'n':shots,'ler':k/shots,'wilson95':wilson(k,shots)}
    shots=300_000
    while True:
        rr=simulate_ler_with_correction_factored(cs,make_hardware_noise_fn(D,N,0.05),S,N,D,n_shots=shots,seed=2000)
        k=int(round(float(rr['logical_error_rate'])*shots))
        if k>=30 or shots>=2_000_000: break
        shots=min(2*shots,2_000_000)
    row['ler_iid_p05']={'k':k,'n':shots,'ler':k/shots,'wilson95':wilson(k,shots)}
    row['wall_s']=round(time.time()-t0,1)
    res['units'][key]=row
    json.dump(res,open(OUT,'w'),indent=2)
    print(f"[rand] {key}: w1={row['ler_w1']['k']}/200k w2={row['ler_w2']['k']}/100k "
          f"iid={row['ler_iid_p05']['ler']:.3e} [{row['wall_s']}s]",flush=True)
iids=[res['units'][f'rand_s{s}']['ler_iid_p05']['ler'] for s in range(500,508)]
res['summary']={'iid_median':float(np.median(iids)),'iid_range':[min(iids),max(iids)],
                'headline_ref':3.42e-4,'benchmark_ref':8.49e-4}
json.dump(res,open(OUT,'w'),indent=2)
print('[rand] DONE median',res['summary']['iid_median'],flush=True)
