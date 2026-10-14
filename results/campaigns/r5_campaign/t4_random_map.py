#!/usr/bin/env python3
"""Correlated-channel MAP arm of the random-encoder control.
Same instrument as weighted_race_test.map_channel_ler, direct code states.
eta=0.8642, 40k shots (matches the central code's t2t4 cell: 161/40k =
4.03e-3). Writes t4_random_map.json."""
import os, sys, json, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,'scripts'))
os.chdir(REPO)
import src.jax_backend
import numpy as np
import jax
from src.jax_backend import create_jax_encoder
from src.errors import ErrorModel
from src.decoders.weighted_map import (build_weighted_correction_set,
                                       simulate_ler_with_weighted_map)
from src.simulation import make_correlated_dephasing_noise_fn
from src.meth_weights import compute_meth_group_weights
from benchmark_ler_meth_kraus import build_gate_pairs
from benchmark_ler_meth_pauli import bootstrap_ler_ci

HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(HERE,'t4_random_baseline.json')
D,N,L=3,5,2
ETA=0.8642
SHOTS=40_000
conns=[[i,j] for i in range(N) for j in range(i+1,N)]
enc,_,ppl=create_jax_encoder(N,D,connections=conns,use_scan=False)
E_full=ErrorModel(d=D,n_qudit=N,distance=3,closed=True).build_grouped(verbose=False)
_,prior=compute_meth_group_weights(D,N,distance=3,eta=ETA,n_max=2)
corr=build_weighted_correction_set(E_full,prior,N,D,max_weight=2)
gp=build_gate_pairs(N,conns)
nf=make_correlated_dephasing_noise_fn(n_qudits=N,d=D,gate_pairs=gp,eta=ETA,
                                      n_max=5,noise_model='physical')
res=json.load(open(OUT))
for seed in range(500,508):
    key=f'rand_s{seed}'
    if 'map_eta08642' in res['units'][key]: continue
    t0=time.time()
    k0=jax.random.PRNGKey(seed)
    ik,_=jax.random.split(k0)
    theta=np.asarray(jax.random.uniform(ik,(L,ppl),minval=0.0,maxval=2*np.pi))
    cs=np.stack([np.asarray(enc(theta,k)).reshape(-1) for k in range(D)])
    out=simulate_ler_with_weighted_map(cs,nf,corr,N,D,n_shots=SHOTS,seed=7)
    ler,lo,hi=bootstrap_ler_ci(out,rng=np.random.default_rng(7))
    res['units'][key]['map_eta08642']={'ler':float(ler),'boot95':[float(lo),float(hi)],
                                       'n':SHOTS,'wall_s':round(time.time()-t0,1)}
    json.dump(res,open(OUT,'w'),indent=2)
    print(f"[randmap] {key}: map={ler:.3e} [{lo:.1e},{hi:.1e}] "
          f"[{res['units'][key]['map_eta08642']['wall_s']}s]",flush=True)
maps=[res['units'][f'rand_s{s}']['map_eta08642']['ler'] for s in range(500,508)]
res['summary']['map_median']=float(np.median(maps))
res['summary']['map_range']=[min(maps),max(maps)]
res['summary']['map_headline_ref_eta08642']=4.03e-3
json.dump(res,open(OUT,'w'),indent=2)
print('[randmap] DONE median',res['summary']['map_median'],flush=True)
