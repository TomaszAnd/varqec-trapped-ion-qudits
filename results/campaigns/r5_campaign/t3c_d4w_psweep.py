#!/usr/bin/env python3
"""Correction-protocol iid p-sweep for the WEIGHTED d=4 arm
(t2c_codes/d4_L2_s102.npz), same instrument/grid as r2_campaign/t1b_panel.py
(>=30 errors, escalate x2 from 3e5, cap 5e6, seed 2000). Makes the rebuilt
distance panel single-protocol (all three arms channel-weighted, MS 84).
Writes t3c_d4w_psweep.json."""
import os, sys, json, math, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0,REPO); os.chdir(REPO)
import src.jax_backend
import numpy as np
from src.decoders._common import encoder_forward
from src.simulation import simulate_ler_with_correction_factored
from src.errors import make_hardware_noise_fn, qudit_hardware_error_basis

HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(HERE,'t3c_d4w_psweep.json')
N,D=7,3
CONNS=[[i,j] for i in range(N) for j in range(i+1,N)]
S=[np.asarray(E,complex) for E in qudit_hardware_error_basis(D)]
def wilson(k,n,z=1.96):
    ph=k/n; dn=1+z*z/n
    c=(ph+z*z/(2*n))/dn
    h=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/dn
    return (max(0.0,c-h),c+h)
res=json.load(open(OUT)) if os.path.exists(OUT) else {'npz':'r3_campaign/t2c_codes/d4_L2_s102.npz','curve':{}}
cs=np.asarray(encoder_forward('results/campaigns/r3_campaign/t2c_codes/d4_L2_s102.npz',CONNS)[0])
for p in (0.02,0.05,0.08,0.12,0.15):
    key=str(p)
    if key in res['curve']: continue
    t0=time.time(); shots=300_000
    while True:
        rr=simulate_ler_with_correction_factored(cs,make_hardware_noise_fn(D,N,p),S,N,D,n_shots=shots,seed=2000)
        k=int(round(float(rr['logical_error_rate'])*shots))
        if k>=30 or shots>=5_000_000: break
        shots=min(2*shots,5_000_000)
    res['curve'][key]={'p':p,'k':k,'n':shots,'ler':k/shots,'wilson95':wilson(k,shots),
                       'resolved':k>=30,'below_detection':k==0,
                       'wall_s':round(time.time()-t0,1)}
    json.dump(res,open(OUT,'w'),indent=2)
    print(f"[t3c] p={p}: {k}/{shots} [{res['curve'][key]['wall_s']}s]",flush=True)
print('[t3c] DONE',flush=True)
