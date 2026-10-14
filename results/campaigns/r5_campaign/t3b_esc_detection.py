#!/usr/bin/env python3
"""Detection-probe escalation: w2/w3 conditioned detection acceptance at 1M
shots for the ((7,3,d))_3 trio -- does the certificate ladder separate at
the 1e-6 level? Writes t3b_esc_detection.json."""
import os, sys, json, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0,REPO); sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
import src.jax_backend
import numpy as np
from src.decoders._common import encoder_forward
from src.errors import qudit_hardware_error_basis
from t23_queue import detection_postselect, wk_noise, wilson, TRIO, CONNS7, N7, D7

OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'t3b_esc_detection.json')
res=json.load(open(OUT)) if os.path.exists(OUT) else {}
S=[np.asarray(E,complex) for E in qudit_hardware_error_basis(D7)]
for lab,path in TRIO.items():
    cs=np.asarray(encoder_forward(path,CONNS7)[0])
    for w in (2,3):
        key=f'{lab}_w{w}_1M'
        if key in res: continue
        t0=time.time()
        r=detection_postselect(cs, wk_noise(N7,D7,w,S), 1_000_000, seed=700+w)
        res[key]={'accepted':r['accepted'],'n':r['n_shots'],
                  'acceptance':r['acceptance'],
                  'acceptance_wilson_ub': wilson(r['accepted'],r['n_shots'])[1],
                  'undet_errors':r['undet_errors'],
                  'wall_s':round(time.time()-t0,1)}
        json.dump(res,open(OUT,'w'),indent=1)
        print(f"[esc] {key}: accepted={r['accepted']}/1M "
              f"undet={r['undet_errors']} [{res[key]['wall_s']}s]",flush=True)
print('[esc] DONE',flush=True)
