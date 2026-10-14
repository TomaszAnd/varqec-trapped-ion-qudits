"""Catalog of variational codes and analytical benchmark codes.

What this module implements
---------------------------
* ``five_qudit_code_states``: the analytical [[5,1,3]]_{Z_q} codewords of
  paper Eq. (13) (Chau 1997, Rains 1997), the stabilizer benchmark of
  Sec. V E.
* ``load_code`` / ``list_codes``: a unified loader that rebuilds the
  codewords of a variational code from its committed parameter file
  (``.npz``) through the encoder of ``src.encoder``, or returns the
  analytical codewords for the benchmark entry.

Registry entries map a name to (npz path relative to ``results/campaigns``,
encoder arguments).  The entry ``central_553_3`` is the variational
((5,3,3))_3 code of the central comparison (paper Sec. VI A; channel-
weighted protocol, L = 2, all-to-all connectivity, MS 40).

Connectivity: the ansatz entangles qudits along the connectivity graph.
Parameter files written by the optimisation driver store the edge list
under the key ``connections``; the loader uses it when present and
otherwise falls back to the ring graph.  All codes reported in the paper
use all-to-all connectivity.

Consumers: ``src.central_comparison``, ``src.stabilizer_encoder``,
``scripts/table3_enumerators.py``, ``scripts/fig6_panels.py``,
``scripts/benchmark_ler.py`` and the tests.
"""
import os
import numpy as np
from pennylane import numpy as pnp

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "campaigns")


def five_qudit_code_states(q=3):
    """
    Construct [[5,1,3]]_{Z_q} codewords analytically (Chau 1997, Rains 1997).

    The encoding isometry (paper Eq. 13):
      T_{a,k,l,m,n,p} = delta(a = k+l+m+n+p mod q) * (1/q^2) * omega^(kl+lm+mn+np+pk)

    where omega = exp(2*pi*i/q): one phase factor per edge of the five-cycle,
    restricted to the basis states whose level sum is the label a.

    Args:
        q: qudit dimension (default 3 for qutrits)

    Returns:
        (K, dim) complex array of K=q orthonormal codewords in q^5 dimensions;
        basis index k*q^4 + l*q^3 + m*q^2 + n*q + p (qudit 0 most significant)
    """
    omega = np.exp(2j * np.pi / q)
    dim = q ** 5
    K = q
    code_states = np.zeros((K, dim), dtype=complex)

    for a in range(q):
        for k in range(q):
            for l in range(q):
                for m in range(q):
                    for n in range(q):
                        p = (a - k - l - m - n) % q
                        phase = omega ** (k*l + l*m + m*n + n*p + p*k)
                        idx = k * q**4 + l * q**3 + m * q**2 + n * q + p
                        code_states[a, idx] = phase / q**2

    return code_states


# Registry: name -> (npz path relative to results/campaigns, encoder arguments)
_CODE_REGISTRY = {
    # The central comparison code (paper Sec. VI A): channel-weighted
    # protocol, L = 2, all-to-all, MS 40; selected run of the four at L = 2.
    'central_553_3': ('r1_campaign/t1_codes/A_L2_s100.npz',
                      {'d': 3, 'n_qudit': 5}),
    # Analytical benchmark
    'five_qudit_d3': (None, {'d': 3, 'n_qudit': 5, 'analytical': True}),
}


def list_codes():
    """List all available code names."""
    return list(_CODE_REGISTRY.keys())


def code_path(name):
    """Absolute path of the parameter file behind a catalog entry (None for
    the analytical benchmark)."""
    npz_file, _ = _CODE_REGISTRY[name]
    return None if npz_file is None else os.path.normpath(
        os.path.join(_RESULTS_DIR, npz_file))


def load_code(name):
    """
    Load a variational code by name and rebuild its codewords.

    Returns dict with:
      'code_states': (K, dim) array of orthonormal codewords
      'params': parameters (None for analytical codes), shape (L, P_layer)
      'metadata': dict with d, n_qudit, K, distance, n_layers, final_loss,
                  n_steps, noise_type, connections (edge list used)
      'name': the code name
    """
    if name not in _CODE_REGISTRY:
        raise ValueError(f"Unknown code '{name}'. Available: {list_codes()}")

    npz_file, info = _CODE_REGISTRY[name]

    # Analytical benchmark code
    if info.get('analytical'):
        q = info['d']
        code_states = five_qudit_code_states(q)
        return {
            'code_states': code_states,
            'params': None,
            'metadata': {
                'd': q, 'n_qudit': 5, 'K': q, 'distance': 3,
                'n_layers': 0, 'loss': 0.0,
                'type': 'stabilizer', 'name': f'[[5,1,3]]_Z{q}',
            },
            'name': name,
        }

    # Load from npz
    from src.loss import load_varqec_result
    path = os.path.normpath(os.path.join(_RESULTS_DIR, npz_file))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trained code not found: {path}")

    data = load_varqec_result(path)
    params = pnp.array(data['params'], requires_grad=False)

    d = info['d']
    n_qudit = info['n_qudit']
    K = d

    # Connectivity: the parameter file stores its edge list (ring if absent).
    connections = None
    if 'connections' in data and np.asarray(data['connections']).ndim == 2:
        connections = [[int(a), int(b)] for a, b in np.asarray(data['connections'])]

    # Extract code states by running the ansatz of Sec. IV A
    from src.encoder import create_native_encoder
    encoder, connections, _ = create_native_encoder(
        n_qudit, d, connections=connections, force_manual=True)

    code_states = np.zeros((K, d ** n_qudit), dtype=complex)
    for k in range(K):
        code_states[k] = np.array(encoder(params, k))

    metadata = {
        'd': d, 'n_qudit': n_qudit, 'K': K,
        'distance': int(data.get('distance', 2)),
        'n_layers': int(data.get('n_layers', 1)),
        'final_loss': float(data.get('final_loss', -1)),
        'n_steps': len(data.get('losses', [])),
        'noise_type': str(data.get('noise_type', 'unknown')),
        'connections': connections,
        'path': path,
    }

    return {
        'code_states': code_states,
        'params': params,
        'metadata': metadata,
        'name': name,
    }
