"""Top-level package of the VarQEC qudit reproduction code.

The package re-exports the main building blocks so that short scripts can
write ``from src import ...``:

  still load;
* the native trapped-ion gate set (``src.gates``), hardware error bases and
  noise functions (``src.errors``), and the native encoder
  (``src.encoder``) of paper Secs. II--IV;
* the detection-loss functions and result I/O of ``src.loss``.

The decoders (``src.decoders``), the calibrated-channel machinery
(``src.correlated_noise``, ``src.meth_weights``) and the central-comparison
helpers (``src.central_comparison``) are imported by their full module path
and are not re-exported here.
"""

# Native trapped-ion pipeline
from src.gates import (XY_gate, Z_gate, MS_gate, CSUM_gate, CSUB_gate, CEX_gate,
                       zz_subspace_gate, ls_global_gate,
                       light_shift_gate, light_shift_gate_hrmo)
from src.errors import (
    qudit_hardware_error_basis, build_native_error_set,
    hardware_error_basis, build_qutrit_error_set,
    make_hardware_noise_fn, build_native_error_set_factored,
    ErrorModel,
)
from src.encoder import create_native_encoder

# Loss functions
from src.loss import (
    kl_loss_fast, kl_loss_detection_minibatch, save_varqec_result, load_varqec_result,
)
