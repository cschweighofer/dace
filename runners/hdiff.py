# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from real_lowering import change_fptype

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(I, J, K):
    from numpy.random import default_rng
    rng = default_rng(42)

    # Define arrays
    in_field = rng.random((I + 4, J + 4, K))
    out_field = rng.random((I, J, K))
    coeff = rng.random((I, J, K))

    return in_field, out_field, coeff


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/hdiff_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (64, 64, 60)
    M = (128, 128, 160)
    L = (384, 384, 160)
    PAPER = (256, 256, 160)

    def __init__(self, I, J, K):
        self.I = I
        self.J = J
        self.K = K

    def init(self):
        in_field, out_field, coeff = initialize(I=self.I, J=self.J, K=self.K)
        return self.I, self.J, self.K, in_field, out_field, coeff

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    I, J, K, in_field, out_field, coeff = size.init()

    in_field_ref = np.copy(in_field)
    out_field_ref = np.copy(out_field)
    coeff_ref = np.copy(coeff)
    sdfg(I=I, J=J, K=K, in_field=in_field_ref, out_field=out_field_ref, coeff=coeff_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    in_field2 = np.copy(in_field)
    out_field2 = np.copy(out_field)
    coeff2 = np.copy(coeff)
    sdfg_copy(I=I, J=J, K=K, _in_field=in_field2, _out_field=out_field2, _coeff=coeff2)

    diff_in_field = np.sqrt(np.mean((in_field2 - in_field_ref)**2))
    diff_out_field = np.sqrt(np.mean((out_field2 - out_field_ref)**2))
    diff_coeff = np.sqrt(np.mean((coeff2 - coeff_ref)**2))
    diff = max(np.max(np.abs(in_field2 - in_field_ref)), np.max(np.abs(out_field2 - out_field_ref)), np.max(np.abs(coeff2 - coeff_ref)))
    print("RMS Difference IN_FIELD:", diff_in_field)
    print("RMS Difference OUT_FIELD:", diff_out_field)
    print("RMS Difference COEFF:", diff_coeff)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (I={model_size.I}, J={model_size.J}, K={model_size.K})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()
