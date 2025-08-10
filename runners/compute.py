# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from real_lowering import change_fptype

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(M, N):
    from numpy.random import default_rng
    rng = default_rng(42)
    array_1 = rng.uniform(0, 1000, size=(M, N)).astype(np.int64)
    array_2 = rng.uniform(0, 1000, size=(M, N)).astype(np.int64)
    a = np.int64(4)
    b = np.int64(3)
    c = np.int64(9)
    return array_1, array_2, a, b, c


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/compute_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (2000, 2000)
    M = (5000, 5000)
    L = (16000, 16000)
    PAPER = (12500, 12500)

    def __init__(self, M, N):
        self.M = M
        self.N = N

    def init(self):
        array_1, array_2, a, b, c = initialize(M=self.M, N=self.N)
        return self.M, self.N, array_1, array_2, a, b, c

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    M, N, array_1, array_2, a, b, c = size.init()

    array_1_ref = np.copy(array_1)
    array_2_ref = np.copy(array_2)
    a_ref = np.copy(a)
    b_ref = np.copy(b)
    c_ref = np.copy(c)
    sdfg(M=M, N=N, array_1=array_1_ref, array_2=array_2_ref, a=a_ref, b=b_ref, c=c_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    array_12 = np.copy(array_1)
    array_22 = np.copy(array_2)
    a2 = np.copy(a)
    b2 = np.copy(b)
    c2 = np.copy(c)
    sdfg_copy(M=M, N=N, _array_1=array_12, _array_2=array_22, _a=a2, _b=b2, _c=c2)

    diff_array_1 = np.sqrt(np.mean((array_12 - array_1_ref)**2))
    diff_array_2 = np.sqrt(np.mean((array_22 - array_2_ref)**2))
    diff_a = np.sqrt(np.mean((a2 - a_ref)**2))
    diff_b = np.sqrt(np.mean((b2 - b_ref)**2))
    diff_c = np.sqrt(np.mean((c2 - c_ref)**2))
    diff = max(np.max(np.abs(array_12 - array_1_ref)), np.max(np.abs(array_22 - array_2_ref)), np.max(np.abs(a2 - a_ref)), np.max(np.abs(b2 - b_ref)), np.max(np.abs(c2 - c_ref)))
    print("RMS Difference ARRAY_1:", diff_array_1)
    print("RMS Difference ARRAY_2:", diff_array_2)
    print("RMS Difference A:", diff_a)
    print("RMS Difference B:", diff_b)
    print("RMS Difference C:", diff_c)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (M={model_size.M}, N={model_size.N})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()
