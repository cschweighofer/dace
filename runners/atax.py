# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from real_lowering import change_fptype

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(M, N, datatype=np.float64):
    fn = datatype(N)
    x = np.fromfunction(lambda i: 1 + (i / fn), (N, ), dtype=datatype)
    A = np.fromfunction(lambda i, j: ((i + j) % N) / (5 * M), (M, N),
                        dtype=datatype)

    return x, A


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/atax_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (4000, 5000)
    M = (10000, 12500)
    L = (20000, 25000)
    PAPER = (18000, 22000)

    def __init__(self, M, N):
        self.M = M
        self.N = N

    def init(self):
        x, A = initialize(M=self.M, N=self.N)
        return self.M, self.N, x, A

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    M, N, x, A = size.init()

    A_ref = np.copy(A)
    x_ref = np.copy(x)
    sdfg(M=M, N=N, A=A_ref, x=x_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    A2 = np.copy(A)
    x2 = np.copy(x)
    sdfg_copy(M=M, N=N, _A=A2, _x=x2)

    diff_A = np.sqrt(np.mean((A2 - A_ref)**2))
    diff_x = np.sqrt(np.mean((x2 - x_ref)**2))
    diff = max(np.max(np.abs(A2 - A_ref)), np.max(np.abs(x2 - x_ref)))
    print("RMS Difference A:", diff_A)
    print("RMS Difference X:", diff_x)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (M={model_size.M}, N={model_size.N})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()
