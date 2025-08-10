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
    A = np.fromfunction(lambda i, j: (i * (j + 1) % N) / N, (N, M),
                        dtype=datatype)
    p = np.fromfunction(lambda i: (i % M) / M, (M, ), dtype=datatype)
    r = np.fromfunction(lambda i: (i % N) / N, (N, ), dtype=datatype)

    return A, p, r


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/bicg_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (4000, 5000)
    M = (10000, 12500)
    L = (20000, 25000)
    PAPER = (18000, 22000)

    def __init__(self, M, N):
        self.M = M
        self.N = N

    def init(self):
        A, p, r = initialize(M=self.M, N=self.N)
        return self.M, self.N, A, p, r

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    M, N, A, p, r = size.init()

    A_ref = np.copy(A)
    p_ref = np.copy(p)
    r_ref = np.copy(r)
    sdfg(M=M, N=N, A=A_ref, p=p_ref, r=r_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    A2 = np.copy(A)
    p2 = np.copy(p)
    r2 = np.copy(r)
    sdfg_copy(M=M, N=N, _A=A2, _p=p2, _r=r2)

    diff_A = np.sqrt(np.mean((A2 - A_ref)**2))
    diff_p = np.sqrt(np.mean((p2 - p_ref)**2))
    diff_r = np.sqrt(np.mean((r2 - r_ref)**2))
    diff = max(np.max(np.abs(A2 - A_ref)), np.max(np.abs(p2 - p_ref)), np.max(np.abs(r2 - r_ref)))
    print("RMS Difference A:", diff_A)
    print("RMS Difference P:", diff_p)
    print("RMS Difference R:", diff_r)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (M={model_size.M}, N={model_size.N})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()
