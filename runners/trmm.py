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
    alpha = datatype(1.5)
    A = np.fromfunction(lambda i, j: ((i * j) % M) / M, (M, M), dtype=datatype)
    for i in range(M):
        A[i, i] = 1.0
    B = np.fromfunction(lambda i, j: ((N + i - j) % N) / N, (M, N),
                        dtype=datatype)

    return alpha, A, B


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/trmm_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (65, 80)
    M = (200, 250)
    L = (600, 700)
    PAPER = (1000, 1200)

    def __init__(self, M, N):
        self.M = M
        self.N = N

    def init(self):
        alpha, A, B = initialize(M=self.M, N=self.N)
        return self.M, self.N, alpha, A, B

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    M, N, alpha, A, B = size.init()

    alpha_ref = np.copy(alpha)
    A_ref = np.copy(A)
    B_ref = np.copy(B)
    sdfg(M=M, N=N, alpha=alpha_ref, A=A_ref, B=B_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    alpha2 = np.copy(alpha)
    A2 = np.copy(A)
    B2 = np.copy(B)
    sdfg_copy(M=M, N=N, _alpha=alpha2, _A=A2, _B=B2)

    diff_alpha = np.sqrt(np.mean((alpha2 - alpha_ref)**2))
    diff_A = np.sqrt(np.mean((A2 - A_ref)**2))
    diff_B = np.sqrt(np.mean((B2 - B_ref)**2))
    diff = max(np.max(np.abs(alpha2 - alpha_ref)), np.max(np.abs(A2 - A_ref)), np.max(np.abs(B2 - B_ref)))
    print("RMS Difference ALPHA:", diff_alpha)
    print("RMS Difference A:", diff_A)
    print("RMS Difference B:", diff_B)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (M={model_size.M}, N={model_size.N})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()
