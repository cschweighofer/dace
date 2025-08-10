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
    beta = datatype(1.2)
    C = np.fromfunction(lambda i, j: ((i * j + 2) % N) / M, (N, N),
                        dtype=datatype)
    A = np.fromfunction(lambda i, j: ((i * j + 1) % N) / N, (N, M),
                        dtype=datatype)

    return alpha, beta, C, A


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/syrk_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (50, 70)
    M = (150, 200)
    # L = (500, 600)
    # PAPER = (1000, 1200)

    def __init__(self, M, N):
        self.M = M
        self.N = N

    def init(self):
        alpha, beta, C, A = initialize(M=self.M, N=self.N)
        return self.M, self.N, alpha, beta, C, A

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    M, N, alpha, beta, C, A = size.init()

    alpha_ref = np.copy(alpha)
    beta_ref = np.copy(beta)
    C_ref = np.copy(C)
    A_ref = np.copy(A)
    sdfg(M=M, N=N, alpha=alpha_ref, beta=beta_ref, C=C_ref, A=A_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    alpha2 = np.copy(alpha)
    beta2 = np.copy(beta)
    C2 = np.copy(C)
    A2 = np.copy(A)
    sdfg_copy(M=M, N=N, _alpha=alpha2, _beta=beta2, _C=C2, _A=A2)

    diff_alpha = np.sqrt(np.mean((alpha2 - alpha_ref)**2))
    diff_beta = np.sqrt(np.mean((beta2 - beta_ref)**2))
    diff_C = np.sqrt(np.mean((C2 - C_ref)**2))
    diff_A = np.sqrt(np.mean((A2 - A_ref)**2))
    diff = max(np.max(np.abs(alpha2 - alpha_ref)), np.max(np.abs(beta2 - beta_ref)), np.max(np.abs(C2 - C_ref)), np.max(np.abs(A2 - A_ref)))
    print("RMS Difference ALPHA:", diff_alpha)
    print("RMS Difference BETA:", diff_beta)
    print("RMS Difference C:", diff_C)
    print("RMS Difference A:", diff_A)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (M={model_size.M}, N={model_size.N})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()
