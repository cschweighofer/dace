# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from real_lowering import change_fptype

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(N, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    A = np.fromfunction(lambda i, j: ((i * j + 1) % N) / N, (N, N),
                        dtype=datatype)
    B = np.fromfunction(lambda i, j: ((i * j + 2) % N) / N, (N, N),
                        dtype=datatype)
    x = np.fromfunction(lambda i: (i % N) / N, (N, ), dtype=datatype)

    return alpha, beta, A, B, x


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/gesummv_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (2000)
    M = (4000)
    L = (14000)
    PAPER = (11200)

    def __init__(self, N):
        self.N = N

    def init(self):
        alpha, beta, A, B, x = initialize(N=self.N)
        return self.N, alpha, beta, A, B, x

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    N, alpha, beta, A, B, x = size.init()

    alpha_ref = np.copy(alpha)
    beta_ref = np.copy(beta)
    A_ref = np.copy(A)
    B_ref = np.copy(B)
    x_ref = np.copy(x)
    sdfg(N=N, alpha=alpha_ref, beta=beta_ref, A=A_ref, B=B_ref, x=x_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    alpha2 = np.copy(alpha)
    beta2 = np.copy(beta)
    A2 = np.copy(A)
    B2 = np.copy(B)
    x2 = np.copy(x)
    sdfg_copy(N=N, _alpha=alpha2, _beta=beta2, _A=A2, _B=B2, _x=x2)

    diff_alpha = np.sqrt(np.mean((alpha2 - alpha_ref)**2))
    diff_beta = np.sqrt(np.mean((beta2 - beta_ref)**2))
    diff_A = np.sqrt(np.mean((A2 - A_ref)**2))
    diff_B = np.sqrt(np.mean((B2 - B_ref)**2))
    diff_x = np.sqrt(np.mean((x2 - x_ref)**2))
    diff = max(np.max(np.abs(alpha2 - alpha_ref)), np.max(np.abs(beta2 - beta_ref)), np.max(np.abs(A2 - A_ref)), np.max(np.abs(B2 - B_ref)), np.max(np.abs(x2 - x_ref)))
    print("RMS Difference ALPHA:", diff_alpha)
    print("RMS Difference BETA:", diff_beta)
    print("RMS Difference A:", diff_A)
    print("RMS Difference B:", diff_B)
    print("RMS Difference X:", diff_x)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (N={model_size.N})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()
