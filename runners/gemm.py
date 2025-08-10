# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from real_lowering import change_fptype

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(NI, NJ, NK, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.fromfunction(lambda i, j: ((i * j + 1) % NI) / NI, (NI, NJ),
                        dtype=datatype)
    A = np.fromfunction(lambda i, k: (i * (k + 1) % NK) / NK, (NI, NK),
                        dtype=datatype)
    B = np.fromfunction(lambda k, j: (k * (j + 2) % NJ) / NJ, (NK, NJ),
                        dtype=datatype)

    return alpha, beta, C, A, B


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/gemm_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (1000, 1100, 1200)
    M = (2500, 2750, 3000)
    L = (7000, 7500, 8000)
    PAPER = (2000, 2300, 2600)

    def __init__(self, NI, NJ, NK):
        self.NI = NI
        self.NJ = NJ
        self.NK = NK

    def init(self):
        alpha, beta, C, A, B = initialize(NI=self.NI, NJ=self.NJ, NK=self.NK)
        return self.NI, self.NJ, self.NK, alpha, beta, C, A, B

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    NI, NJ, NK, alpha, beta, C, A, B = size.init()

    alpha_ref = np.copy(alpha)
    beta_ref = np.copy(beta)
    C_ref = np.copy(C)
    A_ref = np.copy(A)
    B_ref = np.copy(B)
    sdfg(NI=NI, NJ=NJ, NK=NK, alpha=alpha_ref, beta=beta_ref, C=C_ref, A=A_ref, B=B_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    alpha2 = np.copy(alpha)
    beta2 = np.copy(beta)
    C2 = np.copy(C)
    A2 = np.copy(A)
    B2 = np.copy(B)
    sdfg_copy(NI=NI, NJ=NJ, NK=NK, _alpha=alpha2, _beta=beta2, _C=C2, _A=A2, _B=B2)

    diff_alpha = np.sqrt(np.mean((alpha2 - alpha_ref)**2))
    diff_beta = np.sqrt(np.mean((beta2 - beta_ref)**2))
    diff_C = np.sqrt(np.mean((C2 - C_ref)**2))
    diff_A = np.sqrt(np.mean((A2 - A_ref)**2))
    diff_B = np.sqrt(np.mean((B2 - B_ref)**2))
    diff = max(np.max(np.abs(alpha2 - alpha_ref)), np.max(np.abs(beta2 - beta_ref)), np.max(np.abs(C2 - C_ref)), np.max(np.abs(A2 - A_ref)), np.max(np.abs(B2 - B_ref)))
    print("RMS Difference ALPHA:", diff_alpha)
    print("RMS Difference BETA:", diff_beta)
    print("RMS Difference C:", diff_C)
    print("RMS Difference A:", diff_A)
    print("RMS Difference B:", diff_B)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (NI={model_size.NI}, NJ={model_size.NJ}, NK={model_size.NK})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()
