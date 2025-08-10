# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from real_lowering import change_fptype

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(NI, NJ, NK, NL, NM, datatype=np.float64):
    A = np.fromfunction(lambda i, j: ((i * j + 1) % NI) / (5 * NI), (NI, NK),
                        dtype=datatype)
    B = np.fromfunction(lambda i, j: ((i * (j + 1) + 2) % NJ) / (5 * NJ),
                        (NK, NJ),
                        dtype=datatype)
    C = np.fromfunction(lambda i, j: (i * (j + 3) % NL) / (5 * NL), (NJ, NM),
                        dtype=datatype)
    D = np.fromfunction(lambda i, j: ((i * (j + 2) + 2) % NK) / (5 * NK),
                        (NM, NL),
                        dtype=datatype)

    return A, B, C, D


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/k3mm_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (800, 850, 900, 950, 1000)
    M = (2000, 2200, 2400, 2600, 2800)
    L = (5500, 6000, 6500, 7000, 7500)
    PAPER = (3200, 3600, 4000, 4400, 4800)

    def __init__(self, NI, NJ, NK, NL, NM):
        self.NI = NI
        self.NJ = NJ
        self.NK = NK
        self.NL = NL
        self.NM = NM

    def init(self):
        A, B, C, D = initialize(NI=self.NI, NJ=self.NJ, NK=self.NK, NL=self.NL, NM=self.NM)
        return self.NI, self.NJ, self.NK, self.NL, self.NM, A, B, C, D

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    NI, NJ, NK, NL, NM, A, B, C, D = size.init()

    A_ref = np.copy(A)
    B_ref = np.copy(B)
    C_ref = np.copy(C)
    D_ref = np.copy(D)
    sdfg(NI=NI, NJ=NJ, NK=NK, NL=NL, NM=NM, A=A_ref, B=B_ref, C=C_ref, D=D_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    A2 = np.copy(A)
    B2 = np.copy(B)
    C2 = np.copy(C)
    D2 = np.copy(D)
    sdfg_copy(NI=NI, NJ=NJ, NK=NK, NL=NL, NM=NM, _A=A2, _B=B2, _C=C2, _D=D2)

    diff_A = np.sqrt(np.mean((A2 - A_ref)**2))
    diff_B = np.sqrt(np.mean((B2 - B_ref)**2))
    diff_C = np.sqrt(np.mean((C2 - C_ref)**2))
    diff_D = np.sqrt(np.mean((D2 - D_ref)**2))
    diff = max(np.max(np.abs(A2 - A_ref)), np.max(np.abs(B2 - B_ref)), np.max(np.abs(C2 - C_ref)), np.max(np.abs(D2 - D_ref)))
    print("RMS Difference A:", diff_A)
    print("RMS Difference B:", diff_B)
    print("RMS Difference C:", diff_C)
    print("RMS Difference D:", diff_D)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (NI={model_size.NI}, NJ={model_size.NJ}, NK={model_size.NK}, NL={model_size.NL}, NM={model_size.NM})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()
