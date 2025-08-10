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
    A = np.fromfunction(lambda i, j, k: (i + j + (N - k)) * 10 / N, (N, N, N),
                        dtype=datatype)
    B = np.copy(A)

    return A, B


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/heat_3d_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (25, 25)
    M = (50, 40)
    # L = (100, 70)
    # PAPER = (500, 120)

    def __init__(self, TSTEPS, N):
        self.TSTEPS = TSTEPS
        self.N = N

    def init(self):
        A, B = initialize(N=self.N)
        return self.TSTEPS, self.N, A, B

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    TSTEPS, N, A, B = size.init()

    A_ref = np.copy(A)
    B_ref = np.copy(B)
    sdfg(TSTEPS=TSTEPS, N=N, A=A_ref, B=B_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    A2 = np.copy(A)
    B2 = np.copy(B)
    sdfg_copy(TSTEPS=TSTEPS, N=N, _A=A2, _B=B2)

    diff_A = np.sqrt(np.mean((A2 - A_ref)**2))
    diff_B = np.sqrt(np.mean((B2 - B_ref)**2))
    diff = max(np.max(np.abs(A2 - A_ref)), np.max(np.abs(B2 - B_ref)))
    print("RMS Difference A:", diff_A)
    print("RMS Difference B:", diff_B)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (TSTEPS={model_size.TSTEPS}, N={model_size.N})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()
