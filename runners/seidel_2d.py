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
    A = np.fromfunction(lambda i, j: (i * (j + 2) + 2) / N, (N, N),
                        dtype=datatype)

    return A


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/seidel_2d_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (8, 50)
    M = (15, 100)
    L = (40, 200)
    PAPER = (100, 400)

    def __init__(self, TSTEPS, N):
        self.TSTEPS = TSTEPS
        self.N = N

    def init(self):
        A = initialize(N=self.N)
        return self.TSTEPS, self.N, A

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    TSTEPS, N, A = size.init()

    A_ref = np.copy(A)
    sdfg(TSTEPS=TSTEPS, N=N, A=A_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    A2 = np.copy(A)
    sdfg_copy(TSTEPS=TSTEPS, N=N, _A=A2)

    diff_A = np.sqrt(np.mean((A2 - A_ref)**2))
    diff = np.max(np.max(np.abs(A2 - A_ref)))
    print("RMS Difference A:", diff_A)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (TSTEPS={model_size.TSTEPS}, N={model_size.N})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()
