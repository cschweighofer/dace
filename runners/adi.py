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
    u = np.fromfunction(lambda i, j: (i + N - j) / N, (N, N), dtype=datatype)

    return u


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/adi_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    # XS = (1, 100)   # Even smaller test
    S = (5, 100)
    M = (20, 200)
    L = (50, 500)
    # PAPER = (100, 200)

    def __init__(self, TSTEPS, N):
        self.TSTEPS = TSTEPS
        self.N = N

    def init(self):
        u = initialize(N=self.N)
        return self.TSTEPS, self.N, u

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    TSTEPS, N, u = size.init()

    u_ref = np.copy(u)
    sdfg(TSTEPS=TSTEPS, N=N, u=u_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    u2 = np.copy(u)
    sdfg_copy(TSTEPS=TSTEPS, N=N, _u=u2)

    diff_u = np.sqrt(np.mean((u2 - u_ref)**2))
    diff = np.max(np.abs(u2 - u_ref))
    print("RMS Difference U:", diff_u)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (TSTEPS={model_size.TSTEPS}, N={model_size.N})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()
