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
    from numpy.random import default_rng
    rng = default_rng(42)

    A = rng.random((M, N), dtype=datatype)
    while np.linalg.matrix_rank(A) < N:
        A = rng.random((M, N), dtype=datatype)

    return A


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/gramschmidt_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (70, 60)
    M = (220, 180)
    L = (600, 500)
    PAPER = (240, 200)

    def __init__(self, M, N):
        self.M = M
        self.N = N

    def init(self):
        A = initialize(M=self.M, N=self.N)
        return self.M, self.N, A

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    M, N, A = size.init()

    A_ref = np.copy(A)
    sdfg(M=M, N=N, A=A_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    A2 = np.copy(A)
    sdfg_copy(M=M, N=N, _A=A2)

    diff_A = np.sqrt(np.mean((A2 - A_ref)**2))
    diff = max(np.max(np.abs(A2 - A_ref)))
    print("RMS Difference A:", diff_A)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (M={model_size.M}, N={model_size.N})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()
