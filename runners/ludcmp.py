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
    A = np.empty((N, N), dtype=datatype)
    for i in range(N):
        A[i, :i + 1] = np.fromfunction(lambda j: (-j % N) / N + 1, (i + 1, ),
                                       dtype=datatype)
        A[i, i + 1:] = 0.0
        A[i, i] = 1.0
    A[:] = A @ np.transpose(A)
    fn = datatype(N)
    b = np.fromfunction(lambda i: (i + 1) / fn / 2.0 + 4.0, (N, ),
                        dtype=datatype)

    return A, b


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/ludcmp_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (60)
    M = (220)
    L = (650)
    PAPER = (2000)

    def __init__(self, N):
        self.N = N

    def init(self):
        A, b = initialize(N=self.N)
        return self.N, A, b

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    N, A, b = size.init()

    A_ref = np.copy(A)
    b_ref = np.copy(b)
    sdfg(N=N, A=A_ref, b=b_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    A2 = np.copy(A)
    b2 = np.copy(b)
    sdfg_copy(N=N, _A=A2, _b=b2)

    diff_A = np.sqrt(np.mean((A2 - A_ref)**2))
    diff_b = np.sqrt(np.mean((b2 - b_ref)**2))
    diff = max(np.max(np.abs(A2 - A_ref)), np.max(np.abs(b2 - b_ref)))
    print("RMS Difference A:", diff_A)
    print("RMS Difference B:", diff_b)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (N={model_size.N})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()
