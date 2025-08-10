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
    L = np.fromfunction(lambda i, j: (i + N - j + 1) * 2 / N, (N, N),
                        dtype=datatype)
    x = np.full((N, ), -999, dtype=datatype)
    b = np.fromfunction(lambda i: i, (N, ), dtype=datatype)

    return L, x, b


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/trisolv_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (2000)
    M = (5000)
    L = (14000)
    PAPER = (16000)

    def __init__(self, N):
        self.N = N

    def init(self):
        L, x, b = initialize(N=self.N)
        return self.N, L, x, b

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    N, L, x, b = size.init()

    L_ref = np.copy(L)
    x_ref = np.copy(x)
    b_ref = np.copy(b)
    sdfg(N=N, L=L_ref, x=x_ref, b=b_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    L2 = np.copy(L)
    x2 = np.copy(x)
    b2 = np.copy(b)
    sdfg_copy(N=N, _L=L2, _x=x2, _b=b2)

    diff_L = np.sqrt(np.mean((L2 - L_ref)**2))
    diff_x = np.sqrt(np.mean((x2 - x_ref)**2))
    diff_b = np.sqrt(np.mean((b2 - b_ref)**2))
    diff = max(np.max(np.abs(L2 - L_ref)), np.max(np.abs(x2 - x_ref)), np.max(np.abs(b2 - b_ref)))
    print("RMS Difference L:", diff_L)
    print("RMS Difference X:", diff_x)
    print("RMS Difference B:", diff_b)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (N={model_size.N})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()
