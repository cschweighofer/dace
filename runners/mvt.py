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
    x1 = np.fromfunction(lambda i: (i % N) / N, (N, ), dtype=datatype)
    x2 = np.fromfunction(lambda i: ((i + 1) % N) / N, (N, ), dtype=datatype)
    y_1 = np.fromfunction(lambda i: ((i + 3) % N) / N, (N, ), dtype=datatype)
    y_2 = np.fromfunction(lambda i: ((i + 4) % N) / N, (N, ), dtype=datatype)
    A = np.fromfunction(lambda i, j: (i * j % N) / N, (N, N), dtype=datatype)

    return x1, x2, y_1, y_2, A


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/mvt_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (5500)
    M = (11000)
    L = (22000)
    PAPER = (16000)

    def __init__(self, N):
        self.N = N

    def init(self):
        x1, x2, y_1, y_2, A = initialize(N=self.N)
        return self.N, x1, x2, y_1, y_2, A

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    N, x1, x2, y_1, y_2, A = size.init()

    x1_ref = np.copy(x1)
    x2_ref = np.copy(x2)
    y_1_ref = np.copy(y_1)
    y_2_ref = np.copy(y_2)
    A_ref = np.copy(A)
    sdfg(N=N, x1=x1_ref, x2=x2_ref, y_1=y_1_ref, y_2=y_2_ref, A=A_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    x12 = np.copy(x1)
    x22 = np.copy(x2)
    y_12 = np.copy(y_1)
    y_22 = np.copy(y_2)
    A2 = np.copy(A)
    sdfg_copy(N=N, _x1=x12, _x2=x22, _y_1=y_12, _y_2=y_22, _A=A2)

    diff_x1 = np.sqrt(np.mean((x12 - x1_ref)**2))
    diff_x2 = np.sqrt(np.mean((x22 - x2_ref)**2))
    diff_y_1 = np.sqrt(np.mean((y_12 - y_1_ref)**2))
    diff_y_2 = np.sqrt(np.mean((y_22 - y_2_ref)**2))
    diff_A = np.sqrt(np.mean((A2 - A_ref)**2))
    diff = max(np.max(np.abs(x12 - x1_ref)), np.max(np.abs(x22 - x2_ref)), np.max(np.abs(y_12 - y_1_ref)), np.max(np.abs(y_22 - y_2_ref)), np.max(np.abs(A2 - A_ref)))
    print("RMS Difference X1:", diff_x1)
    print("RMS Difference X2:", diff_x2)
    print("RMS Difference Y_1:", diff_y_1)
    print("RMS Difference Y_2:", diff_y_2)
    print("RMS Difference A:", diff_A)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (N={model_size.N})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()
