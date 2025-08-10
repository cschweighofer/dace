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
    float_n = datatype(N)
    data = np.fromfunction(lambda i, j: (i * j) / M, (N, M), dtype=datatype)

    return float_n, data


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/covariance_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (500, 600)
    M = (1400, 1800)
    L = (3200, 4000)
    PAPER = (1200, 1400)

    def __init__(self, M, N):
        self.M = M
        self.N = N

    def init(self):
        float_n, data = initialize(M=self.M, N=self.N)
        return self.M, self.N, float_n, data

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    M, N, float_n, data = size.init()

    float_n_ref = np.copy(float_n)
    data_ref = np.copy(data)
    sdfg(M=M, N=N, float_n=float_n_ref, data=data_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    float_n2 = np.copy(float_n)
    data2 = np.copy(data)
    sdfg_copy(M=M, N=N, _float_n=float_n2, _data=data2)

    diff_float_n = np.sqrt(np.mean((float_n2 - float_n_ref)**2))
    diff_data = np.sqrt(np.mean((data2 - data_ref)**2))
    diff = max(np.max(np.abs(float_n2 - float_n_ref)), np.max(np.abs(data2 - data_ref)))
    print("RMS Difference FLOAT_N:", diff_float_n)
    print("RMS Difference DATA:", diff_data)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (M={model_size.M}, N={model_size.N})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()
