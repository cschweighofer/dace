# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from real_lowering import change_fptype

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(N):
    from numpy.random import default_rng
    rng = default_rng(42)
    data = rng.integers(0, 256, size=(N, ), dtype=np.uint8)
    return data


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/crc16_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (1600)
    M = (16000)
    L = (160000)
    PAPER = (1000000)

    def __init__(self, N):
        self.N = N

    def init(self):
        data = initialize(N=self.N)
        return self.N, data

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    N, data = size.init()

    data_ref = np.copy(data)
    sdfg(N=N, data=data_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    data2 = np.copy(data)
    sdfg_copy(N=N, _data=data2)

    diff_data = np.sqrt(np.mean((data2 - data_ref)**2))
    diff = max(np.max(np.abs(data2 - data_ref)))
    print("RMS Difference DATA:", diff_data)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (N={model_size.N})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()
