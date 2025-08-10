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
    r = np.fromfunction(lambda i: N + 1 - i, (N, ), dtype=datatype)
    return r


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/durbin_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (1000)
    M = (6000)
    L = (20000)
    PAPER = (16000)

    def __init__(self, N):
        self.N = N

    def init(self):
        r = initialize(N=self.N)
        return self.N, r

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    N, r = size.init()

    r_ref = np.copy(r)
    sdfg(N=N, r=r_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    r2 = np.copy(r)
    sdfg_copy(N=N, _r=r2)

    diff_r = np.sqrt(np.mean((r2 - r_ref)**2))
    diff = max(np.max(np.abs(r2 - r_ref)))
    print("RMS Difference R:", diff_r)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (N={model_size.N})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()
