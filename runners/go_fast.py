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
    x = rng.random((N, N), dtype=np.float64)
    return x


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/go_fast_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (2000)
    M = (6000)
    L = (20000)
    PAPER = (12500)

    def __init__(self, N):
        self.N = N

    def init(self):
        a = initialize(N=self.N)
        return self.N, a

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    N, a = size.init()

    a_ref = np.copy(a)
    sdfg(N=N, a=a_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    a2 = np.copy(a)
    sdfg_copy(N=N, _a=a2)

    diff_a = np.sqrt(np.mean((a2 - a_ref)**2))
    diff = max(np.max(np.abs(a2 - a_ref)))
    print("RMS Difference A:", diff_a)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (N={model_size.N})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()
