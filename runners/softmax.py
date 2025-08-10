# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from real_lowering import change_fptype

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(N, H, SM):
    from numpy.random import default_rng
    rng = default_rng(42)
    x = rng.random((N, H, SM, SM), dtype=np.float32)
    return x


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/softmax_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (16, 16, 128)
    M = (32, 8, 256)
    L = (64, 16, 448)
    PAPER = (64, 16, 512)

    def __init__(self, N, H, SM):
        self.N = N
        self.H = H
        self.SM = SM

    def init(self):
        x = initialize(N=self.N, H=self.H, SM=self.SM)
        return self.N, self.H, self.SM, x

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    N, H, SM, x = size.init()

    x_ref = np.copy(x)
    sdfg(N=N, H=H, SM=SM, x=x_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    x2 = np.copy(x)
    sdfg_copy(N=N, H=H, SM=SM, _x=x2)

    diff_x = np.sqrt(np.mean((x2 - x_ref)**2))
    diff = max(np.max(np.abs(x2 - x_ref)))
    print("RMS Difference X:", diff_x)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (N={model_size.N}, H={model_size.H}, SM={model_size.SM})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()
