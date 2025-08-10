# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from real_lowering import change_fptype

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def rng_complex(shape, rng):
    return (rng.random(shape) + rng.random(shape) * 1j)


def initialize(R, K):
    from numpy.random import default_rng
    rng = default_rng(42)

    N = R**K
    X = rng_complex((N, ), rng)
    Y = np.zeros_like(X, dtype=np.complex128)

    return N, X, Y


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/stockham_fft_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (2, 15)
    M = (2, 18)
    L = (2, 21)
    PAPER = (4, 10)

    def __init__(self, R, K):
        self.R = R
        self.K = K

    def init(self):
        N, x, y = initialize(R=self.R, K=self.K)
        return self.R, self.K, N, x, y

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    R, K, N, x, y = size.init()

    N_ref = np.copy(N)
    x_ref = np.copy(x)
    y_ref = np.copy(y)
    sdfg(R=R, K=K, N=N_ref, x=x_ref, y=y_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    N2 = np.copy(N)
    x2 = np.copy(x)
    y2 = np.copy(y)
    sdfg_copy(R=R, K=K, _N=N2, _x=x2, _y=y2)

    diff_N = np.sqrt(np.mean((N2 - N_ref)**2))
    diff_x = np.sqrt(np.mean((x2 - x_ref)**2))
    diff_y = np.sqrt(np.mean((y2 - y_ref)**2))
    diff = max(np.max(np.abs(N2 - N_ref)), np.max(np.abs(x2 - x_ref)), np.max(np.abs(y2 - y_ref)))
    print("RMS Difference N:", diff_N)
    print("RMS Difference X:", diff_x)
    print("RMS Difference Y:", diff_y)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (R={model_size.R}, K={model_size.K})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()
