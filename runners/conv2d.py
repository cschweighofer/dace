# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from real_lowering import change_fptype

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(C_in, C_out, H, K, N, W):
    from numpy.random import default_rng
    rng = default_rng(42)
    # NHWC data layout
    input = rng.random((N, H, W, C_in), dtype=np.float32)
    # Weights
    weights = rng.random((K, K, C_in, C_out), dtype=np.float32)
    bias = rng.random((C_out, ), dtype=np.float32)
    return input, weights, bias


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/conv2d_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (8, 3, 16, 2, 32, 32)
    M = (8, 3, 8, 5, 64, 64)
    L = (8, 3, 8, 10, 128, 128)
    PAPER = (8, 3, 16, 20, 256, 256)

    def __init__(self, N, C_in, C_out, K, H, W):
        self.N = N
        self.C_in = C_in
        self.C_out = C_out
        self.K = K
        self.H = H
        self.W = W

    def init(self):
        input, weights, bias = initialize(C_in=self.C_in, C_out=self.C_out, H=self.H, K=self.K, N=self.N, W=self.W)
        return self.N, self.C_in, self.C_out, self.K, self.H, self.W, input, weights, bias

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    N, C_in, C_out, K, H, W, input, weights, bias = size.init()

    input_ref = np.copy(input)
    weights_ref = np.copy(weights)
    bias_ref = np.copy(bias)
    sdfg(N=N, C_in=C_in, C_out=C_out, K=K, H=H, W=W, input=input_ref, weights=weights_ref, bias=bias_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    input2 = np.copy(input)
    weights2 = np.copy(weights)
    bias2 = np.copy(bias)
    sdfg_copy(N=N, C_in=C_in, C_out=C_out, K=K, H=H, W=W, _input=input2, _weights=weights2, _bias=bias2)

    diff_input = np.sqrt(np.mean((input2 - input_ref)**2))
    diff_weights = np.sqrt(np.mean((weights2 - weights_ref)**2))
    diff_bias = np.sqrt(np.mean((bias2 - bias_ref)**2))
    diff = max(np.max(np.abs(input2 - input_ref)), np.max(np.abs(weights2 - weights_ref)), np.max(np.abs(bias2 - bias_ref)))
    print("RMS Difference INPUT:", diff_input)
    print("RMS Difference WEIGHTS:", diff_weights)
    print("RMS Difference BIAS:", diff_bias)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (N={model_size.N}, C_IN={model_size.C_in}, C_OUT={model_size.C_out}, K={model_size.K}, H={model_size.H}, W={model_size.W})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()
