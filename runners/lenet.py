# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from real_lowering import change_fptype

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(N, H, W):
    from numpy.random import default_rng
    rng = default_rng(42)

    H_conv1 = H - 4
    W_conv1 = W - 4
    H_pool1 = H_conv1 // 2
    W_pool1 = W_conv1 // 2
    H_conv2 = H_pool1 - 4
    W_conv2 = W_pool1 - 4
    H_pool2 = H_conv2 // 2
    W_pool2 = W_conv2 // 2
    C_before_fc1 = 16 * H_pool2 * W_pool2

    # NHWC data layout
    input = rng.random((N, H, W, 1), dtype=np.float32)
    # Weights
    conv1 = rng.random((5, 5, 1, 6), dtype=np.float32)
    conv1bias = rng.random((6, ), dtype=np.float32)
    conv2 = rng.random((5, 5, 6, 16), dtype=np.float32)
    conv2bias = rng.random((16, ), dtype=np.float32)
    fc1w = rng.random((C_before_fc1, 120), dtype=np.float32)
    fc1b = rng.random((120, ), dtype=np.float32)
    fc2w = rng.random((120, 84), dtype=np.float32)
    fc2b = rng.random((84, ), dtype=np.float32)
    fc3w = rng.random((84, 10), dtype=np.float32)
    fc3b = rng.random((10, ), dtype=np.float32)

    return (input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b,
            fc3w, fc3b, C_before_fc1)


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/lenet_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (4, 28, 28)
    M = (8, 56, 56)
    L = (8, 176, 176)
    PAPER = (16, 256, 256)

    def __init__(self, N, H, W):
        self.N = N
        self.H = H
        self.W = W

    def init(self):
        input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b, fc3w, fc3b, C_before_fc1 = initialize(N=self.N, H=self.H, W=self.W)
        return self.N, self.H, self.W, input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b, fc3w, fc3b, C_before_fc1

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    N, H, W, input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b, fc3w, fc3b, C_before_fc1 = size.init()

    input_ref = np.copy(input)
    conv1_ref = np.copy(conv1)
    conv1bias_ref = np.copy(conv1bias)
    conv2_ref = np.copy(conv2)
    conv2bias_ref = np.copy(conv2bias)
    fc1w_ref = np.copy(fc1w)
    fc1b_ref = np.copy(fc1b)
    fc2w_ref = np.copy(fc2w)
    fc2b_ref = np.copy(fc2b)
    fc3w_ref = np.copy(fc3w)
    fc3b_ref = np.copy(fc3b)
    C_before_fc1_ref = np.copy(C_before_fc1)
    sdfg(N=N, H=H, W=W, input=input_ref, conv1=conv1_ref, conv1bias=conv1bias_ref, conv2=conv2_ref, conv2bias=conv2bias_ref, fc1w=fc1w_ref, fc1b=fc1b_ref, fc2w=fc2w_ref, fc2b=fc2b_ref, fc3w=fc3w_ref, fc3b=fc3b_ref, C_before_fc1=C_before_fc1_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    input2 = np.copy(input)
    conv12 = np.copy(conv1)
    conv1bias2 = np.copy(conv1bias)
    conv22 = np.copy(conv2)
    conv2bias2 = np.copy(conv2bias)
    fc1w2 = np.copy(fc1w)
    fc1b2 = np.copy(fc1b)
    fc2w2 = np.copy(fc2w)
    fc2b2 = np.copy(fc2b)
    fc3w2 = np.copy(fc3w)
    fc3b2 = np.copy(fc3b)
    C_before_fc12 = np.copy(C_before_fc1)
    sdfg_copy(N=N, H=H, W=W, _input=input2, _conv1=conv12, _conv1bias=conv1bias2, _conv2=conv22, _conv2bias=conv2bias2, _fc1w=fc1w2, _fc1b=fc1b2, _fc2w=fc2w2, _fc2b=fc2b2, _fc3w=fc3w2, _fc3b=fc3b2, _C_before_fc1=C_before_fc12)

    diff_input = np.sqrt(np.mean((input2 - input_ref)**2))
    diff_conv1 = np.sqrt(np.mean((conv12 - conv1_ref)**2))
    diff_conv1bias = np.sqrt(np.mean((conv1bias2 - conv1bias_ref)**2))
    diff_conv2 = np.sqrt(np.mean((conv22 - conv2_ref)**2))
    diff_conv2bias = np.sqrt(np.mean((conv2bias2 - conv2bias_ref)**2))
    diff_fc1w = np.sqrt(np.mean((fc1w2 - fc1w_ref)**2))
    diff_fc1b = np.sqrt(np.mean((fc1b2 - fc1b_ref)**2))
    diff_fc2w = np.sqrt(np.mean((fc2w2 - fc2w_ref)**2))
    diff_fc2b = np.sqrt(np.mean((fc2b2 - fc2b_ref)**2))
    diff_fc3w = np.sqrt(np.mean((fc3w2 - fc3w_ref)**2))
    diff_fc3b = np.sqrt(np.mean((fc3b2 - fc3b_ref)**2))
    diff_C_before_fc1 = np.sqrt(np.mean((C_before_fc12 - C_before_fc1_ref)**2))
    diff = max(np.max(np.abs(input2 - input_ref)), np.max(np.abs(conv12 - conv1_ref)), np.max(np.abs(conv1bias2 - conv1bias_ref)), np.max(np.abs(conv22 - conv2_ref)), np.max(np.abs(conv2bias2 - conv2bias_ref)), np.max(np.abs(fc1w2 - fc1w_ref)), np.max(np.abs(fc1b2 - fc1b_ref)), np.max(np.abs(fc2w2 - fc2w_ref)), np.max(np.abs(fc2b2 - fc2b_ref)), np.max(np.abs(fc3w2 - fc3w_ref)), np.max(np.abs(fc3b2 - fc3b_ref)), np.max(np.abs(C_before_fc12 - C_before_fc1_ref)))
    print("RMS Difference INPUT:", diff_input)
    print("RMS Difference CONV1:", diff_conv1)
    print("RMS Difference CONV1BIAS:", diff_conv1bias)
    print("RMS Difference CONV2:", diff_conv2)
    print("RMS Difference CONV2BIAS:", diff_conv2bias)
    print("RMS Difference FC1W:", diff_fc1w)
    print("RMS Difference FC1B:", diff_fc1b)
    print("RMS Difference FC2W:", diff_fc2w)
    print("RMS Difference FC2B:", diff_fc2b)
    print("RMS Difference FC3W:", diff_fc3w)
    print("RMS Difference FC3B:", diff_fc3b)
    print("RMS Difference C_BEFORE_FC1:", diff_C_before_fc1)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (N={model_size.N}, H={model_size.H}, W={model_size.W})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()
