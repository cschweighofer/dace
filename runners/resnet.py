# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from real_lowering import change_fptype

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(N, W, H, C1, C2):
    from numpy.random import default_rng
    rng = default_rng(42)

    # Input
    input = rng.random((N, H, W, C1), dtype=np.float32)
    # Weights
    conv1 = rng.random((1, 1, C1, C2), dtype=np.float32)
    conv2 = rng.random((3, 3, C2, C2), dtype=np.float32)
    conv3 = rng.random((1, 1, C2, C1), dtype=np.float32)
    return (input, conv1, conv2, conv3)


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/resnet_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (8, 14, 14, 32, 8)
    M = (8, 28, 28, 64, 16)
    L = (8, 56, 56, 128, 32)
    PAPER = (8, 56, 56, 256, 64)

    def __init__(self, N, W, H, C1, C2):
        self.N = N
        self.W = W
        self.H = H
        self.C1 = C1
        self.C2 = C2

    def init(self):
        input, conv1, conv2, conv3 = initialize(N=self.N, W=self.W, H=self.H, C1=self.C1, C2=self.C2)
        return self.N, self.W, self.H, self.C1, self.C2, input, conv1, conv2, conv3

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    N, W, H, C1, C2, input, conv1, conv2, conv3 = size.init()

    input_ref = np.copy(input)
    conv1_ref = np.copy(conv1)
    conv2_ref = np.copy(conv2)
    conv3_ref = np.copy(conv3)
    sdfg(N=N, W=W, H=H, C1=C1, C2=C2, input=input_ref, conv1=conv1_ref, conv2=conv2_ref, conv3=conv3_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    input2 = np.copy(input)
    conv12 = np.copy(conv1)
    conv22 = np.copy(conv2)
    conv32 = np.copy(conv3)
    sdfg_copy(N=N, W=W, H=H, C1=C1, C2=C2, _input=input2, _conv1=conv12, _conv2=conv22, _conv3=conv32)

    diff_input = np.sqrt(np.mean((input2 - input_ref)**2))
    diff_conv1 = np.sqrt(np.mean((conv12 - conv1_ref)**2))
    diff_conv2 = np.sqrt(np.mean((conv22 - conv2_ref)**2))
    diff_conv3 = np.sqrt(np.mean((conv32 - conv3_ref)**2))
    diff = max(np.max(np.abs(input2 - input_ref)), np.max(np.abs(conv12 - conv1_ref)), np.max(np.abs(conv22 - conv2_ref)), np.max(np.abs(conv32 - conv3_ref)))
    print("RMS Difference INPUT:", diff_input)
    print("RMS Difference CONV1:", diff_conv1)
    print("RMS Difference CONV2:", diff_conv2)
    print("RMS Difference CONV3:", diff_conv3)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (N={model_size.N}, W={model_size.W}, H={model_size.H}, C1={model_size.C1}, C2={model_size.C2})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()
